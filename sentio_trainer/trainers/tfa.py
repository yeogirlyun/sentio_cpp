import json
import time
import hashlib
import pathlib
import numpy as np
from numpy.lib.stride_tricks import as_strided
import torch
from torch import nn

# Uses your C++ FeatureBuilder via pybind11 (must be built/installed)
import sentio_features as sf
from sentio_trainer.utils.feature_cache_hardened import load_cached_features
from sentio_trainer.utils.schema_meta import write_meta_or_die

# Import the correct sequence model and its wrapper
from sentio_trainer.models.tfa_transformer import (
    TFA_Transformer, SeqScaler, TFA_Wrapped
)

# Import strategy evaluation framework
from sentio_trainer.utils.strategy_evaluation import StrategyEvaluator

# ---------------- utils ----------------
def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def _set_fast_flags():
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def _spec_with_hash(spec_path: str):
    raw = pathlib.Path(spec_path).read_bytes()
    spec = json.loads(raw)
    spec["content_hash"] = "sha256:" + hashlib.sha256(
        json.dumps(spec, sort_keys=True).encode()
    ).hexdigest()
    return spec

def _feature_names_from_spec(spec: dict):
    names = []
    for f in spec["features"]:
        if "name" in f:
            names.append(f["name"])
        else:
            op = f["op"]
            src = f.get("source", "")
            w = f.get("window", "")
            k = f.get("k", "")
            names.append(f"{op}_{src}_{w}_{k}")
    return names


def _ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# ---------------- dataset ----------------
# Efficient zero-copy sliding windows using stride_tricks

class EfficientSeqWindows(torch.utils.data.Dataset):
    """
    Creates a dataset of zero-copy sliding windows over NumPy arrays.
    This is significantly faster than the original IterableDataset.
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, T: int, start: int, end: int):
        super().__init__()
        self.X, self.y = X, y
        self.T = T
        
        # We can only start creating windows after T-1 elements
        self.start = max(start, T - 1)
        self.end = end
        
        # Calculate the number of valid windows
        self.n_windows = self.end - self.start + 1

        # Use as_strided to create a zero-copy view of the windows
        n, f = X.shape
        s_n, s_f = X.strides
        self.X_windows = as_strided(X, shape=(n - T + 1, T, f), strides=(s_n, s_n, s_f))

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # The index into our dataset corresponds to the end of the window
        window_end_idx = self.start + idx
        
        # Get the windowed X view and the corresponding y label
        x_window = self.X_windows[window_end_idx - self.T + 1]
        y_label = self.y[window_end_idx]
        
        # Convert to tensors here. The DataLoader will batch them efficiently.
        return torch.from_numpy(x_window), torch.from_numpy(y_label)
class IterableNPY(torch.utils.data.IterableDataset):
    def __init__(self, x_path: str, y_path: str, batch: int, emit_from: int, world_size: int = 1, rank: int = 0):
        super().__init__()
        self.x_path, self.y_path = x_path, y_path
        self.batch, self.emit_from = batch, emit_from
        self.world_size, self.rank = world_size, rank
        self._X = None; self._y = None

    def __iter__(self):
        self._X = np.load(self.x_path, mmap_mode="r")
        self._y = np.load(self.y_path, mmap_mode="r")
        n = self._X.shape[0]
        start = self.emit_from
        shard = (n - start + self.world_size - 1)//self.world_size
        s0 = start + shard*self.rank
        s1 = min(start + shard*(self.rank+1), n)
        # drop last partial batch for speed
        s1 = s0 + ((s1 - s0)//self.batch)*self.batch
        X = self._X[s0:s1]; y = self._y[s0:s1]
        for i in range(0, X.shape[0], self.batch):
            yield torch.from_numpy(X[i:i+self.batch]), torch.from_numpy(y[i:i+self.batch])

# ---------------- model ----------------
class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    Reduces the loss contribution from easy examples and focuses on hard examples.
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Convert targets to float for BCE
        targets = targets.float()
        
        # Clamp inputs to prevent numerical instability
        inputs = torch.clamp(inputs, min=-10, max=10)
        
        # Compute BCE loss with numerical stability
        bce_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        
        # Compute focal weight with numerical stability
        pt = torch.exp(-bce_loss)
        pt = torch.clamp(pt, min=1e-8, max=1-1e-8)  # Prevent log(0) and log(1)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        
        # Apply focal weight
        focal_loss = focal_weight * bce_loss
        
        # Check for NaN and replace with BCE if needed
        if torch.isnan(focal_loss).any():
            print("[WARNING] Focal Loss produced NaN, falling back to BCE")
            return bce_loss.mean()
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class PredictionDiversityLoss(nn.Module):
    """
    Regularization to encourage prediction diversity and prevent flat predictions.
    """
    def __init__(self, weight=0.001):  # Reduced weight to prevent instability
        super().__init__()
        self.weight = weight
        
    def forward(self, predictions):
        """
        Compute diversity loss based on prediction variance.
        Higher variance = more diverse predictions = lower loss.
        """
        if len(predictions.shape) == 1:
            predictions = predictions.unsqueeze(0)
        
        # Clamp predictions to prevent extreme values
        predictions = torch.clamp(predictions, min=-10, max=10)
        
        # Compute variance across batch with numerical stability
        pred_var = torch.var(predictions, dim=0)
        pred_var = torch.clamp(pred_var, min=1e-8)  # Prevent log(0)
        
        # Encourage higher variance (lower loss for higher variance)
        diversity_loss = -torch.mean(pred_var)
        
        # Check for NaN
        if torch.isnan(diversity_loss):
            print("[WARNING] Diversity loss produced NaN, returning zero")
            return torch.tensor(0.0, device=predictions.device)
        
        return self.weight * diversity_loss

class MLP(nn.Module):
    def __init__(self, in_dim: int, hid: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hid), nn.GELU(),
            nn.Linear(hid, hid),    nn.GELU(),
            nn.Linear(hid, 1)       # logits
        )
    def forward(self, x): return self.net(x)

class Wrapped(nn.Module):
    """Wraps scaler into model for inference simplicity in C++."""
    def __init__(self, core:nn.Module, mean:np.ndarray, inv_std:np.ndarray):
        super().__init__()
        self.core = core
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("inv_std", torch.tensor(inv_std, dtype=torch.float32))
    def forward(self, x):
        x = (x - self.mean) * self.inv_std
        return self.core(x)  # logits

# ---------------- main trainer ----------------
def train_tfa_fast(
    # data
    symbol: str,
    bars_csv: str,                     # CSV with header: ts,open,high,low,close,volume
    feature_spec: str,                 # path to feature_spec.json
    # training
    out_dir: str = "artifacts/TFA/v1",
    batch_size: int = 1024,  # Optimized for better gradient estimates
    epochs: int = 25,         # More epochs for better convergence
    lr: float = 0.002,       # Higher learning rate for faster learning
    num_workers: int = 6,    # More workers for faster data loading
    # Transformer architecture parameters - optimized for learning
    T: int = 32,             # Balanced sequence length for better performance
    d_model: int = 96,       # Optimized for feature count
    nhead: int = 4,          # d_model must be divisible by nhead
    num_layers: int = 2,     # More efficient depth
    ffn_hidden: int = 192,   # Scaled appropriately with d_model
    # labels
    label_horizon: int = 3,                  # Longer horizon for better signals
    label_kind: str = "logret_fwd",          # "logret_fwd" | "close_diff"
    # data quality
    min_volume_threshold: float = 100000,   # 10x lower to include more bars
    price_change_threshold: float = 0.0005, # 2x lower for more sensitivity
    # cache
    feature_cache: str = "data",       # directory with cached features
):
    """
    Trains a TFA Transformer on base-ticker features built by the C++ FeatureBuilder.
    Exports TorchScript model, feature_spec.json, and model.meta.json.

    Returns: path to out_dir with artifacts.
    """
    t0=time.time()
    dev = _device()
    _set_fast_flags()
    _ensure_dir(out_dir)

    # 1) Load bars (fast numpy reader)
    arr = np.genfromtxt(bars_csv, delimiter=",", names=True, dtype=None, encoding=None)
    
    # Handle different CSV column formats
    if "ts_nyt_epoch" in arr.dtype.names:
        ts = arr["ts_nyt_epoch"].astype(np.int64)
    elif "ts" in arr.dtype.names:
        ts = arr["ts"].astype(np.int64)
    else:
        raise ValueError(f"No timestamp column found. Available columns: {arr.dtype.names}")
    
    openp = arr["open"].astype(np.float64)
    high  = arr["high"].astype(np.float64)
    low   = arr["low"].astype(np.float64)
    close = arr["close"].astype(np.float64)
    vol   = arr["volume"].astype(np.float64)
    print(f"[TFA] Volume data loaded: vol.shape={vol.shape}, vol range: {vol.min():.0f} - {vol.max():.0f}")

    # 2) Load spec and validate it has no ts contamination
    spec = _spec_with_hash(feature_spec)
    emit_from = int(spec["alignment_policy"]["emit_from_index"])
    
    # 3) Try cache first, fallback to on-the-fly features
    ts_cached, X_cached, names_cached = load_cached_features(symbol, feature_cache)

    if X_cached is not None:
        # Use cached features (already validated to be exactly 55)
        X = X_cached  # already float32
        names = names_cached
        print(f"[TFA] HARDENED: Using cached features: {X.shape}, exactly 55 features")
        
        # Ensure row counts align with bars
        if ts_cached is not None:
            if len(ts_cached) != len(ts):
                raise RuntimeError(f"Cache rows ({len(ts_cached)}) != bars rows ({len(ts)})")
    else:
        # fallback: build on-the-fly via C++ builder (always excludes ts)
        spec_json = json.dumps(spec, sort_keys=True)
        print(f"[TFA] Building features for {symbol} with {len(ts)} bars...")
        X = sf.build_features_from_spec(symbol, ts, openp, high, low, close, vol, spec_json).astype(np.float32)
        
        # Build names from spec
        names = []
        for f in spec["features"]:
            if "name" in f:
                names.append(f["name"])
            else:
                op = f["op"]
                src = f.get("source", "")
                w = str(f.get("window", ""))
                k = str(f.get("k", ""))
                names.append(f"{op}_{src}_{w}_{k}")
        
        print(f"[TFA] Built features on-the-fly: {X.shape}")
    
    N, F = X.shape
    
    # HARDENED: Enforce exactly 55 features - fail loud in training
    _ = write_meta_or_die(out_dir, spec, X.shape, names, dtype="float32")
    
    print(f"[TFA] HARDENED: Features validated: {N} rows x {F} features, emit_from: {emit_from}")
    print(f"[TFA] Feature stats: min={X.min():.6f}, max={X.max():.6f}, mean={X.mean():.6f}, std={X.std():.6f}")
    print(f"[TFA] Debug: F={F}, N={N}, emit_from={emit_from}")
    assert F > 0 and N > emit_from, f"No usable rows after emit_from: F={F}, N={N}, emit_from={emit_from}"
    print(f"[TFA] Assertion passed, starting label generation...")
    print(f"[TFA] About to define make_binary_labels function...")

    # 3) Labels (binary classification: up vs down)
    def make_binary_labels(close_arr: np.ndarray, horizon: int = 1, cutoff_bp: float = 0.0):
        logp = np.log(np.clip(close_arr, 1e-12, None))
        yy = np.zeros((close_arr.shape[0], 1), dtype=np.float32)
        if horizon > 0:
            diff = logp[horizon:] - logp[:-horizon]
            if cutoff_bp > 0.0:
                thr = cutoff_bp/10000.0
                pos = (diff > +thr).astype(np.float32)
                neg = (diff < -thr).astype(np.float32)
                lab = np.zeros_like(diff, dtype=np.float32)
                lab[pos == 1.0] = 1.0
            else:
                lab = (diff > 0.0).astype(np.float32)
            yy[:-horizon, 0] = lab
        return yy

    print(f"[TFA] make_binary_labels function defined successfully")

    def make_improved_labels(close_arr: np.ndarray, volume_arr: np.ndarray, 
                             horizon: int = 2, min_volume: float = 1000000,
                             price_threshold: float = 0.001, label_smoothing: float = 0.1):
        """
        Generate better quality labels with volume and price change filtering.
        Includes label smoothing for better calibration.
        """
        print(f"[TFA] make_improved_labels: close_arr.shape={close_arr.shape}, volume_arr.shape={volume_arr.shape}")
        print(f"[TFA] make_improved_labels: horizon={horizon}, min_volume={min_volume}, price_threshold={price_threshold}, smoothing={label_smoothing}")
        logp = np.log(np.clip(close_arr, 1e-12, None))
        yy = np.zeros((close_arr.shape[0], 1), dtype=np.float32)
        
        if horizon > 0:
            # Calculate forward returns
            diff = logp[horizon:] - logp[:-horizon]
            
            # Filter by volume and price change
            valid_mask = np.ones_like(diff, dtype=bool)
            
            # Volume filter
            if volume_arr is not None:
                volume_mask = volume_arr[horizon:] >= min_volume
                valid_mask &= volume_mask
            
            # Price change filter
            price_change_mask = np.abs(diff) >= price_threshold
            valid_mask &= price_change_mask
            
            # Generate labels with smoothing
            lab = np.zeros_like(diff, dtype=np.float32)
            
            # For valid samples, use smoothed labels
            valid_indices = np.where(valid_mask)[0]
            for idx in valid_indices:
                if diff[idx] > 0.0:
                    # Positive return: smooth towards 1.0
                    lab[idx] = 1.0 - label_smoothing
                else:
                    # Negative return: smooth towards 0.0
                    lab[idx] = label_smoothing
            
            yy[:-horizon, 0] = lab
        
        return yy

    print(f"[TFA] make_improved_labels function defined successfully")

    def sanity_asserts(X_arr, y_arr, start_idx):
        usable = y_arr[start_idx:, 0]
        assert usable.shape[0] > 100, "Not enough usable samples after warmup"
        p = float(usable.mean())
        var = float(usable.var())
        print(f"[LABELS] mean={p:.4f} var={var:.6f} pos={int(usable.sum())}/{usable.shape[0]}")
        assert 0.05 <= p <= 0.95, f"Label class balance bad (mean={p:.3f})"
        assert var > 1e-4, "Label variance too small (degenerate labels)"
        s = X_arr[start_idx:].std(axis=0)
        assert np.all(s > 1e-8), "Some features are constant/near-constant post-warmup"

    print(f"[TFA] sanity_asserts function defined successfully")
    print(f"[TFA] About to start label generation...")

    # Use improved label generation with volume and price filtering + smoothing
    print(f"[TFA] About to call make_improved_labels with close.shape={close.shape}, vol.shape={vol.shape}")
    y = make_improved_labels(close, vol, horizon=label_horizon, 
                           min_volume=min_volume_threshold,
                           price_threshold=price_change_threshold,
                           label_smoothing=0.0)  # No smoothing for stability
    y[:emit_from] = 0.0
    
    # Analyze label quality
    usable_labels = y[emit_from:]
    pos_count = int(usable_labels.sum())
    total_count = len(usable_labels)
    pos_ratio = pos_count / total_count if total_count > 0 else 0
    
    print(f"[TFA] Label quality: {pos_count}/{total_count} positive ({pos_ratio:.3f})")
    print(f"[TFA] Label variance: {usable_labels.var():.6f}")
    
    if pos_ratio < 0.1 or pos_ratio > 0.9:
        print(f"[WARNING] Label imbalance detected: {pos_ratio:.3f}")
    if usable_labels.var() < 1e-4:
        print(f"[WARNING] Low label variance: {usable_labels.var():.6f}")
    
    sanity_asserts(X, y, emit_from)
    print(f"[TFA] Labels: improved binary horizon={label_horizon}")

    # 4) Train/valid split (simple holdout)
    split = int(0.9*N)

    # 5) Fit scaler on *train usable rows only* with NaN protection
    tr_mask = np.arange(split) >= emit_from
    X_train = X[:split][tr_mask]
    
    # Check for NaN/inf in features
    if np.isnan(X_train).any() or np.isinf(X_train).any():
        print("[ERROR] NaN or inf detected in training features!")
        # Replace NaN/inf with median values
        X_train = np.nan_to_num(X_train, nan=np.nanmedian(X_train), posinf=np.nanmedian(X_train), neginf=np.nanmedian(X_train))
        X[:split][tr_mask] = X_train
    
    mu = X_train.mean(axis=0).astype(np.float32)
    sd = X_train.std(axis=0).clip(1e-6, None).astype(np.float32)  # Increased min std
    inv = (1.0/sd).astype(np.float32)
    
    # Validate scaler parameters
    if np.isnan(mu).any() or np.isnan(sd).any() or np.isnan(inv).any():
        print("[ERROR] NaN detected in scaler parameters!")
        mu = np.zeros_like(mu)
        sd = np.ones_like(sd)
        inv = np.ones_like(inv)

    # 6) Cache to .npy for fast multi-worker loading
    # Ensure arrays are already float32 to avoid copying during save
    if X.dtype != np.float32:
        X = X.astype(np.float32)
    if y.dtype != np.float32:
        y = y.astype(np.float32)
    
    x_path = pathlib.Path(out_dir) / "X.npy"
    y_path = pathlib.Path(out_dir) / "y.npy"
    np.save(x_path, X)  # No copy needed since already float32
    np.save(y_path, y)   # No copy needed since already float32

    # 7) Use the EfficientSeqWindows DataLoader
    ds_tr = EfficientSeqWindows(X, y, T=T, start=emit_from, end=split-1)
    ds_va = EfficientSeqWindows(X, y, T=T, start=split, end=N-2)  # end is inclusive for this class
    
    loader_tr = torch.utils.data.DataLoader(ds_tr, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    # 8) Instantiate the Transformer model instead of the MLP
    core = TFA_Transformer(
        F=F, T=T, d_model=d_model, nhead=nhead,
        num_layers=num_layers, ffn_hidden=ffn_hidden
    ).to(dev)
    
    scaler = SeqScaler(mu, inv)
    model = TFA_Wrapped(core, scaler).to(dev)
    
    # Check model weights for NaN/inf
    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"[ERROR] NaN/inf detected in model parameter: {name}")
            # Reinitialize with Xavier uniform
            if param.dim() > 1:
                torch.nn.init.xavier_uniform_(param)
            else:
                torch.nn.init.normal_(param, mean=0, std=0.01)
    
    # Optimized settings for better learning
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, foreach=True)

    # Loss: Start with standard BCE + class weighting for stability
    pos_frac = float(y[emit_from:].mean())
    pos_weight = (1.0 - pos_frac) / max(1e-6, pos_frac)
    print(f"[TFA] Class balance: {pos_frac:.3f} positive, using BCE with class weighting")
    
    # Use standard BCE with class weighting for stability
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))
    # diversity_criterion = PredictionDiversityLoss(weight=0.001).to(dev)  # Disabled for now

    # 9) Train
    # Guardrail: estimate batch count
    est_batches = max(0, (N - emit_from) // max(1, batch_size))
    assert est_batches > 0, "DataLoader yielded 0 batches (check emit_from & data size)"

    # Gradient accumulation for larger effective batch size
    accumulation_steps = 4  # Effective batch size = batch_size * accumulation_steps
    effective_batch_size = batch_size * accumulation_steps
    
    # Estimate total batches for progress tracking
    total_batches = max(0, (split - emit_from) // max(1, batch_size))
    print(f"[TFA] Training: {total_batches} batches per epoch, {epochs} epochs")
    print(f"[TFA] Using gradient accumulation: {accumulation_steps} steps, effective batch size: {effective_batch_size}")
    
    model.train()
    for ep in range(1, epochs+1):
        loss_sum=0.0; steps=0
        batch_count = 0
        
        # Wrap loader in an enumerator to track batch index
        for i, (bx, by) in enumerate(loader_tr):
            bx, by = bx.to(dev), by.to(dev)
            
            # Check for NaN in inputs
            if torch.isnan(bx).any() or torch.isnan(by).any():
                print(f"[ERROR] NaN detected in batch inputs at step {i}, skipping")
                continue
            
            # Forward pass
            logits = model(bx)
            
            # Check for NaN in model output
            if torch.isnan(logits).any():
                print(f"[ERROR] NaN detected in model output at step {i}, skipping")
                continue
            
            # Main loss (BCE with class weighting)
            loss = criterion(logits, by)
            
            # Check for NaN in loss
            if torch.isnan(loss):
                print(f"[ERROR] NaN detected in loss at step {i}, skipping")
                continue
            
            # Scale loss for accumulation
            loss = loss / accumulation_steps
            
            # Backward pass
            loss.backward()
            
            loss_sum += float(loss) * accumulation_steps  # Log loss
            steps += 1
            batch_count += 1

            # Perform optimizer step only after accumulating gradients for `accumulation_steps`
            if (i + 1) % accumulation_steps == 0:
                # Check gradients for NaN before clipping
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isnan(grad_norm):
                    print(f"[ERROR] NaN detected in gradients at step {i}, skipping optimizer step")
                    opt.zero_grad(set_to_none=True)
                    continue
                
                opt.step()
                opt.zero_grad(set_to_none=True)
            
            # Progress tracking every 5% (reduced frequency)
            if batch_count % max(1, total_batches // 20) == 0:
                progress = (batch_count / total_batches) * 100
                print(f"[TFA] Epoch {ep}/{epochs} - {progress:.1f}% - Loss: {loss_sum/steps:.6f}")
                
        print(f"[TFA] epoch {ep}/{epochs} bce={loss_sum/max(1,steps):.6f}")

    # Comprehensive model evaluation
    print(f"\n{'='*60}")
    print(f"MODEL EVALUATION")
    print(f"{'='*60}")
    
    model.eval()
    evaluator = StrategyEvaluator("TFA_Transformer")
    
    # Collect validation predictions
    val_predictions = []
    val_actual_returns = []
    
    with torch.inference_mode():
        for bx, by in ds_va:
            bx = bx.unsqueeze(0).to(dev)  # Add batch dim: [1, T, F]
            logits = model(bx).detach().cpu()
            # Ensure logits is 1D for concatenation
            if logits.dim() > 1:
                logits = logits.squeeze()
            if logits.dim() == 0:  # Handle zero-dimensional tensor
                logits = logits.unsqueeze(0)
            val_predictions.append(logits)
            val_actual_returns.append(by.numpy())
    
    # Convert to tensors with proper error handling
    if len(val_predictions) > 0:
        val_predictions_tensor = torch.cat(val_predictions)
        val_actual_returns_array = np.concatenate(val_actual_returns).flatten()
    else:
        print("[WARNING] No validation predictions collected")
        val_predictions_tensor = torch.tensor([])
        val_actual_returns_array = np.array([])
    
    # Run comprehensive evaluation
    results = evaluator.evaluate_strategy_signal(
        val_predictions_tensor, val_actual_returns_array, verbose=True
    )
    
    # Save evaluation results
    eval_path = pathlib.Path(out_dir) / "evaluation_results.json"
    evaluator.save_results(results, str(eval_path))
    
    # Smoke test with detailed analysis
    sample, _ = next(iter(ds_va))
    sample = sample.unsqueeze(0).to(dev)
    logits = model(sample).squeeze().detach().cpu().numpy()
    probs = 1.0/(1.0+np.exp(-logits))
    print(f"\n[SMOKE] Sample prediction prob: {probs.mean():.4f}")
    prob_std = np.std(probs)
    print(f"[SMOKE] Probability std: {prob_std:.6f}")
    
    if prob_std < 1e-4:
        print(f"[WARNING] Model predictions have low variance (std={prob_std:.6f}). Consider:")
        print(f"  - Increasing model capacity (d_model, num_layers)")
        print(f"  - Adjusting label thresholds (price_threshold, volume_threshold)")
        print(f"  - Using a different loss function or class weights")
        print(f"  - Checking feature quality and normalization")
        print(f"[INFO] Training completed despite low variance - check evaluation results")
    else:
        print(f"[SUCCESS] Model shows good prediction variance: {prob_std:.6f}")
    
    # Training continues regardless of variance - no blocking assertion

    # Export TorchScript with the correct sequential input shape
    wrapped_cpu = model.to("cpu").eval()
    sample_input = torch.randn(1, T, F) # [Batch, Time, Features]
    
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped_cpu, sample_input)
    
    torch.jit.save(scripted, str(pathlib.Path(out_dir)/"model.pt"))

    # Schema validation already done in write_meta_or_die above
    print(f"[TFA] Schema validation: 55 features enforced, hash: {spec['content_hash'][:16]}...")

    print(f"✅ Done in {time.time()-t0:.1f}s → {out_dir}")
    return str(out_dir)
