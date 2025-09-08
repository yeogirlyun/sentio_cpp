import os, json, time, hashlib, pathlib
import numpy as np
import torch
from torch import nn

# Uses your C++ FeatureBuilder via pybind11 (must be built/installed)
import sentio_features as sf
from sentio_trainer.utils.feature_cache_hardened import load_cached_features
from sentio_trainer.utils.schema_meta import write_meta_or_die, feature_names_from_spec

# ---------------- utils ----------------
def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def _set_fast_flags():
    try: torch.set_float32_matmul_precision("high")
    except Exception: pass
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

def _spec_with_hash(spec_path:str):
    raw = pathlib.Path(spec_path).read_bytes()
    spec = json.loads(raw)
    spec["content_hash"] = "sha256:" + hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()
    return spec

def _feature_names_from_spec(spec:dict):
    names=[]
    for f in spec["features"]:
        if "name" in f:
            names.append(f["name"])
        else:
            op = f["op"]; src=f.get("source","")
            w  = f.get("window",""); k=f.get("k","")
            names.append(f"{op}_{src}_{w}_{k}")
    return names

def _ensure_dir(p): pathlib.Path(p).mkdir(parents=True, exist_ok=True)

# ---------------- dataset ----------------
class IterableNPY(torch.utils.data.IterableDataset):
    def __init__(self, x_path:str, y_path:str, batch:int, emit_from:int, world_size:int=1, rank:int=0):
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
class MLP(nn.Module):
    def __init__(self, in_dim:int, hid:int=128):
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
    batch_size: int = 8192,
    epochs: int = 10,
    lr: float = 1e-3,
    num_workers: int = max(1, (os.cpu_count() or 2) - 1),
    prefetch_factor: int = 4,
    hidden: int = 128,
    # labels
    label_horizon: int = 1,            # forward bars for label
    label_kind: str = "logret_fwd",    # "logret_fwd" | "close_diff"
    # cache
    feature_cache: str = "data",       # directory with cached features
):
    """
    Trains a simple MLP on base-ticker features built by the C++ FeatureBuilder.
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
    assert F > 0 and N > emit_from, "No usable rows after emit_from"

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

    y = make_binary_labels(close, horizon=label_horizon, cutoff_bp=0.0)
    y[:emit_from] = 0.0
    sanity_asserts(X, y, emit_from)
    print(f"[TFA] Labels: binary horizon={label_horizon}")

    # 4) Train/valid split (simple holdout)
    split = int(0.9*N)
    X_tr, X_va = X[:split], X[split:]
    y_tr, y_va = y[:split], y[split:]

    # 5) Fit scaler on *train usable rows only*
    tr_mask = np.arange(X_tr.shape[0]) >= emit_from
    mu = X_tr[tr_mask].mean(axis=0).astype(np.float32)
    sd = X_tr[tr_mask].std(axis=0).clip(1e-12, None).astype(np.float32)
    inv = (1.0/sd).astype(np.float32)

    # 6) Cache to .npy for fast multi-worker loading
    x_path = pathlib.Path(out_dir) / "X.npy"
    y_path = pathlib.Path(out_dir) / "y.npy"
    np.save(x_path, X.astype(np.float32, copy=False))
    np.save(y_path, y.astype(np.float32, copy=False))

    # 7) DataLoader (IterableDataset batches contiguous mmapped blocks)
    ds = IterableNPY(str(x_path), str(y_path), batch=batch_size, emit_from=emit_from, world_size=1, rank=0)
    pin = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        ds, batch_size=None, num_workers=num_workers,
        persistent_workers=(num_workers>0),
        prefetch_factor=prefetch_factor if num_workers>0 else None,
        pin_memory=pin,
    )

    # 8) Model / opt
    model = MLP(F, hid=hidden).to(dev)
    # Skip torch.compile for TorchScript compatibility
    # try:
    #     model = torch.compile(model, fullgraph=True, mode="max-autotune")
    # except Exception:
    #     pass
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    # Loss: BCEWithLogits with class weighting (avoid collapse)
    pos_frac = float(y[emit_from:].mean())
    pos_weight = (1.0 - pos_frac) / max(1e-6, pos_frac)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=dev))

    # 9) Train
    # Guardrail: estimate batch count
    est_batches = max(0, (N - emit_from) // max(1, batch_size))
    assert est_batches > 0, "DataLoader yielded 0 batches (check emit_from & data size)"

    model.train()
    for ep in range(1, epochs+1):
        loss_sum=0.0; steps=0
        for bx_cpu, by_cpu in loader:
            bx = bx_cpu.to(dev, non_blocking=True)
            by = by_cpu.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward(); opt.step()
            loss_sum += float(loss); steps += 1
        print(f"[TFA] epoch {ep}/{epochs}  bce={loss_sum/max(1,steps):.6f}")

    # 9b) Smoke test: ensure non-flat predictions
    model.eval()
    with torch.inference_mode():
        sample = torch.from_numpy(X[emit_from:emit_from+2048]).to(dev)
        logits = model(sample).squeeze(1).detach().cpu().numpy()
        probs = 1.0/(1.0+np.exp(-logits))
        print(f"[SMOKE] logits mean={logits.mean():.4f} std={logits.std():.4f} | probs mean={probs.mean():.4f} std={probs.std():.4f}")
        assert probs.std() > 0.005, "Model is too flat; check labels/loss/learning"

    # 10) Export TorchScript (bake scaler)
    wrapped = Wrapped(model, mu, inv).to("cpu").eval()
    
    # Use tracing instead of scripting for better compatibility
    sample_input = torch.from_numpy(X[max(emit_from,0):max(emit_from,0)+1]).float()
    print(f"[TFA] Exporting TorchScript with sample input shape: {sample_input.shape}")
    
    with torch.no_grad():
        scripted = torch.jit.trace(wrapped, sample_input)
    
    torch.jit.save(scripted, str(pathlib.Path(out_dir)/"model.pt"))

    # Schema validation already done in write_meta_or_die above
    print(f"[TFA] Schema validation: 55 features enforced, hash: {spec['content_hash'][:16]}...")

    print(f"✅ Done in {time.time()-t0:.1f}s → {out_dir}")
    return str(out_dir)
