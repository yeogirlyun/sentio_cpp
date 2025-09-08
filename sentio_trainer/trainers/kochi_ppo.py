import json
import time
import pathlib

import numpy as np
import torch
from torch import nn
from sentio_trainer.utils.feature_cache import load_kochi_cached_features


def _device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _ensure_dir(p):
    pathlib.Path(p).mkdir(parents=True, exist_ok=True)


def _load_bars_csv(bars_csv: str):
    # numpy type stubs can be too strict here; ignore signature check
    arr = np.genfromtxt(  # type: ignore
        bars_csv,
        delimiter=",",
        names=True,
        dtype=None,
        encoding=None,
    )
    if "ts_nyt_epoch" in arr.dtype.names:
        ts = arr["ts_nyt_epoch"].astype(np.int64)
    elif "ts" in arr.dtype.names:
        ts = arr["ts"].astype(np.int64)
    else:
        raise ValueError(
            f"No timestamp column found. Available columns: {arr.dtype.names}"
        )
    close = arr["close"].astype(np.float64)
    return ts, close


def _build_sequences(X: np.ndarray, T: int) -> np.ndarray:
    N, _ = X.shape
    if N < T:
        raise ValueError(f"Not enough rows ({N}) for window {T}")
    out = np.zeros((N - T + 1, T, X.shape[1]), dtype=np.float32)
    for i in range(T, N + 1):
        out[i - T] = X[i - T:i]
    return out


def _make_labels(close: np.ndarray, T: int, thr_bp: float = 0.0) -> np.ndarray:
    # 3-class: SELL (0), HOLD (1), BUY (2)
    y = np.zeros((close.shape[0] - T + 1,), dtype=np.int64)
    fwd = np.zeros_like(y, dtype=np.float32)
    fwd[:-1] = (close[T:] / np.clip(close[T - 1:-1], 1e-12, None)) - 1.0
    if thr_bp > 0:
        thr = thr_bp / 10000.0
        y[fwd > +thr] = 2
        y[np.abs(fwd) <= thr] = 1
        y[fwd < -thr] = 0
    else:
        y[fwd > 0] = 2
        y[fwd == 0] = 1
        y[fwd < 0] = 0
    return y


class TemporalCNN(nn.Module):
    def __init__(self, T: int, F: int, hid: int = 128, num_classes: int = 3):
        super().__init__()
        # Input B,T,F → B,F,T for conv1d
        self.conv = nn.Sequential(
            nn.Conv1d(F, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * T, hid),
            nn.ReLU(),
            nn.Linear(hid, num_classes),
        )

    def forward(self, x):
        # x: [B, T, F]
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        return self.proj(x)


class WrappedNoScaler(nn.Module):
    """
    TorchScript-friendly wrapper.

    No scaler inside; C++ applies normalization per metadata. Expects input
    shaped [B, T*F] flattened. Will reshape to [B, T, F].
    """

    def __init__(self, core: nn.Module, T: int, F: int):
        super().__init__()
        self.core = core
        self.T = T
        self.F = F

    def forward(self, x):
        # x: [B, T*F]
        B = x.shape[0]
        x = x.view(B, self.T, self.F)
        logits = self.core(x)
        return logits


def train_kochi_ppo(
    symbol: str,
    bars_csv: str,
    out_dir: str = "artifacts/KochiPPO/v1",
    window_size: int = 20,
    epochs: int = 8,
    batch_size: int = 1024,
    lr: float = 1e-3,
):
    t0 = time.time()
    dev = _device()
    _ensure_dir(out_dir)

    # Require cache: search bars parent, then its parent
    bars_path = pathlib.Path(bars_csv)
    primary_dir = bars_path.parent
    fallback_dir = primary_dir.parent
    ts, X, cols = load_kochi_cached_features(symbol, primary_dir)
    if X is None and fallback_dir and fallback_dir.exists():
        ts, X, cols = load_kochi_cached_features(symbol, fallback_dir)
    if X is None:
        raise RuntimeError("Kochi trainer requires precomputed feature cache. Run generate_kochi_feature_cache.py first.")

    _, close = _load_bars_csv(bars_csv)
    N, F = X.shape

    # Train/valid split
    split = int(0.9 * N)

    # Scaler on train usable rows
    mu = X[:split].mean(axis=0).astype(np.float32)
    sd = X[:split].std(axis=0).clip(1e-6, None).astype(np.float32)

    # Build sequences and labels
    Xs = _build_sequences((X - mu) / sd, window_size)
    ys = _make_labels(close, window_size, thr_bp=0.0)
    Xs_tr = Xs[: split - window_size + 1]
    Xs_va = Xs[split - window_size + 1:]
    ys_tr = ys[: split - window_size + 1]
    ys_va = ys[split - window_size + 1:]

    # Torch loaders
    tr_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xs_tr),
        torch.from_numpy(ys_tr),
    )
    va_ds = torch.utils.data.TensorDataset(
        torch.from_numpy(Xs_va),
        torch.from_numpy(ys_va),
    )
    tr_ld = torch.utils.data.DataLoader(
        tr_ds,
        batch_size=batch_size,
        shuffle=True,
    )
    va_ld = torch.utils.data.DataLoader(
        va_ds,
        batch_size=batch_size,
        shuffle=False,
    )

    # Model
    core = TemporalCNN(T=window_size, F=F).to(dev)
    opt = torch.optim.AdamW(core.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    core.train()
    for ep in range(1, epochs + 1):
        loss_sum = 0.0
        steps = 0
        for xb, yb in tr_ld:
            xb = xb.to(dev, non_blocking=True)
            yb = yb.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits = core(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            loss_sum += float(loss)
            steps += 1
        # simple valid
        core.eval()
        with torch.inference_mode():
            va_acc = 0.0
            va_n = 0
            for xb, yb in va_ld:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = core(xb).argmax(dim=1)
                va_acc += (pred == yb).float().sum().item()
                va_n += yb.numel()
        core.train()
        print(
            f"[KochiPPO] epoch {ep}/{epochs}  loss={loss_sum/max(1,steps):.5f}  "
            f"va_acc={va_acc/max(1,va_n):.4f}"
        )

    # Export TorchScript
    wrapped = WrappedNoScaler(core.cpu().eval(), window_size, F)
    sample = torch.from_numpy(
        Xs_tr[:1].reshape(1, window_size * F)
    ).float()
    with torch.no_grad():
        ts_mod = torch.jit.trace(wrapped, sample)
    torch.jit.save(ts_mod, str(pathlib.Path(out_dir) / "model.pt"))

    # Write minimal metadata.json
    meta = {
        "feature_names": cols,
        "mean": [float(x) for x in mu],
        "std": [float(x) for x in sd],
        "clip": [-5.0, 5.0],
        "actions": ["SELL", "HOLD", "BUY"],
        "seq_len": int(window_size),
        "input_layout": "BTF",
        "format": "torchscript",
    }
    json.dump(meta, open(pathlib.Path(out_dir) / "metadata.json", "w"), indent=2)

    print(f"✅ Kochi trainer done in {time.time()-t0:.1f}s → {out_dir}")
    return str(out_dir)


