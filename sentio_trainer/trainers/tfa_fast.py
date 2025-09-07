import os, json, time, hashlib, pathlib
import numpy as np
import torch
from torch import nn

# Uses your C++ FeatureBuilder via pybind11 (must be built/installed)
import sentio_features as sf

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
    ts    = arr["ts"].astype(np.int64)
    openp = arr["open"].astype(np.float64)
    high  = arr["high"].astype(np.float64)
    low   = arr["low"].astype(np.float64)
    close = arr["close"].astype(np.float64)
    vol   = arr["volume"].astype(np.float64)

    # 2) Spec + features (built by *C++* code via pybind → parity with C++)
    spec = _spec_with_hash(feature_spec)
    spec_json = json.dumps(spec, sort_keys=True)
    X = sf.build_features_from_spec(symbol, ts, openp, high, low, close, vol, spec_json).astype(np.float32)
    N, F = X.shape
    emit_from = int(spec["alignment_policy"]["emit_from_index"])
    assert F > 0 and N > emit_from, "No usable rows after emit_from"

    # 3) Labels (vectorized)
    y = np.zeros((N,1), dtype=np.float32)
    if label_kind == "logret_fwd":
        logp = np.log(np.clip(close, 1e-12, None))
        shift = label_horizon
        if shift > 0:
            y[:-shift, 0] = (logp[shift:] - logp[:-shift]).astype(np.float32)
    elif label_kind == "close_diff":
        shift = label_horizon
        y[:-shift, 0] = (close[shift:] - close[:-shift]).astype(np.float32)
    else:
        raise ValueError(f"Unknown label_kind: {label_kind}")
    y[:emit_from] = 0.0

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
    try:
        model = torch.compile(model, fullgraph=True, mode="max-autotune")
    except Exception:
        pass
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    # 9) Train
    model.train()
    for ep in range(1, epochs+1):
        loss_sum=0.0; steps=0
        for bx_cpu, by_cpu in loader:
            bx = bx_cpu.to(dev, non_blocking=True)
            by = by_cpu.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            pred = model(bx)
            loss = torch.nn.functional.mse_loss(pred, by)
            loss.backward(); opt.step()
            loss_sum += float(loss); steps += 1
        print(f"[TFA] epoch {ep}/{epochs}  loss={loss_sum/max(1,steps):.6f}")

    # 10) Export TorchScript (bake scaler)
    wrapped = Wrapped(model, mu, inv).to("cpu").eval()
    try:
        scripted = torch.jit.script(wrapped)
    except Exception:
        scripted = torch.jit.trace(wrapped, torch.from_numpy(X[max(emit_from,0):max(emit_from,0)+1]).float())
    torch.jit.save(scripted, str(pathlib.Path(out_dir)/"model.pt"))

    # 11) Save spec + meta
    json.dump(spec, open(pathlib.Path(out_dir)/"feature_spec.json","w"), indent=2)
    names = _feature_names_from_spec(spec)
    meta = {
        "schema_version":"1.0",
        "saved_at": int(time.time()),
        "framework":"torchscript",
        "expects":{
            "input_dim": int(F),
            "feature_names": names,
            "spec_hash": spec["content_hash"],
            "emit_from": int(emit_from),
            "pad_value": float(spec["alignment_policy"]["pad_value"]),
            "dtype":"float32",
            "output":"logit"  # C++ will sigmoid
        }
    }
    json.dump(meta, open(pathlib.Path(out_dir)/"model.meta.json","w"), indent=2)

    print(f"✅ Done in {time.time()-t0:.1f}s → {out_dir}")
    return str(out_dir)
