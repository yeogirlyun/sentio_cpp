import os, json, time, hashlib, pathlib
import numpy as np, torch
from torch import nn
from torch.utils.data import IterableDataset, DataLoader

from sentio_trainer.models.tfa_transformer import TFA_Transformer, SeqScaler, TFA_Wrapped
from sentio_trainer.utils.feature_cache_hardened import load_cached_features


def _device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():         return torch.device("cuda")
    return torch.device("cpu")

def _hash(d): return "sha256:" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def _spec_with_hash(path):
    spec = json.loads(pathlib.Path(path).read_text())
    spec["content_hash"] = _hash(spec)
    return spec

def _feature_names_from_spec(spec):
    names=[]
    for f in spec["features"]:
        name = f.get("name") or f'{f["op"]}_{f.get("source","")}_{f.get("window","")}_{f.get("k","")}'
        if name in ("ts","timestamp","bar_index"): raise ValueError("Non-feature column in spec")
        names.append(name)
    return names

class SeqWindows(IterableDataset):
    def __init__(self, X: np.ndarray, close: np.ndarray, T: int, emit_from: int,
                 batch: int, start: int, end: int, cutoff_bp: float = 0.0):
        super().__init__()
        self.X, self.close = X, close
        self.T, self.emit_from, self.batch = T, emit_from, batch
        self.start = max(emit_from, T-1, start)
        self.end   = min(end, X.shape[0]-2)
        self.cut   = cutoff_bp/10000.0

    def __iter__(self):
        B = self.batch
        i = self.start
        while i <= self.end:
            j = min(self.end+1, i + B)
            L = j - i
            T, F = self.T, self.X.shape[1]
            bx = np.empty((L, T, F), dtype=np.float32)
            by = np.empty((L, 1), dtype=np.float32)
            for k, idx in enumerate(range(i, j)):
                lo = idx - T + 1
                bx[k] = self.X[lo:idx+1]
                r = np.log(max(self.close[idx+1], 1e-12)) - np.log(max(self.close[idx], 1e-12))
                if self.cut > 0:
                    by[k,0] = 1.0 if r >  self.cut else (0.0 if r < -self.cut else 0.0)
                else:
                    by[k,0] = 1.0 if r > 0.0 else 0.0
            yield torch.from_numpy(bx), torch.from_numpy(by)
            i = j

def train_tfa_transformer(
    symbol: str,
    bars_csv: str,
    feature_spec: str,
    out_dir: str = "artifacts/TFA/v1",
    T: int = 64,
    batch_size: int = 256,
    epochs: int = 15,
    lr: float = 3e-4,
    num_workers: int = 0,
    hidden: int = 192,
    d_model: int = 96,
    nhead: int = 4,
    num_layers: int = 2,
    cutoff_bp: float = 0.0
):
    t0 = time.time()
    dev = _device()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Load bars (for labels only)
    arr = np.genfromtxt(bars_csv, delimiter=",", names=True, dtype=None, encoding=None)
    if "ts_nyt_epoch" in arr.dtype.names:
        ts = arr["ts_nyt_epoch"].astype(np.int64)
    elif "ts" in arr.dtype.names:
        ts = arr["ts"].astype(np.int64)
    else:
        raise ValueError(f"No timestamp column found. Available columns: {arr.dtype.names}")
    close = arr["close"].astype(np.float64)

    # Enforce cached features
    ts_cached, X, names = load_cached_features(symbol, pathlib.Path(bars_csv).parent)
    if X is None:
        raise RuntimeError("TFA training requires precomputed feature cache. Generate it first.")

    spec = _spec_with_hash(feature_spec)
    emit_from = int(spec["alignment_policy"]["emit_from_index"])
    N, F = X.shape
    assert F == 55 and N > emit_from + T

    # Scaler on train split
    split = int(0.9 * N)
    mu = X[emit_from:split].mean(axis=0).astype(np.float32)
    sd = X[emit_from:split].std(axis=0).clip(1e-12, None).astype(np.float32)
    inv = (1.0 / sd).astype(np.float32)

    # Datasets
    ds_tr = SeqWindows(X, close, T=T, emit_from=emit_from, batch=batch_size,
                       start=emit_from+T-1, end=split-2, cutoff_bp=cutoff_bp)
    ds_va = SeqWindows(X, close, T=T, emit_from=emit_from, batch=batch_size,
                       start=split+T-1, end=N-2, cutoff_bp=cutoff_bp)
    loader_tr = DataLoader(ds_tr, batch_size=None, num_workers=num_workers)
    loader_va = DataLoader(ds_va, batch_size=None, num_workers=num_workers)

    # Model/loss/opt
    core = TFA_Transformer(F=F, T=T, d_model=d_model, nhead=nhead,
                           num_layers=num_layers, ffn_hidden=hidden).to(dev)
    model = TFA_Wrapped(core, SeqScaler(mu, inv)).to(dev)
    criterion = nn.BCEWithLogitsLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    def eval_once():
        model.eval(); loss_sum=0.0; n=0
        with torch.inference_mode():
            for bx, by in loader_va:
                bx = bx.to(dev); by = by.to(dev)
                loss = criterion(model(bx), by)
                loss_sum += float(loss) * bx.shape[0]; n += bx.shape[0]
        return loss_sum / max(1,n)

    for ep in range(1, epochs+1):
        model.train(); loss_sum=steps=0
        for bx, by in loader_tr:
            bx = bx.to(dev); by = by.to(dev)
            opt.zero_grad(set_to_none=True)
            loss = criterion(model(bx), by)
            loss.backward(); opt.step()
            loss_sum += float(loss); steps+=1
        va = eval_once()
        print(f"[TFA-SEQ] epoch {ep}/{epochs} train_bce={loss_sum/max(1,steps):.5f}  val_bce={va:.5f}")

    # Export TS
    scripted = torch.jit.trace(model.to("cpu").eval(), torch.rand(2, T, F))
    torch.jit.save(scripted, str(pathlib.Path(out_dir)/"model.pt"))

    # Save spec+meta
    meta = {
        "schema_version":"1.0",
        "saved_at": int(time.time()),
        "framework":"torchscript",
        "expects":{
            "model_type":"transformer",
            "seq_len": int(T),
            "input_dim": int(F),
            "feature_names": names,
            "spec_hash": spec["content_hash"],
            "emit_from": int(emit_from),
            "pad_value": float(spec["alignment_policy"]["pad_value"]),
            "dtype":"float32",
            "output":"logit"
        }
    }
    json.dump(spec, open(pathlib.Path(out_dir)/"feature_spec.json","w"), indent=2)
    json.dump(meta, open(pathlib.Path(out_dir)/"model.meta.json","w"), indent=2)
    print(f"✅ Transformer TFA exported to {out_dir} in {time.time()-t0:.1f}s")
    return str(out_dir)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--bars_csv", required=True)
    ap.add_argument("--feature_spec", required=True)
    ap.add_argument("--out_dir", default="artifacts/TFA/v1")
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()
    train_tfa_transformer(args.symbol, args.bars_csv, args.feature_spec,
                          out_dir=args.out_dir, T=args.T, batch_size=args.batch_size,
                          epochs=args.epochs, lr=args.lr)


