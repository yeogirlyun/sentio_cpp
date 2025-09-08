import argparse
import json
import pathlib
import time
from typing import Tuple, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


class SeqEncoder(nn.Module):
    def __init__(self, F: int, d_model: int = 96, nhead: int = 4, layers: int = 2, ffn: int = 192, dropout: float = 0.0):
        super().__init__()
        self.in_proj = nn.Linear(F, d_model)
        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc, num_layers=layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.in_proj(x)
        z = self.encoder(z)
        return z[:, -1, :]


class KochiBinary(nn.Module):
    def __init__(self, F: int, T: int, d_model: int = 96, nhead: int = 4, layers: int = 2, ffn: int = 192):
        super().__init__()
        self.encoder = SeqEncoder(F, d_model, nhead, layers, ffn)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        return self.head(h)


class SeqScaler(nn.Module):
    def __init__(self, mean: np.ndarray, inv_std: np.ndarray):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("inv_std", torch.tensor(inv_std, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) * self.inv_std


class Wrapped(nn.Module):
    def __init__(self, core: nn.Module, scaler: nn.Module):
        super().__init__()
        self.scaler = scaler
        self.core = core

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.core(self.scaler(x))


def read_kochi_csv(path: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding=None)
    cols = [c for c in arr.dtype.names if c.lower() not in ("ts", "timestamp", "bar_index", "time")]
    X = np.vstack([arr[c].astype(np.float32) for c in cols]).T
    if "close" not in arr.dtype.names:
        raise RuntimeError("Kochi CSV must include 'close' column for labels")
    close = arr["close"].astype(np.float64)
    return X, close, cols


class WinDS(Dataset):
    def __init__(self, X: np.ndarray, close: np.ndarray, T: int, emit_from: int, dead_bp: float = 0.0):
        self.X = X
        self.close = close
        self.T = T
        self.emit_from = emit_from
        self.start = max(emit_from, T - 1)
        self.end = len(X) - 2
        self.dead = dead_bp / 10000.0

    def __len__(self) -> int:
        return max(0, self.end - self.start + 1)

    def __getitem__(self, i: int):
        idx = self.start + i
        lo = idx - self.T + 1
        x = self.X[lo : idx + 1]
        r = np.log(max(self.close[idx + 1], 1e-12)) - np.log(max(self.close[idx], 1e-12))
        y = 1.0 if r > self.dead else (0.0 if r < -self.dead else 0.0)
        return x.astype(np.float32), np.array([y], np.float32)


def train_binary(
    features_csv: str,
    spec_json: str,
    out_dir: str = "artifacts/KOCHI_BIN/v1",
    epochs: int = 15,
    batch: int = 256,
    lr: float = 3e-4,
    d_model: int = 96,
    nhead: int = 4,
    layers: int = 2,
    ffn: int = 192,
    dead_zone_bp: float = 0.0,
):
    t0 = time.time()
    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    spec = json.loads(pathlib.Path(spec_json).read_text())
    Fk, T, emit_from = spec["input_dim"], spec["seq_len"], spec["emit_from"]
    names = spec["names"]

    X, close, cols = read_kochi_csv(features_csv)
    assert len(cols) == Fk, f"CSV features ({len(cols)}) != spec.input_dim ({Fk})"
    name_to_idx = {n: i for i, n in enumerate(cols)}
    idxs = [name_to_idx[n] for n in names]
    X = X[:, idxs]

    N = X.shape[0]
    assert N > emit_from + T + 32, "Not enough rows for warmup + windows"

    split = int(0.9 * N)
    mu = X[emit_from:split].mean(axis=0).astype(np.float32)
    sd = X[emit_from:split].std(axis=0).clip(1e-12, None).astype(np.float32)
    inv = (1.0 / sd).astype(np.float32)

    ds_tr = WinDS(X, close, T, emit_from, dead_zone_bp)
    dl_tr = DataLoader(ds_tr, batch_size=batch, shuffle=True, drop_last=True)
    ds_va = WinDS(X, close, T, emit_from, dead_zone_bp)
    dl_va = DataLoader(ds_va, batch_size=batch, shuffle=False, drop_last=False)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    core = KochiBinary(F=Fk, T=T, d_model=d_model, nhead=nhead, layers=layers, ffn=ffn).to(device)
    model = Wrapped(core, SeqScaler(mu, inv)).to(device)
    try:
        model = torch.compile(model, mode="max-autotune")  # type: ignore[attr-defined]
    except Exception:
        pass

    with torch.no_grad():
        tmp = next(iter(dl_tr))
        p = float(tmp[1].mean())
    pos_weight = (1.0 - p) / max(1e-6, p if p > 0 else 0.5)

    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, foreach=True)

    def eval_loss() -> float:
        model.eval()
        s = 0.0
        n = 0
        with torch.inference_mode():
            for bx, by in dl_va:
                bx = bx.to(device)
                by = by.to(device)
                s += float(crit(model(bx), by)) * bx.shape[0]
                n += bx.shape[0]
        return s / max(1, n)

    for ep in range(1, epochs + 1):
        model.train()
        sumloss = 0.0
        steps = 0
        for bx, by in dl_tr:
            bx = bx.to(device)
            by = by.to(device)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sumloss += float(loss)
            steps += 1
        val = eval_loss()
        print(f"[KOCHI-BIN] {ep:02d}/{epochs} bce={sumloss / max(1, steps):.5f} val={val:.5f}")

    m = model.to("cpu").eval()
    ex = torch.randn(2, T, Fk)
    try:
        scripted = torch.jit.script(m)
    except Exception:
        scripted = torch.jit.trace(m, ex)
    outp = pathlib.Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)
    torch.jit.save(scripted, str(outp / "kochi_bin.pt"))

    meta = {
        "schema_version": "1.0",
        "framework": "torchscript",
        "expects": {
            "model_type": "kochi_binary",
            "seq_len": int(T),
            "input_dim": int(Fk),
            "feature_names": names,
            "output": "logit",
            "dtype": "float32",
            "emit_from": int(emit_from),
            "pad_value": float(spec["pad_value"]),
        },
    }
    (outp / "kochi_bin.meta.json").write_text(json.dumps(meta, indent=2))
    print(f"✅ Exported Kochi binary → {outp} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_csv", required=True)
    ap.add_argument("--spec_json", default="configs/features/kochi_v1_spec.json")
    ap.add_argument("--out_dir", default="artifacts/KOCHI_BIN/v1")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--dead_zone_bp", type=float, default=0.0)
    args = ap.parse_args()
    train_binary(**vars(args))


