import json, pathlib, numpy as np


def load_cached_features(symbol: str, cache_dir: str, require_csv: bool = False):
    """
    Returns (ts, X, names). If CSV exists, it is authoritative; else try NPY + bars ts.
    If neither exists, return (None, None, None) and caller can fallback to on-the-fly.
    """
    d = pathlib.Path(cache_dir)
    csv = d / f"{symbol}_RTH_features.csv"
    npy = d / f"{symbol}_RTH_features.npy"
    meta= d / f"{symbol}_RTH_features.meta.json"

    if csv.exists():
        print(f"[cache] Using CSV: {csv}")
        # Load header for names; handle both formats: bar_index,timestamp,... or ts,...
        with csv.open("r") as f:
            header = f.readline().strip()
        cols = header.split(",")
        
        if cols[0] == "bar_index" and len(cols) > 2:
            # Format: bar_index,timestamp,feature1,feature2,...
            names = cols[2:]  # Skip bar_index and timestamp  
            M = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
            ts = M[:,1].astype(np.int64)  # Use timestamp column
            X  = M[:,2:].astype(np.float32, copy=False)  # Skip first two columns
        elif cols[0] == "ts":
            # Format: ts,feature1,feature2,...
            names = cols[1:]
            M = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
            ts = M[:,0].astype(np.int64)  # Use ts column
            X  = M[:,1:].astype(np.float32, copy=False)  # Skip first column
        else:
            raise ValueError(f"Unsupported CSV format. Expected 'bar_index,timestamp,...' or 'ts,...', got '{cols[0]},...'")
        
        print(f"[cache] Loaded {X.shape[0]} rows x {X.shape[1]} features ({len(names)} names)")
        return ts, X, names

    if require_csv:
        return None, None, None

    if npy.exists() and meta.exists():
        print(f"[cache] Using NPY: {npy}")
        X = np.load(npy, mmap_mode="r").astype(np.float32)
        names = json.loads(meta.read_text())["feature_names"]
        # No ts in NPY; caller must align with bars CSV
        return None, X, names

    return None, None, None


def load_kochi_cached_features(symbol: str, cache_dir: str):
    """
    Load precomputed Kochi features produced by tools/generate_kochi_feature_cache.py
    Returns (ts[int64], X[float32], names[list[str]]), or (None,None,None) if missing.
    """
    d = pathlib.Path(cache_dir)
    csv = d / f"{symbol}_KOCHI_features.csv"
    if not csv.exists():
        return None, None, None
    with csv.open("r") as f:
        header = f.readline().strip().split(",")
    # Expect bar_index,timestamp,<names...>
    if len(header) < 3 or header[0] != "bar_index":
        raise ValueError("Unexpected KOCHI CSV header; expected bar_index,timestamp,...")
    names = header[2:]
    M = np.loadtxt(csv, delimiter=",", skiprows=1, dtype=np.float64)
    ts = M[:, 1].astype(np.int64)
    X = M[:, 2:].astype(np.float32, copy=False)
    return ts, X, names
