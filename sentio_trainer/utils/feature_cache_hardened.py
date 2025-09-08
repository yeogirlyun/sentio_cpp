import pathlib, numpy as np

NON_FEATURE_COLS = {"ts", "timestamp", "bar_index"}

def load_cached_features(symbol: str, cache_dir: str):
    """
    Returns (ts[int64], X[float32], names[list[str]]), with X excluding non-feature columns.
    HARDENED: Never treats ts/timestamp/bar_index as features.
    """
    p = pathlib.Path(cache_dir) / f"{symbol}_RTH_features.csv"
    if not p.exists():
        return None, None, None

    with p.open("r") as f:
        header = f.readline().strip().split(",")
    assert header, "empty header"
    
    # Partition header - identify feature columns vs metadata columns
    keep_idx, names = [], []
    ts_col_idx = None
    
    for i, col in enumerate(header):
        if col in NON_FEATURE_COLS:
            if i == 0:  # First column is timestamp metadata
                ts_col_idx = i
            continue  # Skip non-feature columns
        else:
            # This is a feature column
            names.append(col)
            keep_idx.append(i)

    # Load data
    M = np.loadtxt(p, delimiter=",", skiprows=1, dtype=np.float64)
    
    # Extract timestamp if available
    ts = None
    if ts_col_idx is not None:
        ts = M[:, ts_col_idx].astype(np.int64)
    
    # Extract feature data (only feature columns)
    X = M[:, keep_idx].astype(np.float32, copy=False)
    
    # HARDENED: Assert exactly 55 features
    if len(names) != 55:
        raise ValueError(f"Expected exactly 55 features, got {len(names)}. Features: {names}")
    
    print(f"[cache] HARDENED: Loaded {X.shape[0]} rows x {X.shape[1]} features ({len(names)} names), ts excluded")
    return ts, X, names
