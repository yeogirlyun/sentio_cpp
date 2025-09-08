import json, numpy as np, pathlib

def validate_cache_against_meta(symbol: str, cache_dir: str, model_meta_path: str):
    meta = json.loads(pathlib.Path(model_meta_path).read_text())
    exp_names = meta["expects"]["feature_names"]
    csv = pathlib.Path(cache_dir) / f"{symbol}_RTH_features.csv"
    if not csv.exists():
        return False, "CSV missing"
    with csv.open("r") as f:
        header = f.readline().strip().split(",")
    if len(header) < 2 or header[0] not in ["bar_index", "ts"]:
        return False, "CSV first column must be bar_index or ts"
    # Skip bar_index and timestamp columns to get feature names
    if header[0] == "bar_index" and len(header) > 2:
        got_names = header[2:]  # Skip bar_index and timestamp
    elif header[0] == "ts" and len(header) > 1:
        got_names = header[1:]  # Skip only timestamp
    else:
        return False, "Invalid CSV header format"
        
    if got_names != exp_names:
        return False, f"CSV feature columns mismatch (expected {len(exp_names)}, got {len(got_names)})\nExpected: {exp_names[:3]}...\nGot: {got_names[:3]}..."
    return True, "OK"
