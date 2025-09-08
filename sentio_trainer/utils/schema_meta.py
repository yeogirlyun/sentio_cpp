import json, hashlib, time
from typing import List

NON_FEATURE_COLS = {"ts", "timestamp", "bar_index"}

def _hash(d: dict) -> str:
    return "sha256:" + hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()

def feature_names_from_spec(spec: dict) -> List[str]:
    out=[]
    for f in spec["features"]:
        if "name" in f: 
            name = f["name"]
        else:
            op=f["op"]; src=f.get("source",""); w=str(f.get("window","")); k=str(f.get("k",""))
            name = f"{op}_{src}_{w}_{k}"
        
        # HARDENED: Never allow ts/timestamp/bar_index as feature names
        if name in NON_FEATURE_COLS:
            raise ValueError(f"Spec must not include a feature named '{name}' - this should be metadata, not a model input")
        out.append(name)
    return out

def write_meta_or_die(out_dir: str, spec: dict, X_shape, names: List[str], dtype="float32"):
    """
    HARDENED: Enforces exactly 55 features and fails fast if ts contamination detected.
    """
    # Check for ts contamination
    if any(n in NON_FEATURE_COLS for n in names):
        raise ValueError("Spec/feature list includes a non-feature column (ts/timestamp/bar_index)")
    
    # Check dimension consistency
    if X_shape[1] != len(names):
        raise ValueError(f"X.shape[1]={X_shape[1]} != names={len(names)}")
    
    # HARDENED: Enforce exactly 55 features
    if X_shape[1] != 55:
        raise ValueError(f"Model must train on exactly 55 features; got {X_shape[1]}")
    
    # Create spec with hash
    spec = dict(spec)
    spec["content_hash"] = _hash(spec)
    
    # Create meta with hardcoded 55 input_dim
    meta = {
        "schema_version":"1.0",
        "saved_at":int(time.time()),
        "framework":"torchscript",
        "expects":{
            "input_dim":55,  # HARDENED: Always 55
            "feature_names":names,
            "spec_hash":spec["content_hash"],
            "emit_from":int(spec["alignment_policy"]["emit_from_index"]),
            "pad_value":float(spec["alignment_policy"]["pad_value"]),
            "dtype":dtype,
            "output":"logit"
        }
    }
    
    with open(f"{out_dir}/model.meta.json","w") as f: json.dump(meta,f,indent=2)
    with open(f"{out_dir}/feature_spec.json","w") as f: json.dump(spec,f,indent=2)
    return meta
