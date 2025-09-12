from sentio_trainer.trainers.tfa import train_tfa_fast
import argparse, json, pathlib, yaml

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="configs/tfa.yaml or .json")
    args = ap.parse_args()
    p = pathlib.Path(args.config)
    cfg = yaml.safe_load(p.read_text()) if p.suffix.lower()==".yaml" else json.loads(p.read_text())

    print("🚀 Training TFA with improved architecture and data pipeline")
    train_tfa_fast(**cfg)
