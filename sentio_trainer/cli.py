import argparse, json, pathlib
from sentio_trainer.trainers.tfa_fast import train_tfa_fast

def main():
    p = argparse.ArgumentParser("sentio-trainer")
    p.add_argument("--config", required=True, help="Path to tfa config (yaml or json)")
    args = p.parse_args()

    cfg_path = pathlib.Path(args.config)
    if cfg_path.suffix.lower() == ".json":
        cfg = json.loads(cfg_path.read_text())
    else:
        try:
            import yaml
        except ImportError as e:
            raise SystemExit("Install pyyaml or use JSON config") from e
        cfg = yaml.safe_load(cfg_path.read_text())

    train_tfa_fast(**cfg)

if __name__ == "__main__":
    main()
