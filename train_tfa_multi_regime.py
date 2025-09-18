#!/usr/bin/env python3
"""
Enhanced TFA Training with Multi-Regime Dataset
Combines 3 years of recent historic QQQ data (2022-2025) with MarS-enhanced future data (2026)
"""

from sentio_trainer.trainers.tfa_multi_dataset import train_tfa_multi_regime
import argparse
import yaml
import pathlib

def main():
    parser = argparse.ArgumentParser(description="Train TFA with multi-regime dataset")
    parser.add_argument("--config", default="configs/tfa_multi_regime.yaml", 
                       help="Configuration file (default: configs/tfa_multi_regime.yaml)")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick training mode (reduced epochs for testing)")
    
    args = parser.parse_args()
    
    # Load configuration
    config_path = pathlib.Path(args.config)
    if not config_path.exists():
        print(f"❌ Configuration file not found: {config_path}")
        return
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Quick mode adjustments
    if args.quick:
        print("🚀 Quick training mode enabled")
        config['epochs'] = 10
        config['batch_size'] = 256
        config['T'] = 24
        config['out_dir'] = config['out_dir'].replace('v2', 'v2_quick')
    
    print("=" * 60)
    print("🎯 TFA Multi-Regime Training")
    print("=" * 60)
    print(f"📊 Historic Data: 3 years (Sep 2022 - Sep 2025)")
    print(f"🎲 Synthetic Data: 9 MarS-enhanced tracks (2026)")
    print(f"📈 Total Training Data: ~1.16M minute bars")
    print(f"🏗️  Architecture: Transformer T={config['T']}, d_model={config['d_model']}")
    print(f"📁 Output Directory: {config['out_dir']}")
    print("=" * 60)
    
    # Verify data files exist
    missing_files = []
    for dataset in config['datasets']:
        file_path = pathlib.Path(dataset['path'])
        if not file_path.exists():
            missing_files.append(str(file_path))
    
    if missing_files:
        print("❌ Missing data files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\n💡 Make sure all historic and future QQQ data files are available")
        return
    
    print("✅ All data files found")
    
    # Start training
    try:
        output_dir = train_tfa_multi_regime(**config)
        
        print("\n" + "=" * 60)
        print("🎉 Training Completed Successfully!")
        print("=" * 60)
        print(f"📁 Model artifacts saved to: {output_dir}")
        print(f"📄 Files created:")
        print(f"   - model.pt (TorchScript model)")
        print(f"   - model.meta.json (metadata)")
        print(f"   - feature_spec.json (feature specification)")
        print(f"   - training_metadata.json (dataset info)")
        print(f"   - combined_training_data.csv (processed data)")
        print("\n💡 Use this model with TFA strategy in Sentio trading system")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
