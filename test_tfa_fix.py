#!/usr/bin/env python3
"""
Quick test script to verify the TFA training fix
"""

import sys
import traceback
from sentio_trainer.trainers.tfa_multi_dataset import train_tfa_multi_regime

def test_tfa_fix():
    """Test the fixed TFA training with minimal config"""
    
    # Minimal test configuration
    test_config = {
        'symbol': 'QQQ',
        'out_dir': 'artifacts/TFA/test_fix',
        'feature_spec': 'configs/features/feature_spec_55_minimal.json',
        'feature_cache': 'data',
        'datasets': [
            {
                'path': 'data/equities/QQQ_RTH_NH.csv',
                'regime': 'historic_real',
                'weight': 1.0,
                'description': 'Historic data test'
            }
        ],
        # Minimal settings for quick test
        'batch_size': 64,
        'epochs': 2,
        'lr': 0.001,
        'num_workers': 0,  # Single-threaded for debugging
        'train_split': 0.9,
        'T': 16,           # Shorter sequences
        'd_model': 32,     # Smaller model
        'nhead': 2,
        'num_layers': 1,
        'ffn_hidden': 64,
        'label_horizon': 1,
        'label_kind': 'logret_fwd',
        'min_volume_threshold': 50000,
        'price_change_threshold': 0.0003,
        'max_training_hours': 1,  # 1 hour limit for test
    }
    
    print("🧪 Testing TFA Training Fix")
    print("=" * 40)
    print(f"Config: {test_config}")
    print("=" * 40)
    
    try:
        output_dir = train_tfa_multi_regime(**test_config)
        print("\n✅ Test PASSED!")
        print(f"📁 Output: {output_dir}")
        return True
        
    except Exception as e:
        print(f"\n❌ Test FAILED: {e}")
        print("\nFull traceback:")
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tfa_fix()
    sys.exit(0 if success else 1)
