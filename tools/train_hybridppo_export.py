#!/usr/bin/env python3
"""
HybridPPO Training and ONNX Export Example

This script demonstrates how to train a PPO model and export it to ONNX
for use with the C++ Sentio ML integration system.

Usage:
    python tools/train_hybridppo_export.py

Requirements:
    pip install torch onnx stable-baselines3 numpy
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
from typing import Dict, List, Tuple

class HybridPPOPolicy(nn.Module):
    """Simple PPO policy network for demonstration"""
    
    def __init__(self, input_dim: int = 7, hidden_dim: int = 64, num_actions: int = 3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits"""
        return self.network(x)

def create_dummy_training_data(n_samples: int = 1000, n_features: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """Create dummy training data for demonstration"""
    # Generate random features
    features = np.random.randn(n_samples, n_features)
    
    # Generate random actions (0=SELL, 1=HOLD, 2=BUY)
    actions = np.random.randint(0, 3, n_samples)
    
    return features, actions

def train_simple_policy(features: np.ndarray, actions: np.ndarray, epochs: int = 100) -> HybridPPOPolicy:
    """Train a simple policy network"""
    model = HybridPPOPolicy(input_dim=features.shape[1], num_actions=3)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        # Convert to tensors
        x = torch.FloatTensor(features)
        y = torch.LongTensor(actions)
        
        # Forward pass
        logits = model(x)
        loss = criterion(logits, y)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    
    return model

def export_to_onnx(model: HybridPPOPolicy, output_dir: str, version: str = "v1"):
    """Export trained model to ONNX format"""
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Dummy input for tracing
    dummy_input = torch.randn(1, 7, dtype=torch.float32)
    
    # Export to ONNX
    onnx_path = os.path.join(output_dir, "model.onnx")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["features"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
        verbose=False
    )
    
    print(f"✅ Model exported to {onnx_path}")
    return onnx_path

def create_metadata(output_dir: str, version: str = "v1"):
    """Create metadata.json file"""
    metadata = {
        "model_id": "HybridPPO",
        "version": version,
        "feature_names": ["ret_1m", "ret_5m", "rsi_14", "sma_10", "sma_30", "vol_1m", "spread_bp"],
        "mean": [0.0, 0.0, 50.0, 0.0, 0.0, 0.0, 1.5],
        "std": [1.0, 1.0, 20.0, 1.0, 1.0, 1.0, 0.5],
        "clip": [-5.0, 5.0],
        "actions": ["SELL", "HOLD", "BUY"],
        "expected_bar_spacing_sec": 60,
        "instrument_family": "QQQ",
        "notes": "kouchi_sentio_hybrid migrated; PPO policy head logits->softmax"
    }
    
    metadata_path = os.path.join(output_dir, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata exported to {metadata_path}")
    return metadata_path

def main():
    """Main training and export pipeline"""
    print("🚀 Starting HybridPPO Training and Export Pipeline")
    print("=" * 50)
    
    # Configuration
    output_dir = "artifacts/HybridPPO/v1"
    version = "v1"
    
    # Create training data
    print("📊 Generating training data...")
    features, actions = create_dummy_training_data(n_samples=1000, n_features=7)
    print(f"   Generated {len(features)} samples with {features.shape[1]} features")
    
    # Train model
    print("🧠 Training policy network...")
    model = train_simple_policy(features, actions, epochs=100)
    print("   Training completed!")
    
    # Export to ONNX
    print("📦 Exporting to ONNX...")
    onnx_path = export_to_onnx(model, output_dir, version)
    
    # Create metadata
    print("📋 Creating metadata...")
    metadata_path = create_metadata(output_dir, version)
    
    # Verify files
    print("✅ Verification:")
    print(f"   ONNX model: {os.path.exists(onnx_path)}")
    print(f"   Metadata: {os.path.exists(metadata_path)}")
    
    print("\n🎉 Export completed successfully!")
    print(f"   Model ready for C++ integration at: {output_dir}")
    print("\nNext steps:")
    print("1. Build C++ project with ONNX Runtime support")
    print("2. Test HybridPPO strategy with: build/sentio_cli backtest QQQ --strategy hybrid_ppo")

if __name__ == "__main__":
    main()
