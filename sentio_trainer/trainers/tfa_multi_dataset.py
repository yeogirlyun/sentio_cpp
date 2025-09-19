"""
Enhanced TFA Trainer with Multi-Dataset Support
Combines historic QQQ data with MarS-enhanced future data for robust training
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pathlib
import time
import json
import yaml
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from .tfa import (
    TFA_Transformer, SeqScaler, TFA_Wrapped,
    _device, _set_fast_flags, _ensure_dir, _spec_with_hash, load_cached_features
)
from numpy.lib.stride_tricks import as_strided

class FixedSeqWindows(torch.utils.data.Dataset):
    """
    Fixed version of EfficientSeqWindows that properly handles scalar labels
    """
    def __init__(self, X: np.ndarray, y: np.ndarray, T: int, start: int, end: int):
        super().__init__()
        self.X, self.y = X, y
        self.T = T
        
        # We can only start creating windows after T-1 elements
        self.start = max(start, T - 1)
        self.end = end
        
        # Calculate the number of valid windows
        self.n_windows = self.end - self.start + 1
        
        # Use as_strided to create a zero-copy view of the windows
        n, f = X.shape
        s_n, s_f = X.strides
        self.X_windows = as_strided(X, shape=(n - T + 1, T, f), strides=(s_n, s_n, s_f))
        
        print(f"FixedSeqWindows: X.shape={X.shape}, y.shape={y.shape}, T={T}, start={start}, end={end}")
        print(f"  Windows: {self.n_windows}, X_windows.shape={self.X_windows.shape}")

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        # The index into our dataset corresponds to the end of the window
        window_end_idx = self.start + idx
        
        # Get the windowed X view and the corresponding y label
        x_window = self.X_windows[window_end_idx - self.T + 1]
        y_label = self.y[window_end_idx]
        
        # **FIX**: Ensure y_label is always a tensor-compatible array
        if np.isscalar(y_label) or y_label.ndim == 0:
            y_label = np.array([y_label], dtype=np.float32)
        else:
            y_label = np.asarray(y_label, dtype=np.float32)
        
        # Convert to tensors
        return torch.from_numpy(x_window.copy()), torch.from_numpy(y_label).squeeze()

class MultiDatasetTFATrainer:
    """Enhanced TFA trainer that combines multiple datasets for robust training"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = _device()
        _set_fast_flags()
        
    def standardize_csv_format(self, csv_path: str, output_path: str = None) -> str:
        """Convert different CSV formats to standardized format for TFA training"""
        df = pd.read_csv(csv_path)
        
        # Detect format and standardize
        if 'ts_nyt_epoch' in df.columns:
            # Historic format: ts_utc,ts_nyt_epoch,open,high,low,close,volume
            df_std = pd.DataFrame({
                'ts': df['ts_nyt_epoch'],
                'open': df['open'],
                'high': df['high'], 
                'low': df['low'],
                'close': df['close'],
                'volume': df['volume']
            })
        elif 'timestamp' in df.columns:
            # Future format: timestamp,symbol,open,high,low,close,volume
            # Convert ISO timestamp to epoch
            df['ts'] = pd.to_datetime(df['timestamp']).astype(int) // 10**9
            df_std = pd.DataFrame({
                'ts': df['ts'],
                'open': df['open'],
                'high': df['high'],
                'low': df['low'], 
                'close': df['close'],
                'volume': df['volume']
            })
        else:
            raise ValueError(f"Unknown CSV format in {csv_path}")
        
        # Save standardized format
        if output_path is None:
            output_path = csv_path.replace('.csv', '_standardized.csv')
        
        df_std.to_csv(output_path, index=False)
        print(f"Standardized {len(df_std)} bars from {csv_path} -> {output_path}")
        return output_path
    
    def combine_datasets(self, dataset_configs: List[Dict]) -> str:
        """Combine multiple datasets into a single training file"""
        combined_data = []
        
        print(f"\n📊 Loading and combining {len(dataset_configs)} datasets...")
        print("=" * 60)
        
        total_bars = 0
        for i, config in enumerate(dataset_configs, 1):
            csv_path = config['path']
            weight = config.get('weight', 1.0)
            regime = config.get('regime', 'unknown')
            
            print(f"[{i:2d}/{len(dataset_configs)}] Loading {regime}")
            print(f"         Path: {csv_path}")
            print(f"         Weight: {weight}")
            
            # Check if file exists
            if not pathlib.Path(csv_path).exists():
                print(f"         ❌ File not found: {csv_path}")
                continue
            
            # Standardize format
            print(f"         🔄 Standardizing format...")
            std_path = self.standardize_csv_format(csv_path)
            
            print(f"         📖 Loading data...")
            df = pd.read_csv(std_path)
            original_size = len(df)
            
            # Add regime information as metadata
            df['regime'] = regime
            df['dataset_weight'] = weight
            
            # Sample based on weight (for balancing)
            if weight != 1.0:
                sample_size = int(len(df) * weight)
                if sample_size > 0:
                    df = df.sample(n=min(sample_size, len(df)), random_state=42)
                    print(f"         ⚖️  Resampled: {original_size} → {len(df)} bars (weight: {weight})")
            
            combined_data.append(df)
            total_bars += len(df)
            print(f"         ✅ Added {len(df):,} bars from {regime}")
            print()
        
        # Combine all datasets
        combined_df = pd.concat(combined_data, ignore_index=True)
        
        # Sort by timestamp for proper temporal order
        combined_df = combined_df.sort_values('ts').reset_index(drop=True)
        
        # Save combined dataset
        output_dir = pathlib.Path(self.config['out_dir'])
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_path = output_dir / 'combined_training_data.csv'
        
        # Save in TFA-compatible format (no regime columns)
        tfa_df = combined_df[['ts', 'open', 'high', 'low', 'close', 'volume']]
        tfa_df.to_csv(combined_path, index=False)
        
        # Save metadata separately
        metadata = {
            'total_bars': len(combined_df),
            'regimes': combined_df['regime'].value_counts().to_dict(),
            'date_range': {
                'start': int(combined_df['ts'].min()),
                'end': int(combined_df['ts'].max())
            },
            'datasets': dataset_configs
        }
        
        metadata_path = output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("=" * 60)
        print(f"📈 Combined Dataset Summary:")
        print(f"   Total bars: {len(combined_df):,}")
        print(f"   Total size: {len(combined_df) * 6 * 8 / 1024 / 1024:.1f} MB (estimated)")
        print(f"   Date range: {pd.to_datetime(combined_df['ts'], unit='s').min()} → {pd.to_datetime(combined_df['ts'], unit='s').max()}")
        print(f"   Regime distribution:")
        for regime, count in combined_df['regime'].value_counts().items():
            pct = count / len(combined_df) * 100
            print(f"     {regime}: {count:,} bars ({pct:.1f}%)")
        print(f"   Saved to: {combined_path}")
        print("=" * 60)
        
        return str(combined_path)
    
    def create_regime_aware_features(self, bars_csv: str, feature_spec: str) -> Tuple[np.ndarray, np.ndarray]:
        """Create features with regime awareness for better generalization"""
        
        # Load the combined dataset with regime info
        metadata_path = pathlib.Path(bars_csv).parent / 'training_metadata.json'
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Training on {metadata['total_bars']} bars across regimes: {list(metadata['regimes'].keys())}")
        
        # Use existing TFA feature generation but with enhanced preprocessing
        arr = np.genfromtxt(bars_csv, delimiter=",", names=True, dtype=None, encoding=None)
        
        ts = arr["ts"].astype(np.int64)
        openp = arr["open"].astype(np.float64)
        high = arr["high"].astype(np.float64)
        low = arr["low"].astype(np.float64)
        close = arr["close"].astype(np.float64)
        vol = arr["volume"].astype(np.float64)
        
        # Load feature spec
        spec = _spec_with_hash(feature_spec)
        emit_from = int(spec["alignment_policy"]["emit_from_index"])
        
        # Try to load cached features first
        symbol = self.config.get('symbol', 'QQQ')
        feature_cache = self.config.get('feature_cache', 'data')
        
        ts_cached, X_cached, names_cached = load_cached_features(symbol, feature_cache)
        
        if X_cached is not None and len(X_cached) >= len(ts):
            print(f"Using cached features: {X_cached.shape}")
            X = X_cached[:len(ts)]  # Trim to match current dataset
        else:
            print("Generating features on-the-fly using C++ feature builder...")
            # Use proper C++ feature generation (same as tfa.py)
            try:
                import sentio_features as sf
                spec_json = json.dumps(spec, sort_keys=True)
                print(f"Building features for {symbol} with {len(ts)} bars...")
                X = sf.build_features_from_spec(symbol, ts, openp, high, low, close, vol, spec_json).astype(np.float32)
                print(f"Generated real features: {X.shape}")
            except ImportError:
                print("❌ ERROR: sentio_features module not available!")
                print("   This means C++ feature generation is not accessible from Python.")
                print("   The model will be trained on random features and will not work!")
                raise RuntimeError("Cannot train without proper feature generation. Build sentio_features module first.")
        
        # Create enhanced labels with regime awareness
        y = self._create_enhanced_labels(close, ts)
        
        return X, y
    
    def _estimate_training_time(self, epochs: int, batches_per_epoch: int, batch_size: int) -> str:
        """Estimate total training time based on dataset size and hardware"""
        
        # Base estimates (adjust based on your hardware)
        if torch.cuda.is_available():
            # NVIDIA GPU estimates
            samples_per_second = 8000 if batch_size <= 256 else 12000 if batch_size <= 512 else 15000
        elif torch.backends.mps.is_available():
            # Apple Silicon with Metal Performance Shaders
            import platform
            machine = platform.machine().lower()
            
            if 'm4' in machine or 'arm64' in machine:
                # Apple M4: Significant performance boost with enhanced Neural Engine
                # M4 has 16-core Neural Engine + improved GPU cores
                samples_per_second = 6000 if batch_size <= 256 else 9000 if batch_size <= 512 else 12000
            elif any(chip in machine for chip in ['m3', 'm2', 'm1']):
                # Apple M1/M2/M3: Good performance with Metal
                samples_per_second = 3000 if batch_size <= 256 else 4500 if batch_size <= 512 else 6000
            else:
                # Other ARM processors
                samples_per_second = 2000 if batch_size <= 256 else 3000 if batch_size <= 512 else 4000
        else:
            # CPU-only estimates (Intel/AMD)
            samples_per_second = 1000 if batch_size <= 256 else 1500 if batch_size <= 512 else 2000
        
        total_samples = epochs * batches_per_epoch * batch_size
        estimated_seconds = total_samples / samples_per_second
        
        # Add overhead for validation, data loading, etc.
        estimated_seconds *= 1.3
        
        return self._format_time(estimated_seconds)
    
    def _format_time(self, seconds: float) -> str:
        """Format seconds into human-readable time string"""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            if hours < 24:
                return f"{hours:.1f}h"
            else:
                days = hours / 24
                return f"{days:.1f}d"

    def _create_enhanced_labels(self, close: np.ndarray, ts: np.ndarray) -> np.ndarray:
        """Create labels with enhanced signal detection across regimes"""
        
        horizon = self.config.get('label_horizon', 3)
        label_kind = self.config.get('label_kind', 'logret_fwd')
        
        N = len(close)
        y = np.zeros(N, dtype=np.float32)
        
        if label_kind == "logret_fwd":
            # Forward log returns with enhanced signal detection
            for i in range(N - horizon):
                if i + horizon < N:
                    ret = np.log(close[i + horizon] / close[i])
                    
                    # Enhanced labeling: stronger signals for clearer moves
                    if abs(ret) > 0.01:  # 1% threshold for strong signals
                        y[i] = np.tanh(ret * 10)  # Amplify strong signals
                    else:
                        y[i] = ret * 5  # Moderate amplification for weak signals
        
        elif label_kind == "close_diff":
            # Price difference based labels
            for i in range(N - horizon):
                if i + horizon < N:
                    diff = (close[i + horizon] - close[i]) / close[i]
                    y[i] = np.tanh(diff * 20)  # Normalize to [-1, 1]
        
        # Ensure y is properly shaped and typed
        y = y.astype(np.float32)
        print(f"Labels created: shape={y.shape}, dtype={y.dtype}, range=[{y.min():.6f}, {y.max():.6f}]")
        
        return y
    
    def train_multi_regime_model(self) -> str:
        """Train TFA model on combined multi-regime dataset"""
        
        print("🚀 Training TFA with Multi-Regime Dataset")
        print(f"Configuration: {self.config}")
        
        t0 = time.time()
        out_dir = self.config['out_dir']
        _ensure_dir(out_dir)
        
        # 1. Combine datasets
        dataset_configs = self.config['datasets']
        combined_csv = self.combine_datasets(dataset_configs)
        
        # 2. Generate features
        feature_spec = self.config['feature_spec']
        X, y = self.create_regime_aware_features(combined_csv, feature_spec)
        
        N, F = X.shape
        print(f"Training data: {N} bars, {F} features")
        
        # 3. Enhanced data quality filtering
        min_vol = self.config.get('min_volume_threshold', 100000)
        price_change_thresh = self.config.get('price_change_threshold', 0.0005)
        
        # Load price data for filtering
        arr = np.genfromtxt(combined_csv, delimiter=",", names=True, dtype=None, encoding=None)
        vol = arr["volume"].astype(np.float64)
        close = arr["close"].astype(np.float64)
        
        # Quality filters
        vol_mask = vol >= min_vol
        price_change = np.abs(np.diff(close, prepend=close[0])) / close
        price_mask = np.concatenate([[True], price_change[1:] >= price_change_thresh])
        
        quality_mask = vol_mask & price_mask
        quality_count = quality_mask.sum()
        
        print(f"Quality filtering: {quality_count}/{N} bars passed ({quality_count/N*100:.1f}%)")
        
        # Apply quality mask
        X = X[quality_mask]
        y = y[quality_mask]
        N = len(X)
        
        # 4. Train/validation split with regime stratification
        split_ratio = self.config.get('train_split', 0.9)
        split = int(split_ratio * N)
        
        # 5. Enhanced feature scaling
        emit_from = int(_spec_with_hash(feature_spec)["alignment_policy"]["emit_from_index"])
        emit_from = min(emit_from, N // 10)  # Ensure reasonable emit_from
        
        tr_mask = np.arange(split) >= emit_from
        X_train = X[:split][tr_mask]
        
        # Robust scaling with outlier handling
        mu = np.median(X_train, axis=0).astype(np.float32)  # Use median for robustness
        mad = np.median(np.abs(X_train - mu), axis=0).astype(np.float32)  # Median absolute deviation
        sd = (mad * 1.4826).clip(1e-6, None)  # Convert MAD to std equivalent
        inv = (1.0 / sd).astype(np.float32)
        
        print(f"Feature scaling: median={mu.mean():.3f}, mad_std={sd.mean():.3f}")
        
        # 6. Save processed data
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        x_path = pathlib.Path(out_dir) / "X.npy"
        y_path = pathlib.Path(out_dir) / "y.npy"
        np.save(x_path, X)
        np.save(y_path, y)
        
        # 7. Create enhanced data loaders
        T = self.config.get('T', 32)
        batch_size = self.config.get('batch_size', 256)
        num_workers = self.config.get('num_workers', 6)
        
        # **DEBUG**: Check data shapes before creating datasets
        print(f"Creating datasets: X.shape={X.shape}, y.shape={y.shape}, y.dtype={y.dtype}")
        print(f"Sample y values: {y[:5]}")
        print(f"Dataset ranges: emit_from={emit_from}, split={split}, N={N}")
        
        ds_tr = FixedSeqWindows(X, y, T=T, start=emit_from, end=split-1)
        ds_va = FixedSeqWindows(X, y, T=T, start=split, end=N-2)
        
        # **MPS COMPATIBILITY**: Disable pin_memory and reduce workers for Apple Silicon
        use_mps = torch.backends.mps.is_available()
        actual_pin_memory = False if use_mps else True
        actual_num_workers = 0  # Always use single-threaded for compatibility
        
        print(f"DataLoader config: MPS={use_mps}, workers={actual_num_workers}, pin_memory={actual_pin_memory}")
        
        loader_tr = torch.utils.data.DataLoader(
            ds_tr, batch_size=batch_size, num_workers=actual_num_workers, 
            shuffle=True, pin_memory=actual_pin_memory, persistent_workers=False
        )
        loader_va = torch.utils.data.DataLoader(
            ds_va, batch_size=batch_size, num_workers=0,  # Always single-threaded for validation
            shuffle=False, pin_memory=actual_pin_memory
        )
        
        print(f"Data loaders: train={len(ds_tr)}, val={len(ds_va)}")
        
        # 8. Enhanced model architecture
        d_model = self.config.get('d_model', 96)
        nhead = self.config.get('nhead', 4)
        num_layers = self.config.get('num_layers', 2)
        ffn_hidden = self.config.get('ffn_hidden', 192)
        
        core = TFA_Transformer(
            F=F, T=T, d_model=d_model, nhead=nhead,
            num_layers=num_layers, ffn_hidden=ffn_hidden
        ).to(self.device)
        
        scaler = SeqScaler(mu, inv)
        model = TFA_Wrapped(core, scaler).to(self.device)
        
        # 9. Enhanced training with regime-aware optimization
        lr = self.config.get('lr', 0.002)
        epochs = self.config.get('epochs', 50)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.MSELoss()
        
        # Training loop with comprehensive progress monitoring
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        # Progress tracking
        total_batches_per_epoch = len(loader_tr)
        total_val_batches = len(loader_va)
        training_start_time = time.time()
        epoch_times = []
        
        # **HARD TIME LIMIT**: Configurable maximum training time (default: 10 hours)
        MAX_TRAINING_HOURS = self.config.get('max_training_hours', 10)
        MAX_TRAINING_SECONDS = MAX_TRAINING_HOURS * 3600
        
        print(f"\n🚀 Starting training: {epochs} epochs, {total_batches_per_epoch} batches/epoch")
        print(f"📊 Training data: {len(ds_tr)} sequences, Validation: {len(ds_va)} sequences")
        print(f"⚡ Device: {self.device}, Batch size: {batch_size}")
        print(f"🕐 Estimated time: {self._estimate_training_time(epochs, total_batches_per_epoch, batch_size)}")
        print(f"⏰ Hard time limit: {MAX_TRAINING_HOURS} hours (safety stop)")
        print("=" * 80)
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            # **TIME LIMIT CHECK**: Check at start of each epoch
            elapsed_total = epoch_start_time - training_start_time
            if elapsed_total > MAX_TRAINING_SECONDS:
                print(f"\n⏰ HARD TIME LIMIT REACHED: {MAX_TRAINING_HOURS} hours")
                print(f"   Training stopped at epoch {epoch+1}/{epochs}")
                print(f"   Total time elapsed: {self._format_time(elapsed_total)}")
                print(f"   Best validation loss achieved: {best_val_loss:.6f}")
                break
            
            # Show remaining time budget
            remaining_time = MAX_TRAINING_SECONDS - elapsed_total
            print(f"\n⏰ Time budget: {self._format_time(remaining_time)} remaining of {MAX_TRAINING_HOURS}h limit")
            
            # Training phase with progress tracking
            model.train()
            train_loss = 0.0
            train_batches = 0
            
            print(f"\n📈 Epoch {epoch+1}/{epochs} - Training Phase")
            batch_start_time = time.time()
            
            time_limit_reached = False
            for batch_idx, (batch_x, batch_y) in enumerate(loader_tr):
                # **TIME LIMIT CHECK**: Check every 100 batches during training
                if batch_idx % 100 == 0:
                    elapsed_total = time.time() - training_start_time
                    if elapsed_total > MAX_TRAINING_SECONDS:
                        print(f"\n⏰ HARD TIME LIMIT REACHED during training batch {batch_idx+1}")
                        print(f"   Stopping mid-epoch after {self._format_time(elapsed_total)}")
                        time_limit_reached = True
                        break
                
                batch_x = batch_x.to(self.device, non_blocking=True)
                batch_y = batch_y.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred.squeeze(), batch_y)
                loss.backward()
                
                # Gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                optimizer.step()
                
                train_loss += loss.item()
                train_batches += 1
                
                # Progress updates every 10% of batches or every 100 batches
                if (batch_idx + 1) % max(1, total_batches_per_epoch // 10) == 0 or (batch_idx + 1) % 100 == 0:
                    elapsed = time.time() - batch_start_time
                    elapsed_total = time.time() - training_start_time
                    batches_per_sec = (batch_idx + 1) / elapsed
                    progress_pct = (batch_idx + 1) / total_batches_per_epoch * 100
                    eta_seconds = (total_batches_per_epoch - batch_idx - 1) / max(batches_per_sec, 0.1)
                    remaining_budget = MAX_TRAINING_SECONDS - elapsed_total
                    
                    print(f"  Batch {batch_idx+1:4d}/{total_batches_per_epoch} ({progress_pct:5.1f}%) | "
                          f"Loss: {loss.item():.6f} | "
                          f"Speed: {batches_per_sec:.1f} batch/s | "
                          f"ETA: {self._format_time(eta_seconds)} | "
                          f"Budget: {self._format_time(remaining_budget)}")
            
            # Break out of epoch loop if time limit reached
            if time_limit_reached:
                break
            
            # Validation phase with progress tracking
            print(f"🔍 Epoch {epoch+1}/{epochs} - Validation Phase")
            model.eval()
            val_loss = 0.0
            val_batches = 0
            val_start_time = time.time()
            
            with torch.no_grad():
                for val_batch_idx, (batch_x, batch_y) in enumerate(loader_va):
                    batch_x = batch_x.to(self.device, non_blocking=True)
                    batch_y = batch_y.to(self.device, non_blocking=True)
                    
                    pred = model(batch_x)
                    loss = criterion(pred.squeeze(), batch_y)
                    
                    val_loss += loss.item()
                    val_batches += 1
                    
                    # Validation progress (less frequent updates)
                    if (val_batch_idx + 1) % max(1, total_val_batches // 5) == 0:
                        val_progress = (val_batch_idx + 1) / total_val_batches * 100
                        print(f"  Validation: {val_batch_idx+1:4d}/{total_val_batches} ({val_progress:5.1f}%)")
            
            scheduler.step()
            
            # Epoch summary
            epoch_time = time.time() - epoch_start_time
            epoch_times.append(epoch_time)
            
            avg_train_loss = train_loss / max(train_batches, 1)
            avg_val_loss = val_loss / max(val_batches, 1)
            
            # Calculate time estimates
            avg_epoch_time = np.mean(epoch_times)
            remaining_epochs = epochs - (epoch + 1)
            eta_total = remaining_epochs * avg_epoch_time
            
            # Progress summary
            print(f"\n📊 Epoch {epoch+1}/{epochs} Summary:")
            print(f"  Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")
            print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            print(f"  Epoch Time: {self._format_time(epoch_time)} | Avg: {self._format_time(avg_epoch_time)}")
            print(f"  ETA: {self._format_time(eta_total)} | Total Elapsed: {self._format_time(time.time() - training_start_time)}")
            
            # Early stopping with progress info
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
                torch.save(model.state_dict(), pathlib.Path(out_dir) / 'best_model.pth')
                print(f"  ✅ New best model saved! (Val Loss: {best_val_loss:.6f})")
            else:
                patience_counter += 1
                print(f"  ⏳ No improvement ({patience_counter}/{patience})")
                if patience_counter >= patience:
                    print(f"\n🛑 Early stopping at epoch {epoch+1}")
                    print(f"   Best validation loss: {best_val_loss:.6f}")
                    break
            
            print("=" * 80)
        
        # 10. Export final model
        model.load_state_dict(torch.load(pathlib.Path(out_dir) / 'best_model.pth'))
        model.eval()
        
        # Export TorchScript
        dummy_input = torch.randn(1, T, F).to(self.device)
        traced_model = torch.jit.trace(model, dummy_input)
        model_path = pathlib.Path(out_dir) / "model.pt"
        traced_model.save(str(model_path))
        
        # Export metadata (two formats for compatibility)
        total_training_time = time.time() - t0
        
        # 1. New format metadata (model.meta.json) - for TfaSeqContext
        new_metadata = {
            "model_type": "TFA_MultiRegime",
            "feature_count": F,
            "sequence_length": T,
            "d_model": d_model,
            "nhead": nhead,
            "num_layers": num_layers,
            "training_bars": N,
            "training_time_sec": total_training_time,
            "training_time_formatted": self._format_time(total_training_time),
            "epochs_completed": epoch + 1,
            "epochs_planned": epochs,
            "best_val_loss": best_val_loss,
            "regimes_trained": list(set(config['regime'] for config in dataset_configs)),
            "time_limit_hours": MAX_TRAINING_HOURS,
            "time_limit_reached": total_training_time > MAX_TRAINING_SECONDS,
            "early_stopped": patience_counter >= patience if 'patience_counter' in locals() else False
        }
        
        with open(pathlib.Path(out_dir) / "model.meta.json", 'w') as f:
            json.dump(new_metadata, f, indent=2)
        
        # 2. Legacy format metadata (metadata.json) - for ModelRegistryTS
        # Generate feature names from spec
        feature_names = []
        for f in spec["features"]:
            if "name" in f:
                feature_names.append(f["name"])
            else:
                op = f["op"]
                src = f.get("source", "")
                w = str(f.get("window", ""))
                k = str(f.get("k", ""))
                feature_names.append(f"{op}_{src}_{w}_{k}")
        
        legacy_metadata = {
            "schema_version": "1.0",
            "saved_at": int(time.time()),
            "framework": "torchscript",
            "feature_names": feature_names,
            "mean": [0.0] * F,    # Model has built-in normalization, so use zeros
            "std": [1.0] * F,     # Model has built-in normalization, so use ones
            "clip": [],           # No clipping used
            "actions": [],        # No discrete actions (regression model)
            "seq_len": T,
            "input_layout": "BTF",
            "format": "torchscript"
        }
        
        with open(pathlib.Path(out_dir) / "metadata.json", 'w') as f:
            json.dump(legacy_metadata, f, indent=2)
        
        # Copy feature spec
        import shutil
        shutil.copy(feature_spec, pathlib.Path(out_dir) / "feature_spec.json")
        
        # Final training summary
        total_training_time = time.time() - t0
        final_summary = []
        
        if total_training_time > MAX_TRAINING_SECONDS:
            final_summary.append(f"⏰ Training completed with HARD TIME LIMIT ({MAX_TRAINING_HOURS}h)")
        else:
            final_summary.append(f"✅ Multi-regime TFA training completed successfully")
        
        final_summary.extend([
            f"📁 Artifacts saved to: {out_dir}",
            f"🎯 Best validation loss: {best_val_loss:.6f}",
            f"⏱️  Total training time: {self._format_time(total_training_time)}",
            f"📊 Training completed: {epoch+1}/{epochs} epochs"
        ])
        
        if total_training_time > MAX_TRAINING_SECONDS:
            final_summary.extend([
                f"⚠️  Training stopped due to time limit",
                f"💡 Consider reducing model size or dataset for faster convergence"
            ])
        
        print("\n" + "=" * 80)
        for line in final_summary:
            print(line)
        print("=" * 80)
        
        return out_dir


def train_tfa_multi_regime(**config):
    """Main entry point for multi-regime TFA training"""
    trainer = MultiDatasetTFATrainer(config)
    return trainer.train_multi_regime_model()
