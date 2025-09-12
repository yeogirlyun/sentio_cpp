#!/usr/bin/env python3
"""
Strategy Evaluation CLI Tool

This tool provides comprehensive evaluation of trading strategies using
the universal evaluation framework.
"""

import argparse
import json
import pathlib
import numpy as np
import torch
from typing import Dict, List, Tuple

from sentio_trainer.utils.strategy_evaluation import (
    StrategyEvaluator, quick_evaluate, compare_strategies_quick
)


def load_strategy_data(data_path: str) -> Tuple[torch.Tensor, np.ndarray]:
    """
    Load strategy data from various formats.
    
    Args:
        data_path: Path to data file (JSON, NPZ, or CSV)
        
    Returns:
        Tuple of (predictions, actual_returns)
    """
    data_path = pathlib.Path(data_path)
    
    if data_path.suffix == '.json':
        with open(data_path, 'r') as f:
            data = json.load(f)
        predictions = torch.tensor(data['predictions'])
        actual_returns = np.array(data['actual_returns'])
    
    elif data_path.suffix == '.npz':
        data = np.load(data_path)
        predictions = torch.tensor(data['predictions'])
        actual_returns = data['actual_returns']
    
    elif data_path.suffix == '.csv':
        import pandas as pd
        df = pd.read_csv(data_path)
        predictions = torch.tensor(df['predictions'].values)
        actual_returns = df['actual_returns'].values
    
    else:
        raise ValueError(f"Unsupported file format: {data_path.suffix}")
    
    return predictions, actual_returns


def evaluate_single_strategy(args):
    """Evaluate a single strategy."""
    print(f"Loading strategy data from: {args.data}")
    predictions, actual_returns = load_strategy_data(args.data)
    
    print(f"Data loaded: {len(predictions)} predictions, {len(actual_returns)} returns")
    
    # Run evaluation
    evaluator = StrategyEvaluator(args.strategy_name)
    results = evaluator.evaluate_strategy_signal(
        predictions, actual_returns, verbose=True
    )
    
    # Save results if requested
    if args.output:
        evaluator.save_results(results, args.output)
        print(f"\nResults saved to: {args.output}")
    
    return results


def compare_multiple_strategies(args):
    """Compare multiple strategies."""
    strategy_data = {}
    
    for data_path in args.data_files:
        strategy_name = pathlib.Path(data_path).stem
        print(f"Loading {strategy_name} from: {data_path}")
        predictions, actual_returns = load_strategy_data(data_path)
        strategy_data[strategy_name] = (predictions, actual_returns)
    
    # Run comparison
    compare_strategies_quick(strategy_data)
    
    # Save comparison if requested
    if args.output:
        results = []
        for name, (predictions, actual_returns) in strategy_data.items():
            evaluator = StrategyEvaluator(name)
            result = evaluator.evaluate_strategy_signal(
                predictions, actual_returns, verbose=False
            )
            results.append(result)
        
        # Save all results
        comparison_data = {
            'comparison_timestamp': evaluator.results_history[0].evaluation_timestamp,
            'strategies': []
        }
        
        for result in results:
            comparison_data['strategies'].append({
                'strategy_name': result.strategy_name,
                'overall_score': result.overall_score,
                'signal_quality': result.signal_quality,
                'calibration': result.calibration,
                'information': result.information,
                'performance': result.performance
            })
        
        with open(args.output, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        print(f"\nComparison saved to: {args.output}")


def evaluate_tfa_model(args):
    """Evaluate a trained TFA model."""
    import sys
    sys.path.append(str(pathlib.Path(__file__).parent))
    
    from trainers.tfa import train_tfa_fast
    
    print("Evaluating TFA model...")
    print("Note: This will retrain the model for evaluation purposes.")
    
    # You would need to modify this to load an existing model
    # For now, this is a placeholder
    print("TFA model evaluation not yet implemented.")
    print("Use the training pipeline which now includes evaluation.")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive Strategy Evaluation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate a single strategy
  python evaluate_strategies.py single --data strategy_data.json --name "MyStrategy"
  
  # Compare multiple strategies
  python evaluate_strategies.py compare --data-files strategy1.json strategy2.json
  
  # Evaluate with custom threshold
  python evaluate_strategies.py single --data strategy_data.json --threshold 0.6
  
  # Save results to file
  python evaluate_strategies.py single --data strategy_data.json --output results.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Single strategy evaluation
    single_parser = subparsers.add_parser('single', help='Evaluate a single strategy')
    single_parser.add_argument('--data', required=True, help='Path to strategy data file')
    single_parser.add_argument('--name', default='Strategy', help='Strategy name')
    single_parser.add_argument('--threshold', type=float, default=0.5, help='Probability threshold')
    single_parser.add_argument('--output', help='Output file for results')
    
    # Multiple strategy comparison
    compare_parser = subparsers.add_parser('compare', help='Compare multiple strategies')
    compare_parser.add_argument('--data-files', nargs='+', required=True, 
                               help='Paths to strategy data files')
    compare_parser.add_argument('--output', help='Output file for comparison')
    
    # TFA model evaluation
    tfa_parser = subparsers.add_parser('tfa', help='Evaluate TFA model')
    tfa_parser.add_argument('--model-path', help='Path to trained TFA model')
    tfa_parser.add_argument('--data-path', help='Path to evaluation data')
    
    args = parser.parse_args()
    
    if args.command == 'single':
        evaluate_single_strategy(args)
    elif args.command == 'compare':
        compare_multiple_strategies(args)
    elif args.command == 'tfa':
        evaluate_tfa_model(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
