"""
Universal Strategy Evaluation Framework

This module provides comprehensive evaluation metrics for all probability-based
trading strategies in the Sentio system.
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve


@dataclass
class EvaluationResults:
    """Container for evaluation results."""
    signal_quality: Dict[str, float]
    calibration: Dict[str, float]
    information: Dict[str, float]
    performance: Dict[str, float]
    overall_score: float
    strategy_name: str
    evaluation_timestamp: str


class StrategyEvaluator:
    """
    Universal evaluator for probability-based trading strategies.
    
    This class provides comprehensive evaluation metrics that can be applied
    to any strategy that outputs probability signals.
    """
    
    def __init__(self, strategy_name: str = "Unknown"):
        self.strategy_name = strategy_name
        self.results_history = []
    
    def evaluate_probability_calibration(self, predictions: torch.Tensor, 
                                      actual_returns: np.ndarray, 
                                      bins: int = 10) -> Dict[str, Any]:
        """
        Evaluate how well-calibrated probability predictions are.
        
        Args:
            predictions: Raw model outputs (logits)
            actual_returns: Binary returns (1 = up, 0 = down)
            bins: Number of calibration bins
            
        Returns:
            Dictionary containing calibration metrics
        """
        # Convert predictions to probabilities
        probs = torch.sigmoid(predictions).cpu().numpy()
        
        # Create bins
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Calculate calibration
        bin_accuracies = []
        bin_predictions = []
        bin_counts = []
        
        for i in range(bins):
            mask = (probs >= bin_edges[i]) & (probs < bin_edges[i + 1])
            if mask.sum() > 0:
                bin_acc = actual_returns[mask].mean()  # Actual up rate
                bin_pred = probs[mask].mean()          # Predicted up rate
                bin_accuracies.append(bin_acc)
                bin_predictions.append(bin_pred)
                bin_counts.append(mask.sum())
            else:
                bin_accuracies.append(0.0)
                bin_predictions.append(0.0)
                bin_counts.append(0)
        
        # Calibration Error
        calibration_error = np.mean([abs(acc - pred) for acc, pred in 
                                   zip(bin_accuracies, bin_predictions)])
        
        return {
            'calibration_error': calibration_error,
            'bin_accuracies': bin_accuracies,
            'bin_predictions': bin_predictions,
            'bin_counts': bin_counts,
            'bin_centers': bin_centers
        }
    
    def evaluate_information_content(self, predictions: torch.Tensor, 
                                   actual_returns: np.ndarray) -> Dict[str, float]:
        """
        Evaluate information content of probability signals.
        
        Args:
            predictions: Raw model outputs (logits)
            actual_returns: Binary returns (1 = up, 0 = down)
            
        Returns:
            Dictionary containing information metrics
        """
        probs = torch.sigmoid(predictions).cpu().numpy()
        
        # Log Loss (Cross-Entropy)
        log_loss = -np.mean(actual_returns * np.log(probs + 1e-15) + 
                           (1 - actual_returns) * np.log(1 - probs + 1e-15))
        
        # Brier Score
        brier_score = np.mean((probs - actual_returns) ** 2)
        
        # Information Ratio
        random_entropy = -0.5 * np.log(0.5) - 0.5 * np.log(0.5)
        signal_entropy = -np.mean(probs * np.log(probs + 1e-15) + 
                                 (1 - probs) * np.log(1 - probs + 1e-15))
        information_ratio = (random_entropy - signal_entropy) / random_entropy
        
        # AUC Score
        try:
            auc_score = roc_auc_score(actual_returns, probs)
        except ValueError:
            auc_score = 0.5  # Random performance
        
        return {
            'log_loss': log_loss,
            'brier_score': brier_score,
            'information_ratio': information_ratio,
            'auc_score': auc_score
        }
    
    def evaluate_trading_performance(self, predictions: torch.Tensor, 
                                   actual_returns: np.ndarray, 
                                   threshold: float = 0.5,
                                   actual_prices: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate trading performance of probability signals.
        
        Args:
            predictions: Raw model outputs (logits)
            actual_returns: Binary returns (1 = up, 0 = down)
            threshold: Probability threshold for binary decisions
            actual_prices: Optional price data for return calculations
            
        Returns:
            Dictionary containing trading performance metrics
        """
        probs = torch.sigmoid(predictions).cpu().numpy()
        
        # Binary predictions
        binary_preds = (probs > threshold).astype(int)
        
        # Confusion Matrix
        tp = np.sum((binary_preds == 1) & (actual_returns == 1))
        tn = np.sum((binary_preds == 0) & (actual_returns == 0))
        fp = np.sum((binary_preds == 1) & (actual_returns == 0))
        fn = np.sum((binary_preds == 0) & (actual_returns == 1))
        
        # Basic Metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        
        # Trading Performance Metrics
        total_trades = tp + fp
        win_rate = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        # Calculate strategy returns
        if actual_prices is not None and len(actual_prices) > 1:
            # Use actual price data for return calculations
            price_returns = np.diff(actual_prices) / actual_prices[:-1]
            strategy_returns = np.where(binary_preds[1:] == 1, price_returns, -price_returns)
        else:
            # Use binary returns as proxy
            strategy_returns = np.where(binary_preds == 1, actual_returns, -actual_returns)
        
        # Risk-Adjusted Metrics
        if len(strategy_returns) > 1 and np.std(strategy_returns) > 0:
            # Annualized Sharpe Ratio
            sharpe_ratio = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
            
            # Monthly Projected Return (MPR)
            annual_return = np.mean(strategy_returns) * 252
            monthly_projected_return = annual_return / 12
            
            # Daily trading frequency
            num_daily_trades = total_trades / len(strategy_returns) if len(strategy_returns) > 0 else 0
        else:
            sharpe_ratio = 0.0
            monthly_projected_return = 0.0
            num_daily_trades = 0.0
        
        return {
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': accuracy,
            'sharpe_ratio': sharpe_ratio,
            'monthly_projected_return': monthly_projected_return,
            'num_daily_trades': num_daily_trades,
            'confusion_matrix': {'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn},
            'total_trades': total_trades,
            'win_rate': win_rate
        }
    
    def evaluate_strategy_signal(self, predictions: torch.Tensor, 
                               actual_returns: np.ndarray, 
                               actual_prices: Optional[np.ndarray] = None,
                               threshold: float = 0.5,
                               verbose: bool = True) -> EvaluationResults:
        """
        Universal evaluation framework for all probability-based strategies.
        
        Args:
            predictions: Raw model outputs (logits)
            actual_returns: Binary returns (1 = up, 0 = down)
            actual_prices: Optional price data for additional metrics
            threshold: Probability threshold for binary decisions
            verbose: Whether to print detailed results
            
        Returns:
            EvaluationResults object containing all metrics
        """
        import datetime
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"STRATEGY EVALUATION: {self.strategy_name}")
            print(f"{'='*60}")
        
        # Convert to probabilities
        probs = torch.sigmoid(predictions).cpu().numpy()
        
        # 1. Basic Signal Quality
        signal_quality = {
            'min_prob': float(probs.min()),
            'max_prob': float(probs.max()),
            'mean_prob': float(probs.mean()),
            'std_prob': float(probs.std()),
            'signal_strength': 'Strong' if probs.std() > 0.15 else 'Weak'
        }
        
        if verbose:
            print(f"\n📊 SIGNAL QUALITY:")
            print(f"   Probability Range: {signal_quality['min_prob']:.3f} - {signal_quality['max_prob']:.3f}")
            print(f"   Probability Mean:  {signal_quality['mean_prob']:.3f}")
            print(f"   Probability Std:   {signal_quality['std_prob']:.3f}")
            print(f"   Signal Strength:   {signal_quality['signal_strength']}")
        
        # 2. Calibration Analysis
        calib = self.evaluate_probability_calibration(predictions, actual_returns)
        if verbose:
            print(f"\n🎯 CALIBRATION:")
            print(f"   Calibration Error: {calib['calibration_error']:.4f}")
            print(f"   Calibration:        {'Good' if calib['calibration_error'] < 0.05 else 'Poor'}")
        
        # 3. Information Content
        info = self.evaluate_information_content(predictions, actual_returns)
        if verbose:
            print(f"\n📈 INFORMATION CONTENT:")
            print(f"   Log Loss:           {info['log_loss']:.4f}")
            print(f"   Brier Score:        {info['brier_score']:.4f}")
            print(f"   Information Ratio:  {info['information_ratio']:.3f}")
            print(f"   AUC Score:          {info['auc_score']:.3f}")
            print(f"   Information:        {'High' if info['information_ratio'] > 0.1 else 'Low'}")
        
        # 4. Trading Performance
        perf = self.evaluate_trading_performance(predictions, actual_returns, threshold, actual_prices)
        if verbose:
            print(f"\n💰 TRADING PERFORMANCE:")
            print(f"   Accuracy:           {perf['accuracy']:.3f}")
            print(f"   Precision:          {perf['precision']:.3f}")
            print(f"   Recall:             {perf['recall']:.3f}")
            print(f"   F1 Score:           {perf['f1_score']:.3f}")
            print(f"   Sharpe Ratio:       {perf['sharpe_ratio']:.3f}")
            print(f"   MPR (Monthly):      {perf['monthly_projected_return']:.2%}")
            print(f"   Daily Trades:       {perf['num_daily_trades']:.1f}")
            print(f"   Total Trades:       {perf['total_trades']}")
            print(f"   Win Rate:           {perf['win_rate']:.3f}")
        
        # 5. Threshold Analysis
        if verbose:
            print(f"\n🎚️  THRESHOLD ANALYSIS:")
            thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
            for thresh in thresholds:
                perf_thresh = self.evaluate_trading_performance(predictions, actual_returns, thresh, actual_prices)
                print(f"   Threshold {thresh}: Accuracy={perf_thresh['accuracy']:.3f}, "
                      f"F1={perf_thresh['f1_score']:.3f}, Sharpe={perf_thresh['sharpe_ratio']:.3f}, "
                      f"MPR={perf_thresh['monthly_projected_return']:.2%}, "
                      f"Daily Trades={perf_thresh['num_daily_trades']:.1f}")
        
        # 6. Overall Assessment
        overall_score = (
            (1 - calib['calibration_error']) * 0.25 +  # Calibration weight
            info['information_ratio'] * 0.25 +           # Information weight
            perf['f1_score'] * 0.25 +                   # Trading performance weight
            info['auc_score'] * 0.25                    # AUC weight
        )
        
        if verbose:
            print(f"\n🏆 OVERALL ASSESSMENT:")
            print(f"   Overall Score:      {overall_score:.3f}")
            rating = ('Excellent' if overall_score > 0.7 else 
                     'Good' if overall_score > 0.5 else 
                     'Fair' if overall_score > 0.3 else 'Poor')
            print(f"   Rating:             {rating}")
        
        # Create results object
        results = EvaluationResults(
            signal_quality=signal_quality,
            calibration=calib,
            information=info,
            performance=perf,
            overall_score=overall_score,
            strategy_name=self.strategy_name,
            evaluation_timestamp=datetime.datetime.now().isoformat()
        )
        
        # Store in history
        self.results_history.append(results)
        
        return results
    
    def compare_strategies(self, strategy_results: List[EvaluationResults]) -> None:
        """
        Compare multiple strategies side by side.
        
        Args:
            strategy_results: List of EvaluationResults from different strategies
        """
        print(f"\n{'='*80}")
        print(f"STRATEGY COMPARISON")
        print(f"{'='*80}")
        
        # Create comparison table
        print(f"\n{'Strategy':<20} {'Overall':<8} {'F1':<6} {'AUC':<6} {'Sharpe':<8} {'MPR':<8} {'Daily':<6} {'Calib':<8}")
        print(f"{'-'*90}")
        
        for result in strategy_results:
            print(f"{result.strategy_name:<20} "
                  f"{result.overall_score:<8.3f} "
                  f"{result.performance['f1_score']:<6.3f} "
                  f"{result.information['auc_score']:<6.3f} "
                  f"{result.performance['sharpe_ratio']:<8.3f} "
                  f"{result.performance['monthly_projected_return']:<8.2%} "
                  f"{result.performance['num_daily_trades']:<6.1f} "
                  f"{result.calibration['calibration_error']:<8.3f}")
    
    def save_results(self, results: EvaluationResults, filepath: str) -> None:
        """
        Save evaluation results to file.
        
        Args:
            results: EvaluationResults object
            filepath: Path to save results
        """
        import json
        import numpy as np
        
        def convert_numpy_types(obj):
            """Recursively convert NumPy types to Python native types."""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            else:
                return obj
        
        # Convert to serializable format
        data = {
            'strategy_name': results.strategy_name,
            'evaluation_timestamp': results.evaluation_timestamp,
            'overall_score': convert_numpy_types(results.overall_score),
            'signal_quality': convert_numpy_types(results.signal_quality),
            'calibration': convert_numpy_types(results.calibration),
            'information': convert_numpy_types(results.information),
            'performance': convert_numpy_types(results.performance)
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Results saved to: {filepath}")


# Convenience functions for quick evaluation
def quick_evaluate(predictions: torch.Tensor, actual_returns: np.ndarray, 
                  strategy_name: str = "Strategy") -> EvaluationResults:
    """
    Quick evaluation function for immediate use.
    
    Args:
        predictions: Raw model outputs (logits)
        actual_returns: Binary returns (1 = up, 0 = down)
        strategy_name: Name of the strategy
        
    Returns:
        EvaluationResults object
    """
    evaluator = StrategyEvaluator(strategy_name)
    return evaluator.evaluate_strategy_signal(predictions, actual_returns)


def compare_strategies_quick(strategy_data: Dict[str, Tuple[torch.Tensor, np.ndarray]]) -> None:
    """
    Quick comparison of multiple strategies.
    
    Args:
        strategy_data: Dictionary mapping strategy names to (predictions, actual_returns) tuples
    """
    results = []
    for name, (predictions, actual_returns) in strategy_data.items():
        evaluator = StrategyEvaluator(name)
        result = evaluator.evaluate_strategy_signal(predictions, actual_returns, verbose=False)
        results.append(result)
    
    evaluator = StrategyEvaluator("Comparison")
    evaluator.compare_strategies(results)
