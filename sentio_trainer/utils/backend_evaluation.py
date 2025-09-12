"""
Backend Trading System Evaluation Framework

This module evaluates the complete trading pipeline including router, sizer, and runner
components that convert probability signals into actual trading performance.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import json
from datetime import datetime, timedelta


@dataclass
class TradeRecord:
    """Individual trade record."""
    timestamp: str
    instrument: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    signal_probability: float
    signal_confidence: float
    commission: float
    slippage: float
    pnl: float
    holding_period: int  # bars held


@dataclass
class BackendEvaluationResults:
    """Results from backend system evaluation."""
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_pnl: float
    gross_pnl: float
    net_pnl: float
    total_commission: float
    total_slippage: float
    max_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    var_95: float
    expected_shortfall: float
    avg_holding_period: float
    turnover_rate: float
    signal_to_pnl_correlation: float
    evaluation_period: str
    strategy_name: str


class BackendEvaluator:
    """
    Evaluates the complete trading backend including router, sizer, and runner.
    """
    
    def __init__(self, strategy_name: str = "Unknown"):
        self.strategy_name = strategy_name
        self.trades: List[TradeRecord] = []
        self.daily_pnl: List[float] = []
        self.portfolio_values: List[float] = []
        
    def add_trade(self, trade: TradeRecord):
        """Add a trade record."""
        self.trades.append(trade)
    
    def add_daily_pnl(self, pnl: float):
        """Add daily PnL."""
        self.daily_pnl.append(pnl)
    
    def calculate_performance_metrics(self) -> BackendEvaluationResults:
        """Calculate comprehensive performance metrics."""
        
        if not self.trades:
            raise ValueError("No trades recorded for evaluation")
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = sum(1 for t in self.trades if t.pnl < 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        # PnL analysis
        total_pnl = sum(t.pnl for t in self.trades)
        gross_pnl = sum(t.pnl for t in self.trades if t.pnl > 0)
        net_pnl = total_pnl
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        
        # Risk metrics
        pnl_series = np.array([t.pnl for t in self.trades])
        returns = np.array(self.daily_pnl) if self.daily_pnl else pnl_series
        
        max_drawdown = self._calculate_max_drawdown(returns)
        sharpe_ratio = self._calculate_sharpe_ratio(returns)
        sortino_ratio = self._calculate_sortino_ratio(returns)
        calmar_ratio = self._calculate_calmar_ratio(returns, max_drawdown)
        var_95 = np.percentile(returns, 5) if len(returns) > 0 else 0
        expected_shortfall = np.mean(returns[returns <= var_95]) if len(returns) > 0 else 0
        
        # Trading behavior metrics
        avg_holding_period = np.mean([t.holding_period for t in self.trades])
        turnover_rate = self._calculate_turnover_rate()
        
        # Signal effectiveness
        signal_to_pnl_correlation = self._calculate_signal_correlation()
        
        return BackendEvaluationResults(
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            total_pnl=total_pnl,
            gross_pnl=gross_pnl,
            net_pnl=net_pnl,
            total_commission=total_commission,
            total_slippage=total_slippage,
            max_drawdown=max_drawdown,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            calmar_ratio=calmar_ratio,
            var_95=var_95,
            expected_shortfall=expected_shortfall,
            avg_holding_period=avg_holding_period,
            turnover_rate=turnover_rate,
            signal_to_pnl_correlation=signal_to_pnl_correlation,
            evaluation_period=f"{datetime.now().isoformat()}",
            strategy_name=self.strategy_name
        )
    
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate maximum drawdown."""
        if len(returns) == 0:
            return 0.0
        
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe ratio."""
        if len(returns) == 0 or np.std(returns) == 0:
            return 0.0
        
        return np.mean(returns) / np.std(returns) * np.sqrt(252)
    
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino ratio."""
        if len(returns) == 0:
            return 0.0
        
        negative_returns = returns[returns < 0]
        if len(negative_returns) == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        downside_std = np.std(negative_returns)
        if downside_std == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        return np.mean(returns) / downside_std * np.sqrt(252)
    
    def _calculate_calmar_ratio(self, returns: np.ndarray, max_drawdown: float) -> float:
        """Calculate Calmar ratio."""
        if max_drawdown == 0:
            return float('inf') if np.mean(returns) > 0 else 0.0
        
        annual_return = np.mean(returns) * 252
        return annual_return / max_drawdown
    
    def _calculate_turnover_rate(self) -> float:
        """Calculate portfolio turnover rate."""
        if not self.trades:
            return 0.0
        
        total_volume = sum(abs(t.quantity * t.price) for t in self.trades)
        # Assuming average portfolio value (this would need to be tracked)
        avg_portfolio_value = 100000  # Placeholder
        return total_volume / avg_portfolio_value
    
    def _calculate_signal_correlation(self) -> float:
        """Calculate correlation between signal strength and PnL."""
        if len(self.trades) < 2:
            return 0.0
        
        signals = np.array([t.signal_probability for t in self.trades])
        pnls = np.array([t.pnl for t in self.trades])
        
        return np.corrcoef(signals, pnls)[0, 1]
    
    def evaluate_router_performance(self) -> Dict[str, Any]:
        """Evaluate router performance (instrument selection)."""
        if not self.trades:
            return {}
        
        # Group trades by instrument
        instrument_stats = {}
        for trade in self.trades:
            inst = trade.instrument
            if inst not in instrument_stats:
                instrument_stats[inst] = {
                    'trades': 0,
                    'total_pnl': 0,
                    'winning_trades': 0,
                    'total_volume': 0
                }
            
            stats = instrument_stats[inst]
            stats['trades'] += 1
            stats['total_pnl'] += trade.pnl
            stats['total_volume'] += abs(trade.quantity * trade.price)
            if trade.pnl > 0:
                stats['winning_trades'] += 1
        
        # Calculate metrics per instrument
        for inst, stats in instrument_stats.items():
            stats['win_rate'] = stats['winning_trades'] / stats['trades']
            stats['avg_pnl_per_trade'] = stats['total_pnl'] / stats['trades']
            stats['pnl_per_volume'] = stats['total_pnl'] / stats['total_volume']
        
        return instrument_stats
    
    def evaluate_sizer_performance(self) -> Dict[str, Any]:
        """Evaluate sizer performance (position sizing)."""
        if not self.trades:
            return {}
        
        # Analyze position sizes vs. outcomes
        position_sizes = [abs(t.quantity * t.price) for t in self.trades]
        pnls = [t.pnl for t in self.trades]
        
        # Correlation between position size and PnL
        size_pnl_correlation = np.corrcoef(position_sizes, pnls)[0, 1]
        
        # Risk-adjusted returns
        risk_adjusted_returns = [pnl / size for pnl, size in zip(pnls, position_sizes) if size > 0]
        
        return {
            'size_pnl_correlation': size_pnl_correlation,
            'avg_position_size': np.mean(position_sizes),
            'position_size_std': np.std(position_sizes),
            'avg_risk_adjusted_return': np.mean(risk_adjusted_returns),
            'risk_adjusted_return_std': np.std(risk_adjusted_returns)
        }
    
    def evaluate_runner_performance(self) -> Dict[str, Any]:
        """Evaluate runner performance (execution quality)."""
        if not self.trades:
            return {}
        
        # Execution cost analysis
        total_commission = sum(t.commission for t in self.trades)
        total_slippage = sum(t.slippage for t in self.trades)
        total_volume = sum(abs(t.quantity * t.price) for t in self.trades)
        
        commission_rate = total_commission / total_volume if total_volume > 0 else 0
        slippage_rate = total_slippage / total_volume if total_volume > 0 else 0
        
        # Timing analysis
        holding_periods = [t.holding_period for t in self.trades]
        
        return {
            'total_commission': total_commission,
            'total_slippage': total_slippage,
            'commission_rate': commission_rate,
            'slippage_rate': slippage_rate,
            'avg_holding_period': np.mean(holding_periods),
            'holding_period_std': np.std(holding_periods),
            'execution_cost_per_trade': (total_commission + total_slippage) / len(self.trades)
        }
    
    def generate_backend_report(self, verbose: bool = True) -> Dict[str, Any]:
        """Generate comprehensive backend evaluation report."""
        
        results = self.calculate_performance_metrics()
        router_perf = self.evaluate_router_performance()
        sizer_perf = self.evaluate_sizer_performance()
        runner_perf = self.evaluate_runner_performance()
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"BACKEND TRADING SYSTEM EVALUATION: {self.strategy_name}")
            print(f"{'='*80}")
            
            # Overall Performance
            print(f"\n💰 OVERALL PERFORMANCE:")
            print(f"   Total Trades:        {results.total_trades}")
            print(f"   Win Rate:            {results.win_rate:.3f}")
            print(f"   Total PnL:           ${results.total_pnl:,.2f}")
            print(f"   Net PnL:             ${results.net_pnl:,.2f}")
            print(f"   Max Drawdown:        {results.max_drawdown:.3f}")
            print(f"   Sharpe Ratio:        {results.sharpe_ratio:.3f}")
            print(f"   Sortino Ratio:       {results.sortino_ratio:.3f}")
            print(f"   Calmar Ratio:        {results.calmar_ratio:.3f}")
            
            # Router Performance
            print(f"\n🎯 ROUTER PERFORMANCE:")
            for instrument, stats in router_perf.items():
                print(f"   {instrument}:")
                print(f"     Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.3f}")
                print(f"     Total PnL: ${stats['total_pnl']:,.2f}, Avg PnL/Trade: ${stats['avg_pnl_per_trade']:,.2f}")
            
            # Sizer Performance
            print(f"\n📊 SIZER PERFORMANCE:")
            print(f"   Size-PnL Correlation: {sizer_perf['size_pnl_correlation']:.3f}")
            print(f"   Avg Position Size:    ${sizer_perf['avg_position_size']:,.2f}")
            print(f"   Risk-Adjusted Return: {sizer_perf['avg_risk_adjusted_return']:.3f}")
            
            # Runner Performance
            print(f"\n⚡ RUNNER PERFORMANCE:")
            print(f"   Commission Rate:     {runner_perf['commission_rate']:.4f}")
            print(f"   Slippage Rate:        {runner_perf['slippage_rate']:.4f}")
            print(f"   Avg Holding Period:   {runner_perf['avg_holding_period']:.1f} bars")
            print(f"   Execution Cost/Trade:  ${runner_perf['execution_cost_per_trade']:.2f}")
            
            # Signal Effectiveness
            print(f"\n📈 SIGNAL EFFECTIVENESS:")
            print(f"   Signal-PnL Correlation: {results.signal_to_pnl_correlation:.3f}")
            print(f"   Turnover Rate:          {results.turnover_rate:.3f}")
        
        return {
            'overall_performance': results,
            'router_performance': router_perf,
            'sizer_performance': sizer_perf,
            'runner_performance': runner_perf
        }
    
    def save_results(self, filepath: str):
        """Save evaluation results to file."""
        report = self.generate_backend_report(verbose=False)
        
        # Convert to serializable format
        serializable_report = {
            'strategy_name': self.strategy_name,
            'evaluation_timestamp': datetime.now().isoformat(),
            'overall_performance': {
                'total_trades': report['overall_performance'].total_trades,
                'win_rate': report['overall_performance'].win_rate,
                'total_pnl': report['overall_performance'].total_pnl,
                'net_pnl': report['overall_performance'].net_pnl,
                'max_drawdown': report['overall_performance'].max_drawdown,
                'sharpe_ratio': report['overall_performance'].sharpe_ratio,
                'sortino_ratio': report['overall_performance'].sortino_ratio,
                'calmar_ratio': report['overall_performance'].calmar_ratio,
                'signal_to_pnl_correlation': report['overall_performance'].signal_to_pnl_correlation
            },
            'router_performance': report['router_performance'],
            'sizer_performance': report['sizer_performance'],
            'runner_performance': report['runner_performance']
        }
        
        with open(filepath, 'w') as f:
            json.dump(serializable_report, f, indent=2)
        
        print(f"Backend evaluation results saved to: {filepath}")


# Convenience functions
def quick_backend_evaluate(trades: List[TradeRecord], strategy_name: str = "Strategy") -> Dict[str, Any]:
    """Quick backend evaluation."""
    evaluator = BackendEvaluator(strategy_name)
    for trade in trades:
        evaluator.add_trade(trade)
    
    return evaluator.generate_backend_report()


def compare_backend_performance(strategy_trades: Dict[str, List[TradeRecord]]) -> None:
    """Compare backend performance across strategies."""
    print(f"\n{'='*100}")
    print(f"BACKEND PERFORMANCE COMPARISON")
    print(f"{'='*100}")
    
    print(f"\n{'Strategy':<20} {'Trades':<8} {'Win Rate':<10} {'Total PnL':<12} {'Sharpe':<8} {'Max DD':<8} {'Signal Corr':<12}")
    print(f"{'-'*100}")
    
    for strategy_name, trades in strategy_trades.items():
        evaluator = BackendEvaluator(strategy_name)
        for trade in trades:
            evaluator.add_trade(trade)
        
        results = evaluator.calculate_performance_metrics()
        
        print(f"{strategy_name:<20} "
              f"{results.total_trades:<8} "
              f"{results.win_rate:<10.3f} "
              f"${results.total_pnl:<11,.0f} "
              f"{results.sharpe_ratio:<8.3f} "
              f"{results.max_drawdown:<8.3f} "
              f"{results.signal_to_pnl_correlation:<12.3f}")
