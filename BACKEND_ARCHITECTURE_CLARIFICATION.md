# Backend Architecture Clarification

## Problem Statement

Current architecture has overlapping responsibilities between Router, AllocationManager, and Strategy allocation decisions, creating redundancy and conflicts. Need to clarify orthogonal roles for maximum profit.

## Correct Architecture (Orthogonal & Non-Redundant)

### 1. Strategy Layer
**Single Responsibility**: Produce probability (0-1) representing "up" confidence
- Input: Market data (bars, features, etc.)
- Output: Single probability value (0-1)
- **NO** instrument selection
- **NO** position sizing
- **NO** allocation decisions

```cpp
class BaseStrategy {
    virtual double calculate_probability(const std::vector<Bar>& bars, int current_index) = 0;
    // REMOVE: get_allocation_decisions()
    // REMOVE: get_router_config() 
};
```

### 2. AllocationManager Layer
**Single Responsibility**: Convert probability → optimal instrument allocation for maximum profit
- Input: probability + current portfolio state + market conditions
- Output: Instrument allocation decisions (which instruments, what weights)
- Handles: QQQ vs TQQQ vs SQQQ selection based on conviction
- Handles: Risk management (position transitions, stop losses, etc.)
- **Strategy-agnostic**: Works with any strategy's probability

```cpp
struct AllocationDecision {
    std::string instrument;     // "QQQ", "TQQQ", "SQQQ", etc.
    double target_weight;       // -1.0 to +1.0 (% of equity)
    std::string reason;         // Why this allocation
};

class AllocationManager {
    std::vector<AllocationDecision> allocate(
        double probability,                    // From strategy
        const Portfolio& current_portfolio,   // Current state
        const MarketConditions& conditions    // Volatility, spreads, etc.
    );
};
```

### 3. Sizer Layer  
**Single Responsibility**: Convert target weights → exact quantities (shares)
- Input: AllocationDecision (instrument + target_weight) + portfolio + prices
- Output: Exact quantity to trade (shares)
- Handles: Equity calculation, price conversion, lot sizing, minimum notional
- **Pure mathematical conversion**: weight → shares

```cpp
class AdvancedSizer {
    double calculate_target_quantity(
        const std::string& instrument,
        double target_weight,           // From AllocationManager
        const Portfolio& portfolio,     // Current positions
        const std::vector<double>& prices
    );
};
```

### 4. PositionCoordinator Layer
**Single Responsibility**: Prevent conflicts and enforce execution constraints
- Input: Multiple AllocationDecisions from AllocationManager
- Output: Approved/rejected/modified decisions
- Handles: Conflict prevention, frequency limits, sequential transitions
- **Strategy-agnostic**: Applies universal trading rules

### 5. Execution Layer (Runner)
**Single Responsibility**: Execute approved trades
- Input: Approved AllocationDecisions + exact quantities
- Output: Actual trades, P&L, audit events
- Handles: Order execution, fill simulation, audit logging

## Information Flow (Clean Pipeline)

```
Strategy → probability (0-1)
    ↓
AllocationManager → instrument decisions (QQQ/TQQQ/SQQQ + weights)
    ↓  
Sizer → exact quantities (shares)
    ↓
PositionCoordinator → conflict-free execution plan
    ↓
Runner → actual trades + P&L
```

## Key Benefits

1. **Single Responsibility**: Each component has one clear job
2. **Strategy Agnostic**: AllocationManager works with any strategy
3. **Maximum Profit**: AllocationManager optimizes instrument selection for profit
4. **No Conflicts**: Clear separation prevents architectural bugs
5. **Testable**: Each component can be tested independently

## What Gets Removed

1. **Router**: Redundant with AllocationManager (remove entirely)
2. **Strategy.get_allocation_decisions()**: Strategies only produce probabilities
3. **Strategy.get_router_config()**: No longer needed
4. **Dynamic allocation path**: All strategies use same pipeline

## AllocationManager Profit Optimization

The AllocationManager becomes the **profit maximization engine**:

- **High conviction (p > 0.8)**: Use TQQQ (3x leverage)
- **Medium conviction (0.6 < p < 0.8)**: Use QQQ (1x)  
- **Low conviction (0.4 < p < 0.6)**: Stay in cash or reduce position
- **Medium bearish (0.2 < p < 0.4)**: Short QQQ
- **High bearish (p < 0.2)**: Use SQQQ (3x inverse)

Plus sophisticated features:
- Position transitions (3x → 1x → cash → short)
- Stop losses and profit taking
- Volatility-based sizing
- Transaction cost optimization
- Momentum and mean reversion detection

## Implementation Priority

1. Remove Router from runner.cpp
2. Integrate AllocationManager into runner.cpp for all strategies  
3. Remove get_allocation_decisions() from BaseStrategy
4. Test with existing strategies (should improve performance)
