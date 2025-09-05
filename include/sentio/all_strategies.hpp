#pragma once

// This file ensures all strategies are included and registered with the factory.
// Include this header once in your main.cpp.

#include "strategy_bollinger_squeeze_breakout.hpp"
#include "strategy_market_making.hpp"
#include "strategy_momentum_volume.hpp"
#include "strategy_opening_range_breakout.hpp"
#include "strategy_order_flow_imbalance.hpp"
#include "strategy_order_flow_scalping.hpp"
#include "strategy_volatility_expansion.hpp"
#include "strategy_vwap_reversion.hpp"