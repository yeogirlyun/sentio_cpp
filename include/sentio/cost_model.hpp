#pragma once
#include <string>
#include <cmath>

namespace sentio {

// Alpaca Trading Cost Model
struct AlpacaCostModel {
  // Commission structure (Alpaca is commission-free for stocks/ETFs)
  static constexpr double commission_per_share = 0.0;
  static constexpr double min_commission = 0.0;
  
  // SEC fees (for sells only)
  static constexpr double sec_fee_rate = 0.0000278; // $0.0278 per $1000 of principal
  
  // FINRA Trading Activity Fee (TAF) - for sells only
  static constexpr double taf_per_share = 0.000145; // $0.000145 per share (max $7.27 per trade)
  static constexpr double taf_max_per_trade = 7.27;
  
  // Slippage model based on market impact
  struct SlippageParams {
    double base_slippage_bps = 1.0;    // Base 1 bps slippage
    double volume_impact_factor = 0.5;  // Additional slippage based on volume
    double volatility_factor = 0.3;     // Additional slippage based on volatility
    double max_slippage_bps = 10.0;     // Cap at 10 bps
  };
  
  static SlippageParams default_slippage;
  
  // Calculate total transaction costs for a trade
  static double calculate_fees([[maybe_unused]] const std::string& symbol, 
                              double quantity, 
                              double price, 
                              bool is_sell) {
    double notional = std::abs(quantity) * price;
    double total_fees = 0.0;
    
    // Commission (free for Alpaca)
    total_fees += commission_per_share * std::abs(quantity);
    total_fees = std::max(total_fees, min_commission);
    
    if (is_sell) {
      // SEC fees (sells only)
      total_fees += notional * sec_fee_rate;
      
      // FINRA TAF (sells only)
      double taf = std::abs(quantity) * taf_per_share;
      total_fees += std::min(taf, taf_max_per_trade);
    }
    
    return total_fees;
  }
  
  // Calculate slippage based on trade characteristics
  static double calculate_slippage_bps(double quantity,
                                      double price, 
                                      double avg_volume,
                                      double volatility,
                                      const SlippageParams& params = default_slippage) {
    double notional = std::abs(quantity) * price;
    
    // Base slippage
    double slippage_bps = params.base_slippage_bps;
    
    // Volume impact (higher for larger trades relative to average volume)
    if (avg_volume > 0) {
      double volume_ratio = notional / (avg_volume * price);
      slippage_bps += params.volume_impact_factor * std::sqrt(volume_ratio) * 100; // Convert to bps
    }
    
    // Volatility impact
    slippage_bps += params.volatility_factor * volatility * 10000; // Convert annual vol to bps
    
    // Cap the slippage
    return std::min(slippage_bps, params.max_slippage_bps);
  }
  
  // Apply slippage to execution price
  static double apply_slippage(double market_price, 
                              double slippage_bps, 
                              bool is_buy) {
    double slippage_factor = slippage_bps / 10000.0; // Convert bps to decimal
    
    if (is_buy) {
      return market_price * (1.0 + slippage_factor); // Pay more when buying
    } else {
      return market_price * (1.0 - slippage_factor); // Receive less when selling
    }
  }
  
  // Complete cost calculation including fees and slippage
  static std::pair<double, double> calculate_total_costs(
      const std::string& symbol,
      double quantity,
      double market_price,
      double avg_volume = 1000000, // Default 1M average volume
      double volatility = 0.20,    // Default 20% annual volatility
      const SlippageParams& params = default_slippage) {
    
    bool is_sell = quantity < 0;
    bool is_buy = quantity > 0;
    
    // Calculate fees
    double fees = calculate_fees(symbol, quantity, market_price, is_sell);
    
    // Calculate slippage
    double slippage_bps = calculate_slippage_bps(quantity, market_price, avg_volume, volatility, params);
    double execution_price = apply_slippage(market_price, slippage_bps, is_buy);
    
    // Slippage cost (difference from market price)
    double slippage_cost = std::abs(quantity) * std::abs(execution_price - market_price);
    
    return {fees, slippage_cost};
  }
};

// Static member definition
inline AlpacaCostModel::SlippageParams AlpacaCostModel::default_slippage = {};

} // namespace sentio