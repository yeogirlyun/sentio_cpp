#include "../include/sentio/sanity.hpp"
#include "../include/sentio/sim_data.hpp"
#include "../include/sentio/property_test.hpp"
#include <iostream>
#include <unordered_map>
#include <vector>

// Example integration showing how to use sanity checks in a real strategy system
class ExamplePriceBook : public sentio::PriceBook {
private:
    std::unordered_map<std::string, sentio::Bar> latest_;
    
public:
    void upsert_latest(const std::string& instrument, const sentio::Bar& b) override {
        latest_[instrument] = b;
    }
    
    const sentio::Bar* get_latest(const std::string& instrument) const override {
        auto it = latest_.find(instrument);
        return it == latest_.end() ? nullptr : &it->second;
    }
    
    bool has_instrument(const std::string& instrument) const override {
        return latest_.count(instrument) > 0;
    }
    
    std::size_t size() const override {
        return latest_.size();
    }
};

// Example strategy that generates signals
class ExampleStrategy {
public:
    struct Signal {
        sentio::SigType type;
        double confidence;
        std::string instrument;
    };
    
    std::vector<Signal> generate_signals(const std::vector<std::pair<std::int64_t, sentio::Bar>>& bars) {
        std::vector<Signal> signals;
        
        // Simple strategy: buy when price goes up, sell when it goes down
        for (size_t i = 1; i < bars.size(); ++i) {
            const auto& prev = bars[i-1].second;
            const auto& curr = bars[i].second;
            
            if (curr.close > prev.close * 1.01) { // 1% up
                signals.push_back({sentio::SigType::BUY, 0.8, "TQQQ"});
            } else if (curr.close < prev.close * 0.99) { // 1% down
                signals.push_back({sentio::SigType::SELL, 0.8, "TQQQ"});
            }
        }
        
        return signals;
    }
};

// Example order execution
class ExampleOrderManager {
public:
    struct Order {
        std::string instrument;
        double qty;
        double price;
        std::string side;
    };
    
    std::vector<Order> execute_signals(const std::vector<ExampleStrategy::Signal>& signals,
                                      const ExamplePriceBook& pb) {
        std::vector<Order> orders;
        
        for (const auto& signal : signals) {
            const auto* bar = pb.get_latest(signal.instrument);
            if (!bar) continue;
            
            double qty = 100.0; // Fixed position size
            if (signal.type == sentio::SigType::SELL) qty = -qty;
            
            orders.push_back({
                signal.instrument,
                qty,
                bar->close,
                signal.type == sentio::SigType::BUY ? "BUY" : "SELL"
            });
        }
        
        return orders;
    }
};

int main() {
    using namespace sentio;
    
    std::cout << "=== Sanity System Integration Example ===" << std::endl;
    
    // 1. Generate test data
    std::cout << "\n1. Generating test data..." << std::endl;
    SimCfg cfg;
    cfg.minutes = 100;
    cfg.seed = 12345;
    auto bars = generate_minute_series(cfg);
    std::cout << "Generated " << bars.size() << " bars" << std::endl;
    
    // 2. Data quality checks
    std::cout << "\n2. Running data quality checks..." << std::endl;
    SanityReport data_rep;
    sanity::check_bar_monotonic(bars, 60, data_rep);
    sanity::check_bar_values_finite(bars, data_rep);
    
    if (!data_rep.ok()) {
        std::cout << "❌ Data quality checks FAILED:" << std::endl;
        for (const auto& issue : data_rep.issues) {
            std::cout << "  " << issue.where << ": " << issue.what << std::endl;
        }
        return 1;
    }
    std::cout << "✅ Data quality checks PASSED" << std::endl;
    
    // 3. Load data into price book
    std::cout << "\n3. Loading data into price book..." << std::endl;
    ExamplePriceBook pb;
    for (const auto& bar : bars) {
        pb.upsert_latest("QQQ", bar.second);
        // Create TQQQ as 3x leveraged version
        auto leveraged = bar.second;
        leveraged.open *= 3;
        leveraged.high *= 3;
        leveraged.low *= 3;
        leveraged.close *= 3;
        pb.upsert_latest("TQQQ", leveraged);
    }
    std::cout << "Loaded " << pb.size() << " instruments" << std::endl;
    
    // 4. Price book coherence check
    std::cout << "\n4. Checking price book coherence..." << std::endl;
    SanityReport pb_rep;
    sanity::check_pricebook_coherence(pb, {"QQQ", "TQQQ"}, pb_rep);
    
    if (!pb_rep.ok()) {
        std::cout << "❌ Price book coherence checks FAILED:" << std::endl;
        for (const auto& issue : pb_rep.issues) {
            std::cout << "  " << issue.where << ": " << issue.what << std::endl;
        }
        return 1;
    }
    std::cout << "✅ Price book coherence checks PASSED" << std::endl;
    
    // 5. Strategy signal generation
    std::cout << "\n5. Generating strategy signals..." << std::endl;
    ExampleStrategy strategy;
    auto signals = strategy.generate_signals(bars);
    std::cout << "Generated " << signals.size() << " signals" << std::endl;
    
    // 6. Signal quality checks
    std::cout << "\n6. Checking signal quality..." << std::endl;
    SanityReport signal_rep;
    for (const auto& signal : signals) {
        sanity::check_signal_confidence_range(signal.confidence, signal_rep, 0);
        sanity::check_routed_instrument_has_price(pb, signal.instrument, signal_rep, 0);
    }
    
    if (!signal_rep.ok()) {
        std::cout << "❌ Signal quality checks FAILED:" << std::endl;
        for (const auto& issue : signal_rep.issues) {
            std::cout << "  " << issue.where << ": " << issue.what << std::endl;
        }
        return 1;
    }
    std::cout << "✅ Signal quality checks PASSED" << std::endl;
    
    // 7. Order execution
    std::cout << "\n7. Executing orders..." << std::endl;
    ExampleOrderManager order_mgr;
    auto orders = order_mgr.execute_signals(signals, pb);
    std::cout << "Executed " << orders.size() << " orders" << std::endl;
    
    // 8. Order quality checks
    std::cout << "\n8. Checking order quality..." << std::endl;
    SanityReport order_rep;
    for (const auto& order : orders) {
        sanity::check_order_qty_min(std::abs(order.qty), 1.0, order_rep, 0);
        sanity::check_order_side_qty_sign_consistency(order.side, order.qty, order_rep, 0);
    }
    
    if (!order_rep.ok()) {
        std::cout << "❌ Order quality checks FAILED:" << std::endl;
        for (const auto& issue : order_rep.issues) {
            std::cout << "  " << issue.where << ": " << issue.what << std::endl;
        }
        return 1;
    }
    std::cout << "✅ Order quality checks PASSED" << std::endl;
    
    // 9. P&L simulation
    std::cout << "\n9. Simulating P&L..." << std::endl;
    AccountState account{10000.0, 0.0, 10000.0};
    std::unordered_map<std::string, Position> positions;
    
    // Execute orders
    for (const auto& order : orders) {
        auto& pos = positions[order.instrument];
        double cash_delta = -order.qty * order.price; // Buy: negative cash, Sell: positive cash
        account.cash += cash_delta;
        
        // Update position
        if (pos.qty == 0.0) {
            pos.qty = order.qty;
            pos.avg_px = order.price;
        } else {
            // Simple average price calculation
            double total_qty = pos.qty + order.qty;
            if (total_qty != 0.0) {
                pos.avg_px = (pos.qty * pos.avg_px + order.qty * order.price) / total_qty;
                pos.qty = total_qty;
            }
        }
    }
    
    // Mark to market
    double mtm = 0.0;
    for (const auto& pos : positions) {
        const auto* bar = pb.get_latest(pos.first);
        if (bar) {
            mtm += pos.second.qty * bar->close;
        }
    }
    account.equity = account.cash + account.realized + mtm;
    
    std::cout << "Final account state:" << std::endl;
    std::cout << "  Cash: $" << account.cash << std::endl;
    std::cout << "  Realized P&L: $" << account.realized << std::endl;
    std::cout << "  Unrealized P&L: $" << mtm << std::endl;
    std::cout << "  Total Equity: $" << account.equity << std::endl;
    
    // 10. P&L consistency check
    std::cout << "\n10. Checking P&L consistency..." << std::endl;
    SanityReport pnl_rep;
    sanity::check_equity_consistency(account, positions, pb, pnl_rep);
    
    if (!pnl_rep.ok()) {
        std::cout << "❌ P&L consistency checks FAILED:" << std::endl;
        for (const auto& issue : pnl_rep.issues) {
            std::cout << "  " << issue.where << ": " << issue.what << std::endl;
        }
        return 1;
    }
    std::cout << "✅ P&L consistency checks PASSED" << std::endl;
    
    // 11. Property-based testing
    std::cout << "\n11. Running property-based tests..." << std::endl;
    std::vector<PropCase> properties;
    
    properties.push_back({"All prices are positive", [&]() {
        for (const auto& bar : bars) {
            if (bar.second.open <= 0 || bar.second.high <= 0 || 
                bar.second.low <= 0 || bar.second.close <= 0) {
                return false;
            }
        }
        return true;
    }});
    
    properties.push_back({"All signals have valid confidence", [&]() {
        for (const auto& signal : signals) {
            if (signal.confidence < 0.0 || signal.confidence > 1.0) {
                return false;
            }
        }
        return true;
    }});
    
    properties.push_back({"All orders have valid quantities", [&]() {
        for (const auto& order : orders) {
            if (!std::isfinite(order.qty) || order.qty == 0.0) {
                return false;
            }
        }
        return true;
    }});
    
    int prop_result = run_properties(properties);
    if (prop_result != 0) {
        std::cout << "❌ Property tests FAILED" << std::endl;
        return 1;
    }
    std::cout << "✅ Property tests PASSED" << std::endl;
    
    // 12. Summary
    std::cout << "\n=== SANITY CHECK SUMMARY ===" << std::endl;
    std::cout << "✅ All sanity checks PASSED" << std::endl;
    std::cout << "✅ System is ready for production use" << std::endl;
    std::cout << "\nKey metrics:" << std::endl;
    std::cout << "  - Bars processed: " << bars.size() << std::endl;
    std::cout << "  - Signals generated: " << signals.size() << std::endl;
    std::cout << "  - Orders executed: " << orders.size() << std::endl;
    std::cout << "  - Final equity: $" << account.equity << std::endl;
    std::cout << "  - Total return: " << ((account.equity - 10000.0) / 10000.0 * 100) << "%" << std::endl;
    
    return 0;
}
