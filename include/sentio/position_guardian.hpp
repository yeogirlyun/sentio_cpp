#pragma once
#include "side.hpp"
#include "family_mapper.hpp"
#include "core.hpp"
#include <mutex>
#include <unordered_map>
#include <optional>
#include <vector>
#include <chrono>
#include <memory>
#include <string>

namespace sentio {

struct Policy {
    bool   allow_conflicts = false; // hard OFF for this use-case
    double max_gross_shares = 1e9;  // guardrail
    double min_flip_bps = 0.0;      // optional flip friction to avoid churn
    int    cooldown_ms = 0;         // optional per-family cooldown after flip
};

struct PositionSnapshot {
    PositionSide   side{PositionSide::Flat};     // net family side
    double qty{0.0};             // positive magnitude
    double avg_px{0.0};          // informational
    uint64_t epoch{0};           // increments on each committed change
    std::chrono::steady_clock::time_point last_change{};
};

struct PlanLeg {
    std::string symbol;
    PositionSide   side;                 // Long = buy, Short = sell/short
    double qty;                  // positive magnitude
    std::string reason;          // "CLOSE_OPPOSITE" / "OPEN_TARGET" / "RESIZE"
};

struct OrderPlan {
    ExposureKey key;
    uint64_t epoch_before;       // optimistic check
    uint64_t reservation_id;     // for idempotency
    std::vector<PlanLeg> legs;   // ordered legs
};

struct Desire {
    // Strategy asks for this net outcome for the family
    PositionSide   target_side{PositionSide::Flat};
    double target_qty{0.0};       // positive magnitude
    std::string preferred_symbol; // e.g., choose TQQQ for strong long
};

class PositionGuardian {
public:
    using Clock = std::chrono::steady_clock;

    PositionGuardian(const FamilyMapper& mapper)
      : mapper_(mapper) {}

    // Inject broker truth + open orders to seed/refresh snapshots.
    void sync_from_broker(const std::string& account,
                          const std::vector<Position>& positions,
                          const std::vector<std::string>& open_orders);

    // Main entry: produce a conflict-free, atomic plan.
    // Thread-safe: locks the family during planning.
    std::optional<OrderPlan> plan(const std::string& account,
                                  const std::string& symbol,
                                  const Desire& desire,
                                  const Policy& policy);

    // Commit hook after successful router acceptance to advance epoch.
    // (You can also advance on fill callbacks—just be consistent.)
    void commit(const OrderPlan& plan);

    // Read-only view (for monitoring/metrics)
    PositionSnapshot snapshot(const ExposureKey& key) const;

private:
    struct Cell {
        mutable std::mutex m;
        PositionSnapshot ps;
        // Track in-flight reserved qty towards target to avoid double-alloc
        double reserved_long{0.0};
        double reserved_short{0.0};
        uint64_t next_reservation{1};
    };

    const FamilyMapper& mapper_;
    mutable std::mutex map_mu_;
    std::unordered_map<ExposureKey, std::unique_ptr<Cell>, ExposureKeyHash> cells_;

    Cell& cell_for(const ExposureKey& key);
    static bool flip_cooldown_active(const PositionSnapshot& ps, const Policy& pol, Clock::time_point now);
};

} // namespace sentio
