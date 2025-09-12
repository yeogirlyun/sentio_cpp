#include "sentio/position_guardian.hpp"
#include "sentio/family_mapper.hpp"
#include "sentio/core.hpp"
#include <cmath>

namespace sentio {

static inline double sgn(PositionSide s) { 
    return (s==PositionSide::Long? 1.0 : (s==PositionSide::Short? -1.0 : 0.0)); 
}

typename PositionGuardian::Cell& PositionGuardian::cell_for(const ExposureKey& key) {
    std::lock_guard<std::mutex> lg(map_mu_);
    auto it = cells_.find(key);
    if (it == cells_.end()) {
        auto c = std::make_unique<Cell>();
        c->ps.last_change = Clock::now();
        it = cells_.emplace(key, std::move(c)).first;
    }
    return *it->second;
}

bool PositionGuardian::flip_cooldown_active(const PositionSnapshot& ps, const Policy& pol, Clock::time_point now) {
    if (pol.cooldown_ms <= 0) return false;
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - ps.last_change).count();
    return elapsed < pol.cooldown_ms;
}

PositionSnapshot PositionGuardian::snapshot(const ExposureKey& key) const {
    std::lock_guard<std::mutex> lg(map_mu_);
    auto it = cells_.find(key);
    if (it == cells_.end()) return {};
    std::lock_guard<std::mutex> lc(it->second->m);
    return it->second->ps;
}

std::optional<OrderPlan> PositionGuardian::plan(const std::string& account,
                                                const std::string& symbol,
                                                const Desire& desire,
                                                const Policy& policy)
{
    ExposureKey key{account, mapper_.family_for(symbol)};
    auto& cell = cell_for(key);
    auto now = Clock::now();

    std::lock_guard<std::mutex> lk(cell.m);
    auto ps = cell.ps; // copy

    if (policy.allow_conflicts == false) {
        // Always net to one side only
    }

    // Flip friction (optional): if desire flips side for tiny edge, you can block.
    if (ps.side != PositionSide::Flat && desire.target_side != PositionSide::Flat && ps.side != desire.target_side) {
        // You can access bid/ask or last to estimate bps; here we just enforce cooldown if set
        if (flip_cooldown_active(ps, policy, now)) {
            return std::nullopt; // reject for now
        }
    }

    // Compute desired delta relative to current net family exposure
    double net_signed = ps.qty * sgn(ps.side);
    double tgt_signed = desire.target_qty * sgn(desire.target_side);

    // Apply reservations so multiple strategies don't double-commit
    double avail_long  = std::max(0.0, policy.max_gross_shares - (ps.qty + cell.reserved_long));
    double avail_short = std::max(0.0, policy.max_gross_shares - (ps.qty + cell.reserved_short));

    // Bound target by max_gross
    if (tgt_signed > 0) {
        tgt_signed = std::min(tgt_signed, avail_long);
    } else if (tgt_signed < 0) {
        tgt_signed = -std::min(std::abs(tgt_signed), avail_short);
    }

    if (std::abs(tgt_signed - net_signed) < 1e-9) {
        return std::nullopt; // nothing to do
    }

    OrderPlan plan;
    plan.key = key;
    plan.epoch_before = ps.epoch;
    plan.reservation_id = cell.next_reservation++;

    // If opposite side exists, close that first
    if ((net_signed > 0 && tgt_signed <= 0) || (net_signed < 0 && tgt_signed >= 0)) {
        double close_qty = std::abs(net_signed);
        if (close_qty > 0) {
            plan.legs.push_back({
                /*symbol*/ desire.preferred_symbol.empty() ? symbol : desire.preferred_symbol,
                /*side*/ (ps.side==PositionSide::Long ? PositionSide::Short : PositionSide::Long),
                /*qty*/  close_qty,
                /*reason*/ "CLOSE_OPPOSITE"
            });
            net_signed = 0.0; // after close leg
        }
    }

    // Open / Resize towards target
    double open_delta = tgt_signed - net_signed; // signed
    if (std::abs(open_delta) > 1e-9) {
        plan.legs.push_back({
            /*symbol*/ desire.preferred_symbol.empty() ? symbol : desire.preferred_symbol,
            /*side*/ (open_delta > 0 ? PositionSide::Long : PositionSide::Short),
            /*qty*/  std::abs(open_delta),
            /*reason*/ (net_signed==0.0 ? "OPEN_TARGET" : "RESIZE")
        });
        if (open_delta > 0) cell.reserved_long  += std::abs(open_delta);
        else                cell.reserved_short += std::abs(open_delta);
    }

    return plan;
}

void PositionGuardian::commit(const OrderPlan& plan) {
    auto& cell = cell_for(plan.key);
    std::lock_guard<std::mutex> lk(cell.m);

    // Update snapshot pessimistically to reflect intent (helps prevent collisions)
    for (auto& leg : plan.legs) {
        if (leg.reason == "CLOSE_OPPOSITE") {
            cell.ps.side = PositionSide::Flat;
            cell.ps.qty  = 0.0;
        } else {
            cell.ps.side = (leg.side==PositionSide::Long ? PositionSide::Long : PositionSide::Short);
            cell.ps.qty  += leg.qty; // simple pessimistic add; you can reconcile on fills
        }
        cell.ps.epoch++;
        cell.ps.last_change = Clock::now();
    }
    // release reservations (we pessimistically moved to snapshot)
    cell.reserved_long = cell.reserved_short = 0.0;
}

void PositionGuardian::sync_from_broker(const std::string& account,
                                       const std::vector<Position>& positions,
                                       const std::vector<std::string>& open_orders) {
    // TODO: Implement broker sync logic
    // This would rebuild ps/epoch using live positions and open-orders
    // For now, this is a placeholder
}

} // namespace sentio
