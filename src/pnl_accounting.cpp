// src/pnl_accounting.cpp
#include "sentio/pnl_accounting.hpp"
#include <shared_mutex>
#include <unordered_map>
#include <utility>

namespace sentio {

namespace {
// Simple in-TU storage for latest bars keyed by instrument.
// If your header already declares members, remove this and
// implement against those members instead.
struct PriceBookStorage {
    std::unordered_map<std::string, Bar> latest;
    mutable std::shared_mutex mtx;
};

// One global storage per process for the default PriceBook()
// If your PriceBook is an instance with its own fields, move this inside the class.
static PriceBookStorage& storage() {
    static PriceBookStorage s;
    return s;
}
} // anonymous namespace

// ---- Interface implementation ----

const Bar* PriceBook::get_latest(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    auto it = S.latest.find(instrument);
    if (it == S.latest.end()) return nullptr;
    return &it->second; // pointer remains valid as long as map bucket survives
}

// ---- Additional helper methods (not declared in header but useful) ----

void PriceBook::upsert_latest(const std::string& instrument, const Bar& b) {
    auto& S = storage();
    std::unique_lock lk(S.mtx);
    S.latest[instrument] = b;
}

bool PriceBook::has_instrument(const std::string& instrument) const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.find(instrument) != S.latest.end();
}

std::size_t PriceBook::size() const {
    auto& S = storage();
    std::shared_lock lk(S.mtx);
    return S.latest.size();
}

double last_trade_price(const PriceBook& book, const std::string& instrument) {
    auto* b = book.get_latest(instrument);
    if (!b) throw std::runtime_error("No bar for instrument: " + instrument);
    return b->close;
}

} // namespace sentio
