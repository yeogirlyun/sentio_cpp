#pragma once
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace sentio {

class FamilyMapper {
public:
    // Provide complete families you trade; keep in config.
    // Key = family id; Values = member symbols (case-insensitive).
    using Map = std::unordered_map<std::string, std::vector<std::string>>;

    explicit FamilyMapper(Map families) : families_(std::move(families)) {
        // build reverse index
        for (auto& [fam, syms] : families_) {
            for (auto s : syms) {
                auto u = upper(s);
                rev_[u] = fam;
            }
        }
    }

    // Return family for a symbol, or the symbol itself if unknown.
    std::string family_for(const std::string& symbol) const {
        auto u = upper(symbol);
        auto it = rev_.find(u);
        return it==rev_.end() ? u : it->second;
    }

private:
    static std::string upper(std::string s) {
        std::transform(s.begin(), s.end(), s.begin(), ::toupper);
        return s;
    }

    Map families_;
    std::unordered_map<std::string,std::string> rev_;
};

} // namespace sentio
