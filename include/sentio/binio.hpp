#pragma once
#include <cstdio>
#include <vector>
#include <string>
#include "core.hpp"

namespace sentio {

inline void save_bin(const std::string& path, const std::vector<Bar>& v) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return;
    
    uint64_t n = v.size();
    std::fwrite(&n, sizeof(n), 1, fp);
    
    for (const auto& bar : v) {
        // Write string length and data
        uint32_t str_len = bar.ts_utc.length();
        std::fwrite(&str_len, sizeof(str_len), 1, fp);
        std::fwrite(bar.ts_utc.c_str(), 1, str_len, fp);
        
        // Write other fields
        std::fwrite(&bar.ts_utc_epoch, sizeof(bar.ts_utc_epoch), 1, fp);
        std::fwrite(&bar.open, sizeof(bar.open), 1, fp);
        std::fwrite(&bar.high, sizeof(bar.high), 1, fp);
        std::fwrite(&bar.low, sizeof(bar.low), 1, fp);
        std::fwrite(&bar.close, sizeof(bar.close), 1, fp);
        std::fwrite(&bar.volume, sizeof(bar.volume), 1, fp);
    }
    std::fclose(fp);
}

inline std::vector<Bar> load_bin(const std::string& path) {
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return {};
    
    uint64_t n = 0; 
    std::fread(&n, sizeof(n), 1, fp);
    std::vector<Bar> v;
    v.reserve(n);
    
    for (uint64_t i = 0; i < n; ++i) {
        Bar bar;
        
        // Read string length and data
        uint32_t str_len = 0;
        std::fread(&str_len, sizeof(str_len), 1, fp);
        if (str_len > 0) {
            std::vector<char> str_data(str_len);
            std::fread(str_data.data(), 1, str_len, fp);
            bar.ts_utc = std::string(str_data.data(), str_len);
        }
        
        // Read other fields
        std::fread(&bar.ts_utc_epoch, sizeof(bar.ts_utc_epoch), 1, fp);
        std::fread(&bar.open, sizeof(bar.open), 1, fp);
        std::fread(&bar.high, sizeof(bar.high), 1, fp);
        std::fread(&bar.low, sizeof(bar.low), 1, fp);
        std::fread(&bar.close, sizeof(bar.close), 1, fp);
        std::fread(&bar.volume, sizeof(bar.volume), 1, fp);
        
        v.push_back(bar);
    }
    std::fclose(fp);
    return v;
}

} // namespace sentio