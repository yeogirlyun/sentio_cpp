#pragma once
#include <chrono>
#include <cstdio>

namespace sentio {

struct Tsc {
    std::chrono::high_resolution_clock::time_point t0;
    
    void tic() { 
        t0 = std::chrono::high_resolution_clock::now(); 
    }
    
    double toc_ms() const {
        using namespace std::chrono;
        return duration<double, std::milli>(high_resolution_clock::now() - t0).count();
    }
    
    double toc_sec() const {
        using namespace std::chrono;
        return duration<double>(high_resolution_clock::now() - t0).count();
    }
};

} // namespace sentio