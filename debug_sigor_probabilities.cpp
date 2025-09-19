#include "sentio/strategy_signal_or.hpp"
#include "sentio/data_loader.hpp"
#include <iostream>
#include <iomanip>

int main() {
    // Load some data
    DataLoader loader;
    auto bars = loader.load_csv("data/equities/QQQ_RTH_NH.csv", 1000);
    
    // Create strategy
    SignalOrStrategy strategy;
    
    std::cout << "Analyzing sigor probability generation:\n";
    std::cout << "Bar\tProbability\tSignal_Strength\tInstrument\n";
    
    for (int i = 250; i < std::min(350, (int)bars.size()); ++i) {
        double prob = strategy.calculate_probability(bars, i);
        double signal_strength = std::abs(prob - 0.5) * 2.0;
        
        std::string instrument = "CASH";
        if (prob > 0.85) instrument = "TQQQ";
        else if (prob > 0.775) instrument = "QQQ";
        else if (prob < 0.15) instrument = "SQQQ";
        else if (prob < 0.225) instrument = "PSQ";
        
        std::cout << std::fixed << std::setprecision(4) 
                  << i << "\t" << prob << "\t\t" << signal_strength 
                  << "\t\t" << instrument << "\n";
    }
    
    return 0;
}
