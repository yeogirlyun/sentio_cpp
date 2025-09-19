#include "sentio/rsi_prob.hpp"
#include <iostream>
#include <iomanip>

int main() {
    std::cout << "RSI -> Probability mapping:\n";
    std::cout << "RSI\tProb\tSignal_Strength\tInstrument\n";
    
    for (int rsi = 10; rsi <= 90; rsi += 5) {
        double prob = sentio::rsi_to_prob_tuned(rsi, 1.0);
        double signal_strength = std::abs(prob - 0.5) * 2.0;
        
        std::string instrument = "CASH";
        if (prob > 0.85) instrument = "TQQQ";
        else if (prob > 0.775) instrument = "QQQ";
        else if (prob < 0.15) instrument = "SQQQ";
        else if (prob < 0.225) instrument = "PSQ";
        
        std::cout << std::fixed << std::setprecision(4) 
                  << rsi << "\t" << prob << "\t" << signal_strength 
                  << "\t\t" << instrument << "\n";
    }
    
    std::cout << "\nAllocation thresholds:\n";
    std::cout << "TQQQ: prob > 0.85 or < 0.15 (signal_strength >= 0.70)\n";
    std::cout << "QQQ:  prob > 0.775 or < 0.225 (signal_strength >= 0.55)\n";
    std::cout << "PSQ:  prob < 0.225 (signal_strength >= 0.55)\n";
    std::cout << "SQQQ: prob < 0.15 (signal_strength >= 0.70)\n";
    
    return 0;
}
