#pragma once
#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>

namespace sentio {

class ProgressBar {
public:
    ProgressBar(int total, const std::string& description = "Progress")
        : total_(total), current_(0), description_(description), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update(int current) {
        current_ = current;
        if (current_ % std::max(1, total_ / 100) == 0 || current_ == total_) {
            display();
        }
    }
    
    void display() {
        double percentage = (double)current_ / total_ * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current_ > 0 && current_ < total_) {
            eta_seconds = (double)elapsed / current_ * (total_ - current_);
        }
        
        std::cout << "\r" << description_ << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current_ << "/" << total_ << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        std::cout << std::flush;
        
        if (current_ == total_) {
            std::cout << std::endl;
        }
    }
    
    void set_description(const std::string& desc) {
        description_ = desc;
    }
    
    int get_current() const { return current_; }
    int get_total() const { return total_; }
    const std::string& get_description() const { return description_; }

private:
    int total_;
    int current_;
    std::string description_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class TrainingProgressBar : public ProgressBar {
public:
    TrainingProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "Training " + strategy_name), best_sharpe_(-999.0), best_return_(-999.0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_metrics(int current, double sharpe_ratio, double total_return, double oos_return = 0.0) {
        // Track best metrics
        if (sharpe_ratio > best_sharpe_) best_sharpe_ = sharpe_ratio;
        if (total_return > best_return_) best_return_ = total_return;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_metrics(sharpe_ratio, total_return, oos_return);
        }
    }
    
    void display_with_metrics(double current_sharpe, double current_return, double oos_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add metrics
        std::cout << " | Sharpe: " << std::fixed << std::setprecision(3) << current_sharpe
                  << " | Return: " << std::fixed << std::setprecision(2) << (current_return * 100) << "%";
        
        if (oos_return != 0.0) {
            std::cout << " | OOS: " << std::fixed << std::setprecision(2) << (oos_return * 100) << "%";
        }
        
        std::cout << " | Best Sharpe: " << std::fixed << std::setprecision(3) << best_sharpe_
                  << " | Best Return: " << std::fixed << std::setprecision(2) << (best_return_ * 100) << "%";
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_sharpe_;
    double best_return_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

class WFProgressBar : public ProgressBar {
public:
    WFProgressBar(int total, const std::string& strategy_name = "TSB")
        : ProgressBar(total, "WF Test " + strategy_name), best_oos_sharpe_(-999.0), best_oos_return_(-999.0), 
          avg_oos_return_(0.0), successful_folds_(0), start_time_(std::chrono::high_resolution_clock::now()) {}
    
    void update_with_wf_metrics(int current, double oos_sharpe, double oos_return, double train_return = 0.0) {
        // Track best OOS metrics
        if (oos_sharpe > best_oos_sharpe_) best_oos_sharpe_ = oos_sharpe;
        if (oos_return > best_oos_return_) best_oos_return_ = oos_return;
        
        // Update running average
        successful_folds_++;
        avg_oos_return_ = (avg_oos_return_ * (successful_folds_ - 1) + oos_return) / successful_folds_;
        
        update(current);
        if (current % std::max(1, get_total() / 100) == 0 || current == get_total()) {
            display_with_wf_metrics(oos_sharpe, oos_return, train_return);
        }
    }
    
    void display_with_wf_metrics(double current_oos_sharpe, double current_oos_return, double train_return = 0.0) {
        int current = get_current();
        int total = get_total();
        double percentage = (double)current / total * 100.0;
        auto now = std::chrono::high_resolution_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - start_time_).count();
        
        // Calculate ETA
        double eta_seconds = 0.0;
        if (current > 0 && current < total) {
            eta_seconds = (double)elapsed / current * (total - current);
        }
        
        std::cout << "\r" << get_description() << ": [";
        
        // Draw progress bar (50 characters wide)
        int bar_width = 50;
        int pos = (int)(percentage / 100.0 * bar_width);
        for (int i = 0; i < bar_width; ++i) {
            if (i < pos) std::cout << "=";
            else if (i == pos) std::cout << ">";
            else std::cout << " ";
        }
        
        std::cout << "] " << std::fixed << std::setprecision(1) << percentage << "% "
                  << "(" << current << "/" << total << ") "
                  << "Elapsed: " << elapsed << "s";
        
        if (eta_seconds > 0) {
            std::cout << " ETA: " << (int)eta_seconds << "s";
        }
        
        // Add WF-specific metrics
        std::cout << " | OOS Sharpe: " << std::fixed << std::setprecision(3) << current_oos_sharpe
                  << " | OOS Return: " << std::fixed << std::setprecision(2) << (current_oos_return * 100) << "%";
        
        if (train_return != 0.0) {
            std::cout << " | Train Return: " << std::fixed << std::setprecision(2) << (train_return * 100) << "%";
        }
        
        std::cout << " | Avg OOS: " << std::fixed << std::setprecision(2) << (avg_oos_return_ * 100) << "%"
                  << " | Best OOS Sharpe: " << std::fixed << std::setprecision(3) << best_oos_sharpe_
                  << " | Folds: " << successful_folds_;
        
        std::cout << std::flush;
        
        if (current == total) {
            std::cout << std::endl;
        }
    }

private:
    double best_oos_sharpe_;
    double best_oos_return_;
    double avg_oos_return_;
    int successful_folds_;
    std::chrono::high_resolution_clock::time_point start_time_;
};

} // namespace sentio