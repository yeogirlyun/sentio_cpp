#include <gtest/gtest.h>
#include "sentio/signal_or.hpp"
#include <vector>
#include <cmath>

using namespace sentio;

class SignalORTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default configuration for most tests
        cfg_.min_conf = 0.05;
        cfg_.aggression = 0.85;
        cfg_.conflict_soften = 0.35;
        cfg_.neutral_band = 0.015;
        cfg_.min_active = 1;
    }
    
    OrCfg cfg_;
};

TEST_F(SignalORTest, FiresWhenAnyStrongLong) {
    cfg_.aggression = 0.9;
    std::vector<RuleOut> rules = {
        {0.51, 0.2}, {0.52, 0.2}, {0.90, 0.9} // one strong long
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_GT(p, 0.70);
}

TEST_F(SignalORTest, StrongShortWins) {
    std::vector<RuleOut> rules = {
        {0.49, 0.4}, {0.10, 0.8} // strong short
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_LT(p, 0.30);
}

TEST_F(SignalORTest, BalancedConflictNearNeutral) {
    cfg_.conflict_soften = 0.4;
    cfg_.neutral_band = 0.02;
    std::vector<RuleOut> rules = {
        {0.85, 0.9}, {0.15, 0.9} // symmetric conflict
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_NEAR(p, 0.5, 0.05);
}

TEST_F(SignalORTest, SingleRuleDominance) {
    std::vector<RuleOut> rules = {
        {0.95, 0.9} // single very strong long
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_GT(p, 0.8);
}

TEST_F(SignalORTest, MultipleWeakRulesAccumulate) {
    cfg_.aggression = 0.7; // Lower aggression for accumulation
    std::vector<RuleOut> rules = {
        {0.6, 0.3}, {0.65, 0.3}, {0.7, 0.3} // multiple weak long signals
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_GT(p, 0.6);
}

TEST_F(SignalORTest, LowConfidenceRulesIgnored) {
    cfg_.min_conf = 0.5;
    std::vector<RuleOut> rules = {
        {0.9, 0.1}, {0.1, 0.1} // low confidence rules
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_EQ(p, 0.5); // Should return neutral when no rules meet min_conf
}

TEST_F(SignalORTest, NeutralBandDebouncing) {
    cfg_.neutral_band = 0.02;
    std::vector<RuleOut> rules = {
        {0.52, 0.1} // very weak signal near neutral
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_EQ(p, 0.5); // Should be debounced to exact neutral
}

TEST_F(SignalORTest, ConflictSoftening) {
    cfg_.conflict_soften = 0.5;
    std::vector<RuleOut> rules = {
        {0.8, 0.8}, {0.2, 0.8} // strong conflict
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_NEAR(p, 0.5, 0.1); // Should be softened toward neutral
}

TEST_F(SignalORTest, AggressionScaling) {
    // Test high aggression
    cfg_.aggression = 0.95;
    std::vector<RuleOut> rules = {
        {0.6, 0.5}
    };
    double p_high = mix_signal_or(rules, cfg_);
    
    // Test low aggression
    cfg_.aggression = 0.5;
    double p_low = mix_signal_or(rules, cfg_);
    
    EXPECT_GT(p_high, p_low); // Higher aggression should produce stronger signal
}

TEST_F(SignalORTest, EdgeCases) {
    // Empty rules
    std::vector<RuleOut> empty_rules;
    double p_empty = mix_signal_or(empty_rules, cfg_);
    EXPECT_EQ(p_empty, 0.5);
    
    // Invalid probabilities
    std::vector<RuleOut> invalid_rules = {
        {1.5, 0.5}, {-0.5, 0.5}, {0.5, 1.5}
    };
    double p_invalid = mix_signal_or(invalid_rules, cfg_);
    EXPECT_GE(p_invalid, 0.0);
    EXPECT_LE(p_invalid, 1.0);
}

TEST_F(SignalORTest, NumericalStability) {
    // Test with very small probabilities
    std::vector<RuleOut> rules = {
        {0.5001, 0.01}, {0.4999, 0.01}
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_GE(p, 0.0);
    EXPECT_LE(p, 1.0);
    EXPECT_TRUE(std::isfinite(p));
}

TEST_F(SignalORTest, MinActiveRequirement) {
    cfg_.min_active = 2;
    std::vector<RuleOut> rules = {
        {0.8, 0.9} // Only one rule
    };
    double p = mix_signal_or(rules, cfg_);
    EXPECT_EQ(p, 0.5); // Should return neutral when min_active not met
}
