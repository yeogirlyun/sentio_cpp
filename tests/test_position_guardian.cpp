#include <gtest/gtest.h>
#include "sentio/position_guardian.hpp"
#include "sentio/family_mapper.hpp"
#include "sentio/side.hpp"
#include <iostream>

using namespace sentio;

class PositionGuardianTest : public ::testing::Test {
protected:
    void SetUp() override {
        mapper_ = std::make_unique<FamilyMapper>(FamilyMapper::Map{
            {"QQQ*", {"QQQ", "TQQQ", "SQQQ"}},
            {"SPY*", {"SPY", "UPRO", "SPXU"}}
        });
        guardian_ = std::make_unique<PositionGuardian>(*mapper_);
        
        policy_.allow_conflicts = false;
        policy_.max_gross_shares = 1000;
        policy_.cooldown_ms = 100; // 100ms cooldown for testing
    }

    std::unique_ptr<FamilyMapper> mapper_;
    std::unique_ptr<PositionGuardian> guardian_;
    Policy policy_;
};

TEST_F(PositionGuardianTest, FlipLongToShortNoOverlap) {
    const std::string account = "test_account";
    
    // Start with flat position
    auto key = ExposureKey{account, "QQQ*"};
    auto ps = guardian_->snapshot(key);
    EXPECT_EQ(ps.side, PositionSide::Flat);
    EXPECT_EQ(ps.qty, 0.0);
    
    // Ask to go long 100 QQQ
    Desire desire1{PositionSide::Long, 100.0, "QQQ"};
    auto plan1 = guardian_->plan(account, "QQQ", desire1, policy_);
    ASSERT_TRUE(plan1);
    EXPECT_EQ(plan1->legs.size(), 1);
    EXPECT_EQ(plan1->legs[0].reason, "OPEN_TARGET");
    EXPECT_EQ(plan1->legs[0].side, PositionSide::Long);
    EXPECT_EQ(plan1->legs[0].qty, 100.0);
    
    // Commit the plan
    guardian_->commit(*plan1);
    
    // Verify position is now long
    ps = guardian_->snapshot(key);
    EXPECT_EQ(ps.side, PositionSide::Long);
    EXPECT_EQ(ps.qty, 100.0);
    
    // Now request short 150 SQQQ (flip)
    Desire desire2{PositionSide::Short, 150.0, "SQQQ"};
    auto plan2 = guardian_->plan(account, "SQQQ", desire2, policy_);
    ASSERT_TRUE(plan2);
    ASSERT_EQ(plan2->legs.size(), 2);
    EXPECT_EQ(plan2->legs[0].reason, "CLOSE_OPPOSITE"); // close long first
    EXPECT_EQ(plan2->legs[0].side, PositionSide::Short);
    EXPECT_EQ(plan2->legs[0].qty, 100.0);
    EXPECT_EQ(plan2->legs[1].reason, "OPEN_TARGET");    // then open short
    EXPECT_EQ(plan2->legs[1].side, PositionSide::Short);
    EXPECT_EQ(plan2->legs[1].qty, 150.0);
    
    // Commit the flip plan
    guardian_->commit(*plan2);
    
    // Verify position is now short
    ps = guardian_->snapshot(key);
    EXPECT_EQ(ps.side, PositionSide::Short);
    EXPECT_EQ(ps.qty, 150.0);
}

TEST_F(PositionGuardianTest, NoActionWhenTargetMatchesCurrent) {
    const std::string account = "test_account";
    
    // Go long 100 QQQ
    Desire desire1{PositionSide::Long, 100.0, "QQQ"};
    auto plan1 = guardian_->plan(account, "QQQ", desire1, policy_);
    ASSERT_TRUE(plan1);
    guardian_->commit(*plan1);
    
    // Try to go long 100 QQQ again (should be no-op)
    Desire desire2{PositionSide::Long, 100.0, "QQQ"};
    auto plan2 = guardian_->plan(account, "QQQ", desire2, policy_);
    EXPECT_FALSE(plan2); // Should return nullopt for no-op
}

TEST_F(PositionGuardianTest, ResizeWithinSameSide) {
    const std::string account = "test_account";
    
    // Go long 100 QQQ
    Desire desire1{PositionSide::Long, 100.0, "QQQ"};
    auto plan1 = guardian_->plan(account, "QQQ", desire1, policy_);
    ASSERT_TRUE(plan1);
    guardian_->commit(*plan1);
    
    // Resize to 150 QQQ (same side, larger)
    Desire desire2{PositionSide::Long, 150.0, "QQQ"};
    auto plan2 = guardian_->plan(account, "QQQ", desire2, policy_);
    ASSERT_TRUE(plan2);
    EXPECT_EQ(plan2->legs.size(), 1);
    EXPECT_EQ(plan2->legs[0].reason, "RESIZE");
    EXPECT_EQ(plan2->legs[0].side, PositionSide::Long);
    EXPECT_EQ(plan2->legs[0].qty, 50.0); // Only the delta
}

TEST_F(PositionGuardianTest, CooldownPreventsRapidFlipping) {
    const std::string account = "test_account";
    
    // Go long 100 QQQ
    Desire desire1{PositionSide::Long, 100.0, "QQQ"};
    auto plan1 = guardian_->plan(account, "QQQ", desire1, policy_);
    ASSERT_TRUE(plan1);
    guardian_->commit(*plan1);
    
    // Immediately try to flip to short (should be blocked by cooldown)
    Desire desire2{PositionSide::Short, 100.0, "SQQQ"};
    auto plan2 = guardian_->plan(account, "SQQQ", desire2, policy_);
    EXPECT_FALSE(plan2); // Should be blocked by cooldown
}

TEST_F(PositionGuardianTest, MaxGrossLimit) {
    const std::string account = "test_account";
    
    // Try to exceed max gross limit
    Desire desire{PositionSide::Long, 2000.0, "QQQ"}; // Exceeds 1000 limit
    auto plan = guardian_->plan(account, "QQQ", desire, policy_);
    ASSERT_TRUE(plan);
    EXPECT_EQ(plan->legs.size(), 1);
    EXPECT_EQ(plan->legs[0].qty, 1000.0); // Should be capped at max_gross_shares
}

TEST_F(PositionGuardianTest, FamilyMapping) {
    // Test that different symbols in same family map to same key
    const std::string account = "test_account";
    
    // QQQ and TQQQ should map to same family
    EXPECT_EQ(mapper_->family_for("QQQ"), "QQQ*");
    EXPECT_EQ(mapper_->family_for("TQQQ"), "QQQ*");
    EXPECT_EQ(mapper_->family_for("SQQQ"), "QQQ*");
    
    // Unknown symbol should map to itself
    EXPECT_EQ(mapper_->family_for("UNKNOWN"), "UNKNOWN");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
