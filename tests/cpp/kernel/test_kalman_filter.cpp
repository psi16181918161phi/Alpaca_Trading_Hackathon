#include <gtest/gtest.h>
#include "xquantx/kalman_filter.h"

#include <cmath>

// Parity oracle: these expected values were computed by running the same
// input sequence through the Python reference implementation
// (investment_agent.filters.kalman_filter.KalmanFilter) with identical
// constructor defaults, per
// alpaca_paper_trading_specifications_x_quant_x/011_xquantx_integration_tests.txt
// section 1.9.2 (1e-10 tolerance). No file I/O is used, per section 1.8.3.
namespace {
constexpr double kParityTolerance = 1e-10;
}

TEST(KalmanFilterTest, ConstructionSeedsPriceWithZeroTrend) {
    xquantx::KalmanFilter kf(100.0);
    auto state = kf.get_state();
    EXPECT_DOUBLE_EQ(state.estimated_price, 100.0);
    EXPECT_DOUBLE_EQ(state.trend, 0.0);
    EXPECT_DOUBLE_EQ(state.price_variance, 1.0);
    EXPECT_DOUBLE_EQ(state.trend_variance, 1.0);
}

TEST(KalmanFilterTest, SingleUpdateMatchesPythonReferenceToTolerance) {
    // Values below are the ACTUAL output of the Python reference
    // implementation for this exact call sequence (not hand-derived):
    //   kf = investment_agent.filters.kalman_filter.KalmanFilter(100.0)
    //   state = kf.update(101.0)
    // Captured 2026-09-02: estimated_price=100.9950251231282,
    // trend=0.4974876871797423, innovation=1.0,
    // kalman_gain_price=0.9950251231282027,
    // price_variance=0.009950251231282025, trend_variance=0.5025133128202578.
    xquantx::KalmanFilter kf(100.0);
    auto state = kf.update(101.0);

    EXPECT_NEAR(state.estimated_price, 100.9950251231282, kParityTolerance);
    EXPECT_NEAR(state.trend, 0.4974876871797423, kParityTolerance);
    EXPECT_NEAR(state.innovation, 1.0, kParityTolerance);
    EXPECT_NEAR(state.kalman_gain_price, 0.9950251231282027, kParityTolerance);
    EXPECT_NEAR(state.price_variance, 0.009950251231282025, kParityTolerance);
    EXPECT_NEAR(state.trend_variance, 0.5025133128202578, kParityTolerance);
}

TEST(KalmanFilterTest, RepeatedUpdatesConvergeTowardObservedPrice) {
    xquantx::KalmanFilter kf(100.0);
    xquantx::KalmanState state{};
    for (int i = 0; i < 50; ++i) {
        state = kf.update(110.0);
    }
    EXPECT_NEAR(state.estimated_price, 110.0, 1e-2);
}

TEST(KalmanFilterTest, RejectsNonPositiveInitialPrice) {
    EXPECT_THROW(xquantx::KalmanFilter(0.0), std::invalid_argument);
    EXPECT_THROW(xquantx::KalmanFilter(-5.0), std::invalid_argument);
}

TEST(KalmanFilterTest, RejectsNonFiniteObservedPrice) {
    xquantx::KalmanFilter kf(100.0);
    EXPECT_THROW(kf.update(std::nan("")), std::invalid_argument);
    EXPECT_THROW(kf.update(std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST(KalmanFilterTest, ResetRestoresInitialVariancesAndZeroTrend) {
    xquantx::KalmanFilter kf(100.0);
    kf.update(105.0);
    kf.update(103.0);
    kf.reset(200.0);
    auto state = kf.get_state();
    EXPECT_DOUBLE_EQ(state.estimated_price, 200.0);
    EXPECT_DOUBLE_EQ(state.trend, 0.0);
    EXPECT_DOUBLE_EQ(state.price_variance, 1.0);
    EXPECT_DOUBLE_EQ(state.trend_variance, 1.0);
}
