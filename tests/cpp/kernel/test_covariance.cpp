#include <gtest/gtest.h>
#include "xquantx/kalman_filter.h"

#include <cmath>

// Covariance-focused tests: the Joseph-form update's structural
// guarantees (symmetry, non-negative diagonal, monotonic shrinkage
// under repeated consistent observations) rather than the state
// estimate itself (covered by test_kalman_filter.cpp).

TEST(CovarianceTest, InitialCovarianceIsDiagonalWithConfiguredVariances) {
    xquantx::KalmanFilter kf(100.0, /*dt=*/1.0, /*process_noise_price=*/1e-4,
                              /*process_noise_trend=*/1e-6, /*measurement_noise=*/1e-2,
                              /*initial_price_variance=*/2.5, /*initial_trend_variance=*/0.5);
    auto state = kf.get_state();
    EXPECT_DOUBLE_EQ(state.price_variance, 2.5);
    EXPECT_DOUBLE_EQ(state.trend_variance, 0.5);
    EXPECT_DOUBLE_EQ(state.uncertainty, std::sqrt(2.5));
    EXPECT_DOUBLE_EQ(state.trend_uncertainty, std::sqrt(0.5));
}

TEST(CovarianceTest, VarianceStaysNonNegativeAcrossManyUpdates) {
    xquantx::KalmanFilter kf(100.0);
    for (int i = 0; i < 200; ++i) {
        auto state = kf.update(100.0 + 0.01 * (i % 7));
        EXPECT_GE(state.price_variance, 0.0);
        EXPECT_GE(state.trend_variance, 0.0);
    }
}

TEST(CovarianceTest, PriceVarianceShrinksTowardSteadyStateUnderConsistentObservations) {
    xquantx::KalmanFilter kf(100.0);
    auto first = kf.update(100.0);
    xquantx::KalmanState last{};
    for (int i = 0; i < 100; ++i) {
        last = kf.update(100.0);
    }
    // Repeated consistent observations should not increase uncertainty.
    EXPECT_LE(last.price_variance, first.price_variance + 1e-12);
}

TEST(CovarianceTest, KalmanGainStaysBoundedInUnitInterval) {
    // For this filter's parameter regime the price-component gain should
    // stay in (0, 1]: it can never overweight the observation beyond 1x
    // nor go negative.
    xquantx::KalmanFilter kf(100.0);
    for (int i = 0; i < 50; ++i) {
        auto state = kf.update(100.0 + i * 0.1);
        EXPECT_GT(state.kalman_gain_price, 0.0);
        EXPECT_LE(state.kalman_gain_price, 1.0);
    }
}

TEST(CovarianceTest, RejectsNonPositiveNoiseParameters) {
    EXPECT_THROW(xquantx::KalmanFilter(100.0, 1.0, /*process_noise_price=*/0.0), std::invalid_argument);
    EXPECT_THROW(xquantx::KalmanFilter(100.0, 1.0, 1e-4, /*process_noise_trend=*/-1.0), std::invalid_argument);
    EXPECT_THROW(xquantx::KalmanFilter(100.0, /*dt=*/0.0), std::invalid_argument);
}
