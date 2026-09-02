#pragma once

#include <Eigen/Dense>

namespace xquantx {

/// Immutable snapshot of the Kalman filter's posterior state.
///
/// Mirrors ``investment_agent.filters.kalman_filter.KalmanState`` in the
/// Python reference implementation field-for-field, so a C++ and Python
/// KalmanFilter fed the same observation sequence must agree on every
/// field to within the parity tolerance asserted by the test suite.
struct KalmanState {
    double estimated_price;
    double trend;
    double uncertainty;
    double trend_uncertainty;
    double price_variance;
    double trend_variance;
    double innovation;
    double kalman_gain_price;
};

/// Linear Kalman filter for financial price/trend state estimation.
///
/// Ports ``investment_agent.filters.kalman_filter.KalmanFilter`` (2D
/// state vector [price, trend], Joseph-form covariance update) to C++.
/// See ``high_level_proofs/high_level_kalman_filter_states_capital_allocation_proof.tex``
/// for the mathematical specification this class implements.
///
/// This class performs no networking, file I/O, or Alpaca API calls; it
/// is a pure numerical kernel, per
/// ``alpaca_paper_trading_specifications_x_quant_x/006_xquantx_scaffolding.txt``
/// section 1.4.2 ("C++ does not call Alpaca API").
class KalmanFilter {
public:
    /// Construct a filter seeded at ``initial_price`` with zero trend.
    ///
    /// @throws std::invalid_argument if any parameter is NaN, infinite,
    ///         or non-positive (initial_price must be > 0; dt, the two
    ///         process-noise variances, the measurement-noise variance,
    ///         and the two initial variances must all be > 0).
    explicit KalmanFilter(
        double initial_price,
        double dt = 1.0,
        double process_noise_price = 1e-4,
        double process_noise_trend = 1e-6,
        double measurement_noise = 1e-2,
        double initial_price_variance = 1.0,
        double initial_trend_variance = 1.0);

    /// Incorporate a new price observation and return the updated state.
    ///
    /// Executes predict -> correct with local candidate variables and a
    /// Joseph-form covariance update, matching the Python reference's
    /// atomic-commit contract: on failure, the instance's state is left
    /// unchanged.
    ///
    /// @throws std::invalid_argument if ``observed_price`` is NaN,
    ///         infinite, or non-positive.
    /// @throws std::runtime_error if the candidate covariance/gain
    ///         becomes non-finite or numerically degenerate.
    KalmanState update(double observed_price);

    /// Return the current posterior state without performing an update.
    KalmanState get_state() const;

    /// Re-initialise the filter with a new starting price (trend reset
    /// to 0, covariance reset to the originally configured variances).
    ///
    /// @throws std::invalid_argument if ``initial_price`` is invalid.
    void reset(double initial_price);

private:
    KalmanState build_state() const;

    double dt_;
    double initial_price_variance_;
    double initial_trend_variance_;

    Eigen::Vector2d x_;      // state: [price, trend]
    Eigen::Matrix2d P_;      // posterior covariance
    Eigen::Matrix2d F_;      // transition matrix
    Eigen::RowVector2d H_;   // observation matrix
    Eigen::Matrix2d Q_;      // process noise covariance
    double R_;               // measurement noise variance (scalar)
    Eigen::Matrix2d I_;      // 2x2 identity

    double innovation_;
    Eigen::Vector2d kalman_gain_;
    int step_;
};

}  // namespace xquantx
