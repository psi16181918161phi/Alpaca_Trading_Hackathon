#include "xquantx/kalman_filter.h"

#include <cmath>
#include <stdexcept>

namespace xquantx {

namespace {

double validate_positive_finite(double value, const char* label) {
    if (std::isnan(value)) {
        throw std::invalid_argument(std::string(label) + " must not be NaN");
    }
    if (std::isinf(value)) {
        throw std::invalid_argument(std::string(label) + " must be finite");
    }
    if (value <= 0.0) {
        throw std::invalid_argument(std::string(label) + " must be strictly positive");
    }
    return value;
}

}  // namespace

KalmanFilter::KalmanFilter(
    double initial_price,
    double dt,
    double process_noise_price,
    double process_noise_trend,
    double measurement_noise,
    double initial_price_variance,
    double initial_trend_variance)
    : dt_(validate_positive_finite(dt, "dt")),
      initial_price_variance_(validate_positive_finite(initial_price_variance, "initial_price_variance")),
      initial_trend_variance_(validate_positive_finite(initial_trend_variance, "initial_trend_variance")),
      R_(validate_positive_finite(measurement_noise, "measurement_noise")),
      innovation_(0.0),
      kalman_gain_(Eigen::Vector2d::Zero()),
      step_(0) {
    validate_positive_finite(initial_price, "initial_price");
    validate_positive_finite(process_noise_price, "process_noise_price");
    validate_positive_finite(process_noise_trend, "process_noise_trend");

    x_ << initial_price, 0.0;

    P_ = Eigen::Matrix2d::Zero();
    P_(0, 0) = initial_price_variance_;
    P_(1, 1) = initial_trend_variance_;

    F_ << 1.0, dt_,
          0.0, 1.0;

    H_ << 1.0, 0.0;

    Q_ = Eigen::Matrix2d::Zero();
    Q_(0, 0) = process_noise_price;
    Q_(1, 1) = process_noise_trend;

    I_ = Eigen::Matrix2d::Identity();
}

KalmanState KalmanFilter::update(double observed_price) {
    validate_positive_finite(observed_price, "observed_price");
    const double z = observed_price;

    // Local candidate prediction step.
    const Eigen::Vector2d x_pred = F_ * x_;
    const Eigen::Matrix2d P_pred = F_ * P_ * F_.transpose() + Q_;

    // Local candidate measurement update (correction step).
    const double y_innov = z - (H_ * x_pred)(0);
    const double S = (H_ * P_pred * H_.transpose())(0) + R_;

    if (std::isnan(S) || std::isinf(S) || S <= 0.0) {
        throw std::runtime_error("Kalman filter innovation covariance S is non-positive or non-finite");
    }

    const double S_inv = 1.0 / S;
    const Eigen::Vector2d K = P_pred * H_.transpose() * S_inv;
    const Eigen::Vector2d x_upd = x_pred + K * y_innov;

    // Joseph-form covariance update with enforced symmetry.
    const Eigen::Matrix2d I_KH = I_ - K * H_;
    Eigen::Matrix2d P_upd = I_KH * P_pred * I_KH.transpose() + K * R_ * K.transpose();
    P_upd = 0.5 * (P_upd + P_upd.transpose());

    if (!x_upd.allFinite()) {
        throw std::runtime_error("Kalman filter state vector estimate contains non-finite values");
    }
    if (!P_upd.allFinite()) {
        throw std::runtime_error("Kalman filter covariance matrix contains non-finite values");
    }
    if (!K.allFinite()) {
        throw std::runtime_error("Kalman filter gain matrix contains non-finite values");
    }
    if (P_upd(0, 0) < -1e-12 || P_upd(1, 1) < -1e-12) {
        throw std::runtime_error("Kalman filter covariance diagonal became negative");
    }
    if (P_upd(0, 0) < 0.0) {
        P_upd(0, 0) = 0.0;
    }
    if (P_upd(1, 1) < 0.0) {
        P_upd(1, 1) = 0.0;
    }

    // Atomic commit.
    x_ = x_upd;
    P_ = P_upd;
    innovation_ = y_innov;
    kalman_gain_ = K;
    step_ += 1;

    return build_state();
}

KalmanState KalmanFilter::get_state() const {
    return build_state();
}

void KalmanFilter::reset(double initial_price) {
    validate_positive_finite(initial_price, "initial_price");
    x_ << initial_price, 0.0;
    P_ = Eigen::Matrix2d::Zero();
    P_(0, 0) = initial_price_variance_;
    P_(1, 1) = initial_trend_variance_;
    innovation_ = 0.0;
    kalman_gain_ = Eigen::Vector2d::Zero();
    step_ = 0;
}

KalmanState KalmanFilter::build_state() const {
    KalmanState state{};
    state.estimated_price = x_(0);
    state.trend = x_(1);
    state.price_variance = P_(0, 0);
    state.trend_variance = P_(1, 1);
    state.uncertainty = std::sqrt(P_(0, 0));
    state.trend_uncertainty = std::sqrt(P_(1, 1));
    state.innovation = innovation_;
    state.kalman_gain_price = kalman_gain_(0);
    return state;
}

}  // namespace xquantx
