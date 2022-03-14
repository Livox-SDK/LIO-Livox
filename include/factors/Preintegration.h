#ifndef INTEGRATIONBASE_H_
#define INTEGRATIONBASE_H_

#include <Eigen/Eigen>

#include "utils/math_tools.h"
#include "utils/common.h"

using Eigen::Matrix;
using Eigen::Matrix3d;
using Eigen::MatrixXd;
using Eigen::Quaterniond;
using Eigen::Vector3d;

enum StateOrder
{
    O_P = 0,
    O_R = 3,
    O_V = 6,
    O_BA = 9,
    O_BG = 12,
};

class Preintegration
{

public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Preintegration() = delete;

    Preintegration(const Vector3d &acc0, const Vector3d &gyr0,
                   const Vector3d &linearized_ba, const Vector3d &linearized_bg)
        : acc0_{acc0},
          gyr0_{gyr0},
          linearized_acc_{acc0},
          linearized_gyr_{gyr0},
          linearized_ba_{linearized_ba},
          linearized_bg_{linearized_bg},
          jacobian_{Matrix<double, 15, 15>::Identity()},
          sum_dt_{0.0},
          delta_p_{Vector3d::Zero()},
          delta_q_{Quaterniond::Identity()},
          delta_v_{Vector3d::Zero()}
    {

        //      acc_n = 0.00059;
        //      gyr_n = 0.000061;
        //      acc_w = 0.000011;
        //      gyr_w = 0.000001;

        nh.param<double>("/IMU/acc_n", acc_n, 0.00059);
        nh.param<double>("/IMU/gyr_n", gyr_n, 0.000061);
        nh.param<double>("/IMU/acc_w", acc_w, 0.000011);
        nh.param<double>("/IMU/gyr_w", gyr_w, 0.000001);

        covariance_ = 0.001 * Matrix<double, 15, 15>::Identity();
        g_vec_ = -Eigen::Vector3d(0, 0, 9.805);

        noise_ = Matrix<double, 18, 18>::Zero();
        noise_.block<3, 3>(0, 0) = (acc_n * acc_n) * Matrix3d::Identity();
        noise_.block<3, 3>(3, 3) = (gyr_n * gyr_n) * Matrix3d::Identity();
        noise_.block<3, 3>(6, 6) = (acc_n * acc_n) * Matrix3d::Identity();
        noise_.block<3, 3>(9, 9) = (gyr_n * gyr_n) * Matrix3d::Identity();
        noise_.block<3, 3>(12, 12) = (acc_w * acc_w) * Matrix3d::Identity();
        noise_.block<3, 3>(15, 15) = (gyr_w * gyr_w) * Matrix3d::Identity();
    }

    void push_back(double dt, const Vector3d &acc, const Vector3d &gyr)
    {
        dt_buf_.push_back(dt);
        acc_buf_.push_back(acc);
        gyr_buf_.push_back(gyr);
        Propagate(dt, acc, gyr);
    }

    void Repropagate(const Vector3d &linearized_ba, const Vector3d &linearized_bg)
    {
        sum_dt_ = 0.0;
        acc0_ = linearized_acc_;
        gyr0_ = linearized_gyr_;
        delta_p_.setZero();
        delta_q_.setIdentity();
        delta_v_.setZero();
        linearized_ba_ = linearized_ba;
        linearized_bg_ = linearized_bg;
        jacobian_.setIdentity();
        covariance_.setZero();
        for (size_t i = 0; i < dt_buf_.size(); ++i)
        {
            Propagate(dt_buf_[i], acc_buf_[i], gyr_buf_[i]);
        }
    }

    void MidPointIntegration(double dt,
                             const Vector3d &acc0, const Vector3d &gyr0,
                             const Vector3d &acc1, const Vector3d &gyr1,
                             const Vector3d &delta_p, const Quaterniond &delta_q,
                             const Vector3d &delta_v, const Vector3d &linearized_ba,
                             const Vector3d &linearized_bg, Vector3d &result_delta_p,
                             Quaterniond &result_delta_q, Vector3d &result_delta_v,
                             Vector3d &result_linearized_ba, Vector3d &result_linearized_bg,
                             bool update_jacobian)
    {

        Vector3d un_acc_0 = delta_q * (acc0 - linearized_ba);
        Vector3d un_gyr = 0.5 * (gyr0 + gyr1) - linearized_bg;
        result_delta_q = delta_q * Quaterniond(1, un_gyr(0) * dt / 2, un_gyr(1) * dt / 2, un_gyr(2) * dt / 2);
        Vector3d un_acc_1 = result_delta_q * (acc1 - linearized_ba);
        Vector3d un_acc = 0.5 * (un_acc_0 + un_acc_1);
        result_delta_p = delta_p + delta_v * dt + 0.5 * un_acc * dt * dt;
        result_delta_v = delta_v + un_acc * dt;
        result_linearized_ba = linearized_ba;
        result_linearized_bg = linearized_bg;

        if (update_jacobian)
        {
            Vector3d w_x = 0.5 * (gyr0 + gyr1) - linearized_bg;
            Vector3d a_0_x = acc0 - linearized_ba;
            Vector3d a_1_x = acc1 - linearized_ba;
            Matrix3d R_w_x, R_a_0_x, R_a_1_x;

            R_w_x << 0, -w_x(2), w_x(1),
                w_x(2), 0, -w_x(0),
                -w_x(1), w_x(0), 0;
            R_a_0_x << 0, -a_0_x(2), a_0_x(1),
                a_0_x(2), 0, -a_0_x(0),
                -a_0_x(1), a_0_x(0), 0;
            R_a_1_x << 0, -a_1_x(2), a_1_x(1),
                a_1_x(2), 0, -a_1_x(0),
                -a_1_x(1), a_1_x(0), 0;

            MatrixXd F = MatrixXd::Zero(15, 15);
            F.block<3, 3>(0, 0) = Matrix3d::Identity();
            F.block<3, 3>(0, 3) = -0.25 * delta_q.toRotationMatrix() * R_a_0_x * dt * dt +
                                  -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt) * dt * dt;
            F.block<3, 3>(0, 6) = MatrixXd::Identity(3, 3) * dt;
            F.block<3, 3>(0, 9) = -0.25 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt * dt;
            F.block<3, 3>(0, 12) = -0.1667 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * -dt;
            F.block<3, 3>(3, 3) = Matrix3d::Identity() - R_w_x * dt;
            F.block<3, 3>(3, 12) = -MatrixXd::Identity(3, 3) * dt;
            F.block<3, 3>(6, 3) = -0.5 * delta_q.toRotationMatrix() * R_a_0_x * dt +
                                  -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * (Matrix3d::Identity() - R_w_x * dt) * dt;
            F.block<3, 3>(6, 6) = Matrix3d::Identity();
            F.block<3, 3>(6, 9) = -0.5 * (delta_q.toRotationMatrix() + result_delta_q.toRotationMatrix()) * dt;
            F.block<3, 3>(6, 12) = -0.5 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * -dt;
            F.block<3, 3>(9, 9) = Matrix3d::Identity();
            F.block<3, 3>(12, 12) = Matrix3d::Identity();

            // NOTE: V = Fd * G_c
            // FIXME: verify if it is right, the 0.25 part
            MatrixXd V = MatrixXd::Zero(15, 18);
            V.block<3, 3>(0, 0) = 0.5 * delta_q.toRotationMatrix() * dt * dt;
            V.block<3, 3>(0, 3) = -0.25 * result_delta_q.toRotationMatrix() * R_a_1_x * dt * dt * 0.5 * dt;
            V.block<3, 3>(0, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt * dt;
            V.block<3, 3>(0, 9) = V.block<3, 3>(0, 3);
            V.block<3, 3>(3, 3) = 0.5 * MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(3, 9) = 0.5 * MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(6, 0) = 0.5 * delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 3) = 0.5 * -result_delta_q.toRotationMatrix() * R_a_1_x * dt * 0.5 * dt;
            V.block<3, 3>(6, 6) = 0.5 * result_delta_q.toRotationMatrix() * dt;
            V.block<3, 3>(6, 9) = V.block<3, 3>(6, 3);
            V.block<3, 3>(9, 12) = MatrixXd::Identity(3, 3) * dt;
            V.block<3, 3>(12, 15) = MatrixXd::Identity(3, 3) * dt;

            jacobian_ = F * jacobian_;
            covariance_ = F * covariance_ * F.transpose() + V * noise_ * V.transpose();
        }
    }

    void Propagate(double dt, const Vector3d &acc1, const Vector3d &gyr1)
    {
        dt_ = dt;
        acc1_ = acc1;
        gyr1_ = gyr1;
        Vector3d result_delta_p;
        Quaterniond result_delta_q;
        Vector3d result_delta_v;
        Vector3d result_linearized_ba;
        Vector3d result_linearized_bg;

        MidPointIntegration(dt, acc0_, gyr0_, acc1, gyr1, delta_p_, delta_q_, delta_v_,
                            linearized_ba_, linearized_bg_,
                            result_delta_p, result_delta_q, result_delta_v,
                            result_linearized_ba, result_linearized_bg, true);

        delta_p_ = result_delta_p;
        delta_q_ = result_delta_q;
        delta_v_ = result_delta_v;
        linearized_ba_ = result_linearized_ba;
        linearized_bg_ = result_linearized_bg;
        delta_q_.normalize();
        sum_dt_ += dt_;
        acc0_ = acc1_;
        gyr0_ = gyr1_;
    }

    Matrix<double, 15, 1> evaluate(const Vector3d &Pi,
                                   const Quaterniond &Qi,
                                   const Vector3d &Vi,
                                   const Vector3d &Bai,
                                   const Vector3d &Bgi,
                                   const Vector3d &Pj,
                                   const Quaterniond &Qj,
                                   const Vector3d &Vj,
                                   const Vector3d &Baj,
                                   const Vector3d &Bgj)
    {
        // NOTE: low cost update jacobian here

        Matrix<double, 15, 1> residuals;

        residuals.setZero();

        Matrix3d dp_dba = jacobian_.block<3, 3>(O_P, O_BA);
        Matrix3d dp_dbg = jacobian_.block<3, 3>(O_P, O_BG);

        Matrix3d dq_dbg = jacobian_.block<3, 3>(O_R, O_BG);

        Matrix3d dv_dba = jacobian_.block<3, 3>(O_V, O_BA);
        Matrix3d dv_dbg = jacobian_.block<3, 3>(O_V, O_BG);

        Vector3d dba = Bai - linearized_ba_;
        Vector3d dbg = Bgi - linearized_bg_; // NOTE: optimized one minus the linearized one

        Quaterniond corrected_delta_q = delta_q_ * deltaQ(dq_dbg * dbg);
        Vector3d corrected_delta_v = delta_v_ + dv_dba * dba + dv_dbg * dbg;
        Vector3d corrected_delta_p = delta_p_ + dp_dba * dba + dp_dbg * dbg;

        residuals.block<3, 1>(O_P, 0) =
            Qi.inverse() * (-0.5 * g_vec_ * sum_dt_ * sum_dt_ + Pj - Pi - Vi * sum_dt_) - corrected_delta_p;
        residuals.block<3, 1>(O_R, 0) = 2.0 * (corrected_delta_q.inverse() * (Qi.inverse() * Qj)).normalized().vec();
        residuals.block<3, 1>(O_V, 0) = Qi.inverse() * (-g_vec_ * sum_dt_ + Vj - Vi) - corrected_delta_v;
        residuals.block<3, 1>(O_BA, 0) = Baj - Bai;
        residuals.block<3, 1>(O_BG, 0) = Bgj - Bgi;

        return residuals;
    }

    double dt_;
    Vector3d acc0_, gyr0_;
    Vector3d acc1_, gyr1_;

    const Vector3d linearized_acc_, linearized_gyr_;
    Vector3d linearized_ba_, linearized_bg_;

    Matrix<double, 15, 15> jacobian_, covariance_;
    Matrix<double, 18, 18> noise_;

    double sum_dt_;
    Vector3d delta_p_;
    Quaterniond delta_q_;
    Vector3d delta_v_;

    std::vector<double> dt_buf_;
    std::vector<Vector3d> acc_buf_;
    std::vector<Vector3d> gyr_buf_;

    Eigen::Vector3d g_vec_;
    double nf, cf;
    double acc_n;
    double gyr_n;
    double acc_w;
    double gyr_w;
    ros::NodeHandle nh;
};

#endif // INTEGRATIONBASE_H_
