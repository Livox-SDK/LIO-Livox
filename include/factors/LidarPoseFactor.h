#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "utils/math_tools.h"

#include <ceres/ceres.h>


struct LidarPoseFactorAutoDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarPoseFactorAutoDiff(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_): delta_q(delta_q_), delta_p(delta_p_)
    {}

    template <typename T>
    bool operator()(const T *p1, const T *p2, const T *p3, const T *p4, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> P1(p1[0], p1[1], p1[2]);
        Eigen::Quaternion<T> Q1(p2[0], p2[1], p2[2], p2[3]);

        Eigen::Matrix<T, 3, 1> P2(p3[0], p3[1], p3[2]);
        Eigen::Quaternion<T> Q2(p4[0], p4[1], p4[2], p4[3]);

        Eigen::Quaternion<T> tmp_delta_q(T(delta_q.w()), T(delta_q.x()), T(delta_q.y()), T(delta_q.z()));
        Eigen::Matrix<T, 3, 1> tmp_delta_p(T(delta_p.x()), T(delta_p.y()), T(delta_p.z()));

        Eigen::Matrix<T, 3, 1> residual1 = T(2.0) * (tmp_delta_q.inverse() * Q1.inverse() * Q2).vec();
        Eigen::Matrix<T, 3, 1> residual2 = Q1.inverse() * (P2 - P1) - tmp_delta_p;

        residuals[0] = T(0.2) * residual1[0];
        residuals[1] = T(0.2) * residual1[1];
        residuals[2] = T(0.2) * residual1[2];
        residuals[3] = T(0.2) * residual2[0];
        residuals[4] = T(0.2) * residual2[1];
        residuals[5] = T(0.2) * residual2[2];

        return true;
    }

    static ceres::CostFunction *Create(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_)
    {
        return (new ceres::AutoDiffCostFunction<LidarPoseFactorAutoDiff, 6, 3, 4, 3, 4>(
                    new LidarPoseFactorAutoDiff(delta_q_, delta_p_)));
    }

private:
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_p;
};




struct LidarPoseLeftFactorAutoDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarPoseLeftFactorAutoDiff(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_, Eigen::Quaterniond ql_, Eigen::Vector3d pl_): delta_q(delta_q_), delta_p(delta_p_), ql(ql_), pl(pl_)
    {}

    template <typename T>
    bool operator()(const T *p1, const T *p2, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> P1(T(pl.x()), T(pl.y()), T(pl.z()));
        Eigen::Quaternion<T> Q1(T(ql.w()), T(ql.x()), T(ql.y()), T(ql.z()));

        Eigen::Matrix<T, 3, 1> P2(p1[0], p1[1], p1[2]);
        Eigen::Quaternion<T> Q2(p2[0], p2[1], p2[2], p2[3]);

        Eigen::Quaternion<T> tmp_delta_q(T(delta_q.w()), T(delta_q.x()), T(delta_q.y()), T(delta_q.z()));
        Eigen::Matrix<T, 3, 1> tmp_delta_p(T(delta_p.x()), T(delta_p.y()), T(delta_p.z()));

        Eigen::Matrix<T, 3, 1> residual1 = T(2.0) * (tmp_delta_q.inverse() * Q1.inverse() * Q2).vec();
        Eigen::Matrix<T, 3, 1> residual2 = Q1.inverse() * (P2 - P1) - tmp_delta_p;

        residuals[0] = T(0.2) * residual1[0];
        residuals[1] = T(0.2) * residual1[1];
        residuals[2] = T(0.2) * residual1[2];
        residuals[3] = T(0.2) * residual2[0];
        residuals[4] = T(0.2) * residual2[1];
        residuals[5] = T(0.2) * residual2[2];


        return true;
    }

    static ceres::CostFunction *Create(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_, Eigen::Quaterniond ql_, Eigen::Vector3d pl_)
    {
        return (new ceres::AutoDiffCostFunction<LidarPoseLeftFactorAutoDiff, 6, 3, 4>(
                    new LidarPoseLeftFactorAutoDiff(delta_q_, delta_p_, ql_, pl_)));
    }

private:
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond ql;
    Eigen::Vector3d pl;
};




struct LidarPoseRightFactorAutoDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    LidarPoseRightFactorAutoDiff(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_, Eigen::Quaterniond qr_, Eigen::Vector3d pr_): delta_q(delta_q_), delta_p(delta_p_), qr(qr_), pr(pr_)
    {}

    template <typename T>
    bool operator()(const T *p1, const T *p2, T *residuals) const
    {
        Eigen::Matrix<T, 3, 1> P2(T(pr.x()), T(pr.y()), T(pr.z()));
        Eigen::Quaternion<T> Q2(T(qr.w()), T(qr.x()), T(qr.y()), T(qr.z()));

        Eigen::Matrix<T, 3, 1> P1(p1[0], p1[1], p1[2]);
        Eigen::Quaternion<T> Q1(p2[0], p2[1], p2[2], p2[3]);

        Eigen::Quaternion<T> tmp_delta_q(T(delta_q.w()), T(delta_q.x()), T(delta_q.y()), T(delta_q.z()));
        Eigen::Matrix<T, 3, 1> tmp_delta_p(T(delta_p.x()), T(delta_p.y()), T(delta_p.z()));

        Eigen::Matrix<T, 3, 1> residual1 = T(2.0) * (tmp_delta_q.inverse() * Q1.inverse() * Q2).vec();
        Eigen::Matrix<T, 3, 1> residual2 = Q1.inverse() * (P2 - P1) - tmp_delta_p;

        residuals[0] = T(0.2) * residual1[0];
        residuals[1] = T(0.2) * residual1[1];
        residuals[2] = T(0.2) * residual1[2];
        residuals[3] = T(0.2) * residual2[0];
        residuals[4] = T(0.2) * residual2[1];
        residuals[5] = T(0.2) * residual2[2];


        return true;
    }

    static ceres::CostFunction *Create(Eigen::Quaterniond delta_q_, Eigen::Vector3d delta_p_, Eigen::Quaterniond qr_, Eigen::Vector3d pr_)
    {
        return (new ceres::AutoDiffCostFunction<LidarPoseRightFactorAutoDiff, 6, 3, 4>(
                    new LidarPoseRightFactorAutoDiff(delta_q_, delta_p_, qr_, pr_)));
    }

private:
    Eigen::Quaterniond delta_q;
    Eigen::Vector3d delta_p;
    Eigen::Quaterniond qr;
    Eigen::Vector3d pr;
};
