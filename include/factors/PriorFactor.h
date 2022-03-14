#pragma once
#include <ros/assert.h>
#include <iostream>
#include <eigen3/Eigen/Dense>

#include "utils/math_tools.h"

#include <ceres/ceres.h>

struct SpeedBiasPriorFactorAutoDiff
{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    SpeedBiasPriorFactorAutoDiff(vector<double> speedBias) : speedBias_(speedBias) {}

    template <typename T>
    bool operator()(const T *speedBias, T *residuals) const
    {
        residuals[0] = T(8) * (speedBias[0] - T(speedBias_[0]));
        residuals[1] = T(8) * (speedBias[1] - T(speedBias_[1]));
        residuals[2] = T(1) * (speedBias[2] - T(speedBias_[2]));
        residuals[3] = T(1) * (speedBias[3] - T(speedBias_[3]));
        residuals[4] = T(1) * (speedBias[4] - T(speedBias_[4]));
        residuals[5] = T(1) * (speedBias[5] - T(speedBias_[5]));
        residuals[6] = T(1) * (speedBias[6] - T(speedBias_[6]));
        residuals[7] = T(1) * (speedBias[7] - T(speedBias_[7]));
        residuals[8] = T(1) * (speedBias[8] - T(speedBias_[8]));

        return true;
    }

    static ceres::CostFunction *Create(vector<double> speedBias_)
    {
        return (new ceres::AutoDiffCostFunction<SpeedBiasPriorFactorAutoDiff, 9, 9>(
                    new SpeedBiasPriorFactorAutoDiff(speedBias_)));
    }

private:
    vector<double> speedBias_;
};
