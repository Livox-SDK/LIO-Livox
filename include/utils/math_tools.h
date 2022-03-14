#ifndef MATH_UTILS_H
#define MATH_UTILS_H

#include <cmath>
#include <eigen3/Eigen/Dense>

// Sign function
template <typename T>
T sgnFunc(T val)
{
    return (T(0) < val) - (val < T(0));
}

// Hat (skew) operator
template <typename T>
inline Eigen::Matrix<T, 3, 3> hat(const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 3> mat;
    mat << 0, -vec(2), vec(1),
            vec(2), 0, -vec(0),
            -vec(1), vec(0), 0;
    return mat;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 3, 3> skewSymmetric(const Eigen::MatrixBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 3, 3> ans;
    ans << typename Derived::Scalar(0), -q(2), q(1),
            q(2), typename Derived::Scalar(0), -q(0),
            -q(1), q(0), typename Derived::Scalar(0);
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qleft(const Eigen::QuaternionBase<Derived> &q)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = q.w(), ans.template block<1, 3>(0, 1) = -q.vec().transpose();
    ans.template block<3, 1>(1, 0) = q.vec(), ans.template block<3, 3>(1, 1) = q.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() + skewSymmetric(q.vec());
    return ans;
}

template <typename Derived>
static Eigen::Matrix<typename Derived::Scalar, 4, 4> Qright(const Eigen::QuaternionBase<Derived> &p)
{
    Eigen::Matrix<typename Derived::Scalar, 4, 4> ans;
    ans(0, 0) = p.w(), ans.template block<1, 3>(0, 1) = -p.vec().transpose();
    ans.template block<3, 1>(1, 0) = p.vec(), ans.template block<3, 3>(1, 1) = p.w() * Eigen::Matrix<typename Derived::Scalar, 3, 3>::Identity() - skewSymmetric(p.vec());
    return ans;
}

// Convert from quaternion to rotation vector
template <typename T>
inline Eigen::Matrix<T, 3, 1> quaternionToRotationVector(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 3> mat = qua.toRotationMatrix();
    Eigen::Matrix<T, 3, 1> rotation_vec;
    Eigen::AngleAxis<T> angle_axis;
    angle_axis.fromRotationMatrix(mat);
    rotation_vec = angle_axis.angle() * angle_axis.axis();
    return rotation_vec;
}

// Right Jacobian matrix
template <typename T>
inline Eigen::Matrix3d Jright(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 3> mat;
    Eigen::Matrix<T, 3, 1> rotation_vec = quaternionToRotationVector(qua);
    double theta_norm = rotation_vec.norm();
    mat = Eigen::Matrix<T, 3, 3>::Identity()
            - (1 - cos(theta_norm)) / (theta_norm * theta_norm + 1e-10) * hat(rotation_vec)
            + (theta_norm - sin(theta_norm)) / (theta_norm * theta_norm * theta_norm + 1e-10) * hat(rotation_vec) * hat(rotation_vec);
    return mat;
}

// Calculate the Jacobian with respect to the quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + quaternion_imaginary.cross(vec);
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity()
            + quaternion_imaginary * vec.transpose()
            - vec * quaternion_imaginary.transpose()
            - qua.w() * hat(vec);
    return T(2) * mat;
}

// Calculate the Jacobian with respect to the inverse quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> quaternionInvJacobian(const Eigen::Quaternion<T> &qua, const Eigen::Matrix<T, 3, 1> &vec)
{
    Eigen::Matrix<T, 3, 4> mat;
    Eigen::Matrix<T, 3, 1> quaternion_imaginary(qua.x(), qua.y(), qua.z());

    mat.template block<3, 1>(0, 0) = qua.w() * vec + vec.cross(quaternion_imaginary);
    mat.template block<3, 3>(0, 1) = quaternion_imaginary.dot(vec) * Eigen::Matrix<T, 3, 3>::Identity()
            + quaternion_imaginary * vec.transpose()
            - vec * quaternion_imaginary.transpose()
            + qua.w() * hat(vec);
    return T(2) * mat;
}

// Calculate the Jacobian rotation vector to quaternion
template <typename T>
inline Eigen::Matrix<T, 3, 4> JacobianV2Q(const Eigen::Quaternion<T> &qua)
{
    Eigen::Matrix<T, 3, 4> mat;

    T c = 1 / (1 - qua.w() * qua.w());
    T d = acos(qua.w()) / sqrt(1 - qua.w() * qua.w());

    mat.template block<3, 1>(0, 0) = Eigen::Matrix<T, 3, 1>(c * qua.x() * (d * qua.x() - 1),
                                                            c * qua.y() * (d * qua.x() - 1),
                                                            c * qua.z() * (d * qua.x() - 1));
    mat.template block<3, 3>(0, 1) = d * Eigen::Matrix<T, 3, 4>::Identity();
    return T(2) * mat;
}

//get quaternion from rotation vector
template <typename Derived>
Eigen::Quaternion<typename Derived::Scalar> deltaQ(const Eigen::MatrixBase<Derived> &theta)
{
    typedef typename Derived::Scalar Scalar_t;

    Eigen::Quaternion<Scalar_t> dq;
    Eigen::Matrix<Scalar_t, 3, 1> half_theta = theta;
    half_theta /= static_cast<Scalar_t>(2.0);
    dq.w() = static_cast<Scalar_t>(1.0);
    dq.x() = half_theta.x();
    dq.y() = half_theta.y();
    dq.z() = half_theta.z();
    return dq;
}

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4> LeftQuatMatrix(const Eigen::QuaternionBase<Derived> &q) {
    Eigen::Matrix<typename Derived::Scalar, 4, 4> m;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> vq = q.vec();
    typename Derived::Scalar q4 = q.w();
    m.block(0, 0, 3, 3) << q4 * Eigen::Matrix3d::Identity() + skewSymmetric(vq);
    m.block(3, 0, 1, 3) << -vq.transpose();
    m.block(0, 3, 3, 1) << vq;
    m(3, 3) = q4;
    return m;
}

template<typename Derived>
inline Eigen::Matrix<typename Derived::Scalar, 4, 4> RightQuatMatrix(const Eigen::QuaternionBase<Derived> &p) {
    Eigen::Matrix<typename Derived::Scalar, 4, 4> m;
    Eigen::Matrix<typename Derived::Scalar, 3, 1> vp = p.vec();
    typename Derived::Scalar p4 = p.w();
    m.block(0, 0, 3, 3) << p4 * Eigen::Matrix3d::Identity() - skewSymmetric(vp);
    m.block(3, 0, 1, 3) << -vp.transpose();
    m.block(0, 3, 3, 1) << vp;
    m(3, 3) = p4;
    return m;
}


template <typename T>
Eigen::Quaternion<T> unifyQuaternion(const Eigen::Quaternion<T> &q)
{
    if(q.w() >= 0) return q;
    else {
        Eigen::Quaternion<T> resultQ(-q.w(), -q.x(), -q.y(), -q.z());
        return resultQ;
    }
}

#endif // MATH_UTILS_H
