#ifndef UTILS_H
#define UTILS_H

#include <ros/ros.h>

#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/Imu.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <eigen_conversions/eigen_msg.h>
#include <message_filters/subscriber.h>
#include <geometry_msgs/Vector3Stamped.h>
#include <geometry_msgs/QuaternionStamped.h>
#include <sensor_msgs/NavSatFix.h>

#include <tf/transform_listener.h>
#include <tf/transform_broadcaster.h>
#include <tf/transform_datatypes.h>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_ros/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/pcl_macros.h>
#include <pcl/range_image/range_image.h>
#include <pcl/registration/icp.h>

#include <cmath>
#include <ctime>
#include <array>
#include <string>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <thread>
#include <mutex>
#include <queue>
#include <assert.h>

#include <Eigen/Dense>

using namespace std;

struct PointPoseInfo
{
    double x;
    double y;
    double z;
    double qw;
    double qx;
    double qy;
    double qz;
    int idx;
    double time;
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
} EIGEN_ALIGN16;

POINT_CLOUD_REGISTER_POINT_STRUCT (PointPoseInfo,
                                   (double, x, x) (double, y, y) (double, z, z)
                                   (double, qw, qw) (double, qx, qx) (double, qy, qy) (double, qz, qz)
                                   (int, idx, idx) (double, time, time)
                                   )

// PCL point types
using pcl::PointXYZI;
//using pcl::PointXYZINormal;
typedef pcl::PointXYZI PointType;



// Get parameters from yaml file
template <class class_name>
bool getParameter(const std::string& paramName, class_name& param)
{
    std::string nodeName = ros::this_node::getName();
    std::string paramKey;
    if (!ros::param::search(paramName, paramKey))
    {
        ROS_ERROR("%s: Failed to search for parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    if (!ros::param::has(paramKey))
    {
        ROS_ERROR("%s: Missing required parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    if (!ros::param::get(paramKey, param))
    {
        ROS_ERROR("%s: Failed to get parameter '%s'.", nodeName.c_str(), paramName.c_str());
        return false;
    }

    return true;
}

#endif // UTILS_H
