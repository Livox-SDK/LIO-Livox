#ifndef LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
#define LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
#include <ros/ros.h>
#include <livox_ros_driver/CustomMsg.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <future>
#include "opencv2/core.hpp"
#include "segment/segment.hpp"
class LidarFeatureExtractor{
    typedef pcl::PointXYZINormal PointType;
public:
    /** \brief constructor of LidarFeatureExtractor
      * \param[in] n_scans: lines used to extract lidar features
      */
    LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,int PartNum,float FlatThreshold,
                          float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis);

    /** \brief transform float to int
      */
    static uint32_t _float_as_int(float f){
      union{uint32_t i; float f;} conv{};
      conv.f = f;
      return conv.i;
    }

    /** \brief transform int to float
      */
    static float _int_as_float(uint32_t i){
      union{float f; uint32_t i;} conv{};
      conv.i = i;
      return conv.f;
    }

    /** \brief Determine whether the point_list is flat
      * \param[in] point_list: points need to be judged
      * \param[in] plane_threshold
      */
    bool plane_judge(const std::vector<PointType>& point_list,const int plane_threshold);

    /** \brief Detect lidar feature points
      * \param[in] cloud: original lidar cloud need to be detected
      * \param[in] pointsLessSharp: less sharp index of cloud
      * \param[in] pointsLessFlat: less flat index of cloud
      */
    void detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                            std::vector<int>& pointsLessSharp,
                            std::vector<int>& pointsLessFlat);

    void detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                             pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                             pcl::PointCloud<PointType>::Ptr& pointsNonFeature);

    void detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                             std::vector<int>& pointsLessSharp);
                
    void FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                     pcl::PointCloud<PointType>::Ptr& laserCloud,
                                     pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                     pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                     pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                     sensor_msgs::PointCloud2 &msg2,
                                     int Used_Line = 1);
    /** \brief Detect lidar feature points of CustomMsg
      * \param[in] msg: original CustomMsg need to be detected
      * \param[in] laserCloud: transform CustomMsg to pcl point cloud format
      * \param[in] laserConerFeature: less Coner features extracted from laserCloud
      * \param[in] laserSurfFeature: less Surf features extracted from laserCloud
      */
    void FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                        int Used_Line = 1);

private:
    /** \brief lines used to extract lidar features */
    const int N_SCANS;

    /** \brief store original points of each line */
    std::vector<pcl::PointCloud<PointType>::Ptr> vlines;

    /** \brief store corner feature index of each line */
    std::vector<std::vector<int>> vcorner;

    /** \brief store surf feature index of each line */
    std::vector<std::vector<int>> vsurf;

    int thNumCurvSize;

    float thDistanceFaraway;

    int thNumFlat;
    
    int thPartNum;

    float thFlatThreshold;

    float thBreakCornerDis;

    float thLidarNearestDis;  
};

#endif //LIO_LIVOX_LIDARFEATUREEXTRACTOR_H
