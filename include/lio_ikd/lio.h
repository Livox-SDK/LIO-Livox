#ifndef LIO_IKD_ESTIMATOR_H
#define LIO_IKD_ESTIMATOR_H

#include <ros/ros.h>
#include <pcl_conversions/pcl_conversions.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/NavSatFix.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <tf/tf.h>
#include <tf/transform_broadcaster.h>
#include <Eigen/Core>
#include <sensor_msgs/Imu.h>
#include <queue>
#include <iterator>
#include <future>

#include <Eigen/Eigen>
#include <Eigen/Core>
#include <Eigen/Dense>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>

#include "Estimator/Map_Manager.h"
#include <ikd-Tree/ikd_Tree.h>
#include "Estimator/IMUIntegrator.h"
#include "Estimator/ceresfunc.h"
#include <chrono>

#define NUM_MATCH_POINTS (5)

class LIO
{
    typedef pcl::PointXYZINormal PointType;
    typedef vector<PointType, Eigen::aligned_allocator<PointType>> PointVector;

public:
    /** \brief slide window size */
    static const int SLIDEWINDOWSIZE = 2;

    vector<PointVector> Nearest_Points_corner, Nearest_Points_surf;

    float calc_dist(PointType p1, PointType p2)
    {
        float d = (p1.x - p2.x) * (p1.x - p2.x) + (p1.y - p2.y) * (p1.y - p2.y) + (p1.z - p2.z) * (p1.z - p2.z);
        return d;
    }

    /** \brief lidar frame struct */
    struct LidarFrame
    {
        pcl::PointCloud<PointType>::Ptr laserCloud;
        IMUIntegrator imuIntegrator;
        Eigen::Vector3d P;
        Eigen::Vector3d V;
        Eigen::Quaterniond Q;
        Eigen::Vector3d bg;
        Eigen::Vector3d ba;
        double timeStamp;
        LidarFrame()
        {
            P.setZero();
            V.setZero();
            Q.setIdentity();
            bg.setZero();
            ba.setZero();
            timeStamp = 0;
        }
    };

    /** \brief point to line feature */
    struct FeatureLine
    {
        Eigen::Vector3d pointOri;
        Eigen::Vector3d lineP1;
        Eigen::Vector3d lineP2;
        double error;
        bool valid;
        FeatureLine(Eigen::Vector3d po, Eigen::Vector3d p1, Eigen::Vector3d p2)
            : pointOri(std::move(po)), lineP1(std::move(p1)), lineP2(std::move(p2))
        {
            valid = false;
            error = 0;
        }
        double ComputeError(const Eigen::Matrix4d &pose)
        {
            Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
            double l12 = std::sqrt((lineP1(0) - lineP2(0)) * (lineP1(0) - lineP2(0)) + (lineP1(1) - lineP2(1)) * (lineP1(1) - lineP2(1)) + (lineP1(2) - lineP2(2)) * (lineP1(2) - lineP2(2)));
            double a012 = std::sqrt(
                ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) * ((P_to_Map(0) - lineP1(0)) * (P_to_Map(1) - lineP2(1)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(1) - lineP1(1))) + ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) * ((P_to_Map(0) - lineP1(0)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(0) - lineP2(0)) * (P_to_Map(2) - lineP1(2))) + ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))) * ((P_to_Map(1) - lineP1(1)) * (P_to_Map(2) - lineP2(2)) - (P_to_Map(1) - lineP2(1)) * (P_to_Map(2) - lineP1(2))));
            error = a012 / l12;
        }
    };

    /** \brief point to plan feature */
    struct FeaturePlan
    {
        Eigen::Vector3d pointOri;
        double pa;
        double pb;
        double pc;
        double pd;
        double error;
        bool valid;
        FeaturePlan(const Eigen::Vector3d &po, const double &pa_, const double &pb_, const double &pc_, const double &pd_)
            : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_)
        {
            valid = false;
            error = 0;
        }
        double ComputeError(const Eigen::Matrix4d &pose)
        {
            Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
            error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
        }
    };

    /** \brief point to plan feature */
    struct FeaturePlanVec
    {
        Eigen::Vector3d pointOri;
        Eigen::Vector3d pointProj;
        Eigen::Matrix3d sqrt_info;
        double error;
        bool valid;
        FeaturePlanVec(const Eigen::Vector3d &po, const Eigen::Vector3d &p_proj, Eigen::Matrix3d sqrt_info_)
            : pointOri(po), pointProj(p_proj), sqrt_info(sqrt_info_)
        {
            valid = false;
            error = 0;
        }
        double ComputeError(const Eigen::Matrix4d &pose)
        {
            Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
            error = (P_to_Map - pointProj).norm();
        }
    };

    /** \brief non feature */
    struct FeatureNon
    {
        Eigen::Vector3d pointOri;
        double pa;
        double pb;
        double pc;
        double pd;
        double error;
        bool valid;
        FeatureNon(const Eigen::Vector3d &po, const double &pa_, const double &pb_, const double &pc_, const double &pd_)
            : pointOri(po), pa(pa_), pb(pb_), pc(pc_), pd(pd_)
        {
            valid = false;
            error = 0;
        }
        double ComputeError(const Eigen::Matrix4d &pose)
        {
            Eigen::Vector3d P_to_Map = pose.topLeftCorner(3, 3) * pointOri + pose.topRightCorner(3, 1);
            error = pa * P_to_Map(0) + pb * P_to_Map(1) + pc * P_to_Map(2) + pd;
        }
    };

public:
    /** \brief constructor of LIO
     */
    LIO(const float &filter_corner, const float &filter_surf);

    ~LIO();

    /** \brief construct sharp feature Ceres Costfunctions
     * \param[in] edges: store costfunctions
     * \param[in] m4d: lidar pose, represented by matrix 4X4
     */
    void processPointToLine(std::vector<ceres::CostFunction *> &edges,
                            std::vector<FeatureLine> &vLineFeatures,
                            const pcl::PointCloud<PointType>::Ptr &laserCloudCorner,
                            const Eigen::Matrix4d &exTlb,
                            const Eigen::Matrix4d &m4d);

    /** \brief construct Plan feature Ceres Costfunctions
     * \param[in] edges: store costfunctions
     * \param[in] m4d: lidar pose, represented by matrix 4X4
     */
    void processPointToPlanVec(std::vector<ceres::CostFunction *> &edges,
                               std::vector<FeaturePlanVec> &vPlanFeatures,
                               const pcl::PointCloud<PointType>::Ptr &laserCloudSurf,
                               const Eigen::Matrix4d &exTlb,
                               const Eigen::Matrix4d &m4d);

    /** \brief Transform Lidar Pose in slidewindow to double array
     * \param[in] lidarFrameList: Lidar Poses in slidewindow
     */
    void vector2double(const std::list<LidarFrame> &lidarFrameList);

    /** \brief Transform double array to Lidar Pose in slidewindow
     * \param[in] lidarFrameList: Lidar Poses in slidewindow
     */
    void double2vector(std::list<LidarFrame> &lidarFrameList);

    /** \brief estimate lidar pose by matching current lidar cloud with map cloud and tightly coupled IMU message
     * \param[in] lidarFrameList: multi-frames of lidar cloud and lidar pose
     * \param[in] exTlb: extrinsic matrix between lidar and IMU
     * \param[in] gravity: gravity vector
     */
    void EstimateLidarPose(std::list<LidarFrame> &lidarFrameList,
                           const Eigen::Matrix4d &exTlb,
                           const Eigen::Vector3d &gravity,
                           nav_msgs::Odometry &debugInfo);

    void Estimate(std::list<LidarFrame> &lidarFrameList,
                  const Eigen::Matrix4d &exTlb,
                  const Eigen::Vector3d &gravity);

    void LasermapFovSegment_corner(Eigen::Vector3d &);
    void LasermapFovSegment_surf(Eigen::Vector3d &);

    /*pcl::PointCloud<PointType>::Ptr get_corner_map()
    {
        return map_manager->get_corner_map();
    }
    pcl::PointCloud<PointType>::Ptr get_surf_map()
    {
        return map_manager->get_surf_map();
    }*/
    void MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr &laserCloudCornerStack,
                           const pcl::PointCloud<PointType>::Ptr &laserCloudSurfStack,
                           const Eigen::Matrix4d &transformTobeMapped);

    void setPublishHandle(std::function<bool(const std::string &topic_name, const pcl::PointCloud<PointType>::Ptr &cloud_ptr)> &func)
    {
        pub_cloud_to_ros_ = func;
    }

private:
    std::function<bool(const std::string &topic_name, const pcl::PointCloud<PointType>::Ptr &cloud_ptr)> pub_cloud_to_ros_;
    /** \brief store map points */
    // MAP_MANAGER *map_manager;
    KD_TREE<PointType> ikdtree_surf, ikdtree_corner;
    bool flag_inited;

    std::vector<BoxPointType> cub_needrm;
    bool Localmap_init_corner = false;
    bool Localmap_init_surf = false;
    BoxPointType LocalMap_Points_corner, LocalMap_Points_surf;
    double cube_len = 800;

    const float MOV_THRESHOLD = 1.5f;
    float DET_RANGE = 200.f;

   double para_PR[SLIDEWINDOWSIZE][6];
    double para_VBias[SLIDEWINDOWSIZE][9];

    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerLast;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfLast;

    pcl::PointCloud<PointType>::Ptr feats_surf_world;
    pcl::PointCloud<PointType>::Ptr feats_corner_world;
    pcl::PointCloud<PointType>::Ptr featrue_world;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromLocal;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromLocal;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerForMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfForMap;

    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudCornerStack;
    std::vector<pcl::PointCloud<PointType>::Ptr> laserCloudSurfStack;

    pcl::KdTreeFLANN<PointType>::Ptr kdtreeCornerFromLocal;
    pcl::KdTreeFLANN<PointType>::Ptr kdtreeSurfFromLocal;
    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;

    std::mutex mtx_Map;

    pcl::KdTreeFLANN<PointType> CornerKdMap[10000];
    pcl::KdTreeFLANN<PointType> SurfKdMap[10000];

    pcl::PointCloud<PointType> GlobalSurfMap[10000];
    pcl::PointCloud<PointType> GlobalCornerMap[10000];

    static const int localMapWindowSize = 30;
    int localMapID = 0;
    pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
    pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];

    int map_update_ID = 0;

    int map_skip_frame = 3; // every map_skip_frame frame update map
    double plan_weight_tan = 0.0;
    double thres_dist = 1.0;

    double filter_size_map_min = 0.5;
};

#endif // LIO_IKD_ESTIMATOR_H
