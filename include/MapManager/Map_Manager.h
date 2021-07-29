#ifndef LIO_LIVOX_MAP_MANAGER_H
#define LIO_LIVOX_MAP_MANAGER_H
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <future>
class MAP_MANAGER{
    typedef pcl::PointXYZINormal PointType;
public:

    std::mutex mtx_MapManager;
    /** \brief constructor of MAP_MANAGER */
    MAP_MANAGER(const float& filter_corner, const float& filter_surf);

    static size_t ToIndex(int i, int j, int k);

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

    /** \brief transform point pi to the MAP coordinate
     * \param[in] pi: point to be transformed
     * \param[in] po: point after transfomation
     * \param[in] _transformTobeMapped: transform matrix between pi and po
     */
    static void pointAssociateToMap(PointType const * const pi,
                                    PointType * const po,
                                    const Eigen::Matrix4d& _transformTobeMapped);

    void featureAssociateToMap(const pcl::PointCloud<PointType>::Ptr& laserCloudCorner,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudSurf,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeature,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudCornerToMap,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudSurfToMap,
                               const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureToMap,
                               const Eigen::Matrix4d& transformTobeMapped);
    /** \brief add new lidar points to the map
     * \param[in] laserCloudCornerStack: coner feature points that need to be added to map
     * \param[in] laserCloudSurfStack: surf feature points that need to be added to map
     * \param[in] transformTobeMapped: transform matrix of the lidar pose
     */
    void MapIncrement(const pcl::PointCloud<PointType>::Ptr& laserCloudCornerStack,
                      const pcl::PointCloud<PointType>::Ptr& laserCloudSurfStack,
                      const pcl::PointCloud<PointType>::Ptr& laserCloudNonFeatureStack,
                      const Eigen::Matrix4d& transformTobeMapped);

    /** \brief retrieve map points according to the lidar pose
     * \param[in] laserCloudCornerFromMap: store coner feature points retrieved from map
     * \param[in] laserCloudSurfFromMap: tore surf feature points retrieved from map
     * \param[in] transformTobeMapped: transform matrix of the lidar pose
     */
    void MapMove(const Eigen::Matrix4d& transformTobeMapped);


    size_t FindUsedCornerMap(const PointType *p,int a,int b,int c);

    size_t FindUsedSurfMap(const PointType *p,int a,int b,int c);

    size_t FindUsedNonFeatureMap(const PointType *p,int a,int b,int c);

    pcl::KdTreeFLANN<PointType> getCornerKdMap(int i){
      return CornerKdMap_last[i];
    }
    pcl::KdTreeFLANN<PointType> getSurfKdMap(int i){
      return SurfKdMap_last[i];
    }
    pcl::KdTreeFLANN<PointType> getNonFeatureKdMap(int i){
      return NonFeatureKdMap_last[i];
    }
		pcl::PointCloud<PointType>::Ptr get_corner_map(){
			return laserCloudCornerFromMap;
		}
		pcl::PointCloud<PointType>::Ptr get_surf_map(){
			return laserCloudSurfFromMap;
		}
    pcl::PointCloud<PointType>::Ptr get_nonfeature_map(){
			return laserCloudNonFeatureFromMap;
		}
    int get_map_current_pos(){
      return currentUpdatePos;
    }
    int get_laserCloudCenWidth_last(){
      return laserCloudCenWidth_last;
    }
    int get_laserCloudCenHeight_last(){
      return laserCloudCenHeight_last;
    }
    int get_laserCloudCenDepth_last(){
      return laserCloudCenDepth_last;
    }
    pcl::PointCloud<PointType> laserCloudSurf_for_match[4851];
    pcl::PointCloud<PointType> laserCloudCorner_for_match[4851];
    pcl::PointCloud<PointType> laserCloudNonFeature_for_match[4851];

private:
    int laserCloudCenWidth = 10;
    int laserCloudCenHeight = 5;
    int laserCloudCenDepth = 10;

    int laserCloudCenWidth_last = 10;
    int laserCloudCenHeight_last = 5;
    int laserCloudCenDepth_last = 10;

    static const int laserCloudWidth = 21;
    static const int laserCloudHeight = 11;
    static const int laserCloudDepth = 21;
    static const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;//4851
    pcl::PointCloud<PointType>::Ptr laserCloudCornerArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudSurfArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureArray[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudCornerArrayStack[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudSurfArrayStack[laserCloudNum];
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureArrayStack[laserCloudNum];

    pcl::VoxelGrid<PointType> downSizeFilterCorner;
    pcl::VoxelGrid<PointType> downSizeFilterSurf;
    pcl::VoxelGrid<PointType> downSizeFilterNonFeature;

    pcl::PointCloud<PointType>::Ptr laserCloudCornerFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudSurfFromMap;
    pcl::PointCloud<PointType>::Ptr laserCloudNonFeatureFromMap;

    pcl::KdTreeFLANN<PointType>::Ptr laserCloudCornerKdMap[laserCloudNum];
    pcl::KdTreeFLANN<PointType>::Ptr laserCloudSurfKdMap[laserCloudNum];
    pcl::KdTreeFLANN<PointType>::Ptr laserCloudNonFeatureKdMap[laserCloudNum];

    pcl::KdTreeFLANN<PointType> CornerKdMap_copy[laserCloudNum];
    pcl::KdTreeFLANN<PointType> SurfKdMap_copy[laserCloudNum];
    pcl::KdTreeFLANN<PointType> NonFeatureKdMap_copy[laserCloudNum];

    pcl::KdTreeFLANN<PointType> CornerKdMap_last[laserCloudNum];
    pcl::KdTreeFLANN<PointType> SurfKdMap_last[laserCloudNum];
    pcl::KdTreeFLANN<PointType> NonFeatureKdMap_last[laserCloudNum];

    static const int localMapWindowSize = 60;
    pcl::PointCloud<PointType>::Ptr localCornerMap[localMapWindowSize];
    pcl::PointCloud<PointType>::Ptr localSurfMap[localMapWindowSize];
    pcl::PointCloud<PointType>::Ptr localNonFeatureMap[localMapWindowSize];

    int localMapID = 0;

    int currentUpdatePos = 0;
    int estimatorPos = 0;
};

#endif //LIO_LIVOX_MAP_MANAGER_H
