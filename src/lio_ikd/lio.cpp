#include "lio_ikd/lio.h"

// #define USE_VOXEL_MAP_

LIO::LIO(const float &filter_corner, const float &filter_surf)
{
    laserCloudCornerFromLocal.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfFromLocal.reset(new pcl::PointCloud<PointType>);

    feats_surf_world.reset(new pcl::PointCloud<PointType>);
    feats_corner_world.reset(new pcl::PointCloud<PointType>);
    featrue_world.reset(new pcl::PointCloud<PointType>);

    laserCloudCornerLast.resize(SLIDEWINDOWSIZE);
    for (auto &p : laserCloudCornerLast)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfLast.resize(SLIDEWINDOWSIZE);
    for (auto &p : laserCloudSurfLast)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudCornerStack.resize(SLIDEWINDOWSIZE);
    for (auto &p : laserCloudCornerStack)
        p.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfStack.resize(SLIDEWINDOWSIZE);
    for (auto &p : laserCloudSurfStack)
        p.reset(new pcl::PointCloud<PointType>);

    laserCloudCornerForMap.reset(new pcl::PointCloud<PointType>);
    laserCloudSurfForMap.reset(new pcl::PointCloud<PointType>);

    kdtreeCornerFromLocal.reset(new pcl::KdTreeFLANN<PointType>);
    kdtreeSurfFromLocal.reset(new pcl::KdTreeFLANN<PointType>);

    for (int i = 0; i < localMapWindowSize; i++)
    {
        localCornerMap[i].reset(new pcl::PointCloud<PointType>);
        localSurfMap[i].reset(new pcl::PointCloud<PointType>);
    }

    downSizeFilterCorner.setLeafSize(filter_corner, filter_corner, filter_corner);
    downSizeFilterSurf.setLeafSize(filter_surf, filter_surf, filter_surf);

    flag_inited = false;
}

LIO::~LIO()
{
}

void LIO::processPointToLine(std::vector<ceres::CostFunction *> &edges,
                             std::vector<FeatureLine> &vLineFeatures,
                             const pcl::PointCloud<PointType>::Ptr &laserCloudCorner,
                             const Eigen::Matrix4d &exTlb,
                             const Eigen::Matrix4d &m4d)
{

    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3, 3) = exTlb.topLeftCorner(3, 3).transpose();
    Tbl.topRightCorner(3, 1) = -1.0 * Tbl.topLeftCorner(3, 3) * exTlb.topRightCorner(3, 1);
    if (!vLineFeatures.empty())
    {

        for (const auto &l : vLineFeatures)
        {
            auto *e = Cost_NavState_IMU_Line::Create(l.pointOri,
                                                     l.lineP1,
                                                     l.lineP2,
                                                     Tbl,
                                                     Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));
            edges.push_back(e);
        }
        return;
    }

    Eigen::Matrix<double, 3, 3> _matA1;
    _matA1.setZero();

    int laserCloudCornerStackNum = laserCloudCorner->points.size();
    //  TODO: 多线程处理
    for (int i = 0; i < laserCloudCornerStackNum; i++)
    {
        PointType _pointOri, _pointSel, _coeff;
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points_corner[i];

        _pointOri = laserCloudCorner->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);
        ikdtree_corner.Nearest_Search(_pointSel, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

        if (points_near.size() >= NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] < thres_dist)
        {

            float cx = 0;
            float cy = 0;
            float cz = 0;
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                cx += points_near[j].x;
                cy += points_near[j].y;
                cz += points_near[j].z;
            }
            cx /= NUM_MATCH_POINTS;
            cy /= NUM_MATCH_POINTS;
            cz /= NUM_MATCH_POINTS;

            float a11 = 0;
            float a12 = 0;
            float a13 = 0;
            float a22 = 0;
            float a23 = 0;
            float a33 = 0;
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                float ax = points_near[j].x - cx;
                float ay = points_near[j].y - cy;
                float az = points_near[j].z - cz;

                a11 += ax * ax;
                a12 += ax * ay;
                a13 += ax * az;
                a22 += ay * ay;
                a23 += ay * az;
                a33 += az * az;
            }
            a11 /= NUM_MATCH_POINTS;
            a12 /= NUM_MATCH_POINTS;
            a13 /= NUM_MATCH_POINTS;
            a22 /= NUM_MATCH_POINTS;
            a23 /= NUM_MATCH_POINTS;
            a33 /= NUM_MATCH_POINTS;

            _matA1(0, 0) = a11;
            _matA1(0, 1) = a12;
            _matA1(0, 2) = a13;
            _matA1(1, 0) = a12;
            _matA1(1, 1) = a22;
            _matA1(1, 2) = a23;
            _matA1(2, 0) = a13;
            _matA1(2, 1) = a23;
            _matA1(2, 2) = a33;

            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> saes(_matA1);
            Eigen::Vector3d unit_direction = saes.eigenvectors().col(2);

            if (saes.eigenvalues()[2] > 3 * saes.eigenvalues()[1])
            {

                float x1 = cx + 0.1 * unit_direction[0];
                float y1 = cy + 0.1 * unit_direction[1];
                float z1 = cz + 0.1 * unit_direction[2];
                float x2 = cx - 0.1 * unit_direction[0];
                float y2 = cy - 0.1 * unit_direction[1];
                float z2 = cz - 0.1 * unit_direction[2];

                Eigen::Vector3d tripod1(x1, y1, z1);
                Eigen::Vector3d tripod2(x2, y2, z2);

                auto *e = Cost_NavState_IMU_Line::Create(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                                         tripod1,
                                                         tripod2,
                                                         Tbl,
                                                         Eigen::Matrix<double, 1, 1>(1 / IMUIntegrator::lidar_m));

                edges.push_back(e);
                vLineFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                           tripod1,
                                           tripod2);
                vLineFeatures.back().ComputeError(m4d);
            }
        }
    }
}

void LIO::processPointToPlanVec(std::vector<ceres::CostFunction *> &edges,
                                std::vector<FeaturePlanVec> &vPlanFeatures,
                                const pcl::PointCloud<PointType>::Ptr &laserCloudSurf,
                                const Eigen::Matrix4d &exTlb,
                                const Eigen::Matrix4d &m4d)
{
    Eigen::Matrix4d Tbl = Eigen::Matrix4d::Identity();
    Tbl.topLeftCorner(3, 3) = exTlb.topLeftCorner(3, 3).transpose();
    Tbl.topRightCorner(3, 1) = -1.0 * Tbl.topLeftCorner(3, 3) * exTlb.topRightCorner(3, 1);
    if (!vPlanFeatures.empty())
    {
        for (const auto &p : vPlanFeatures)
        {
            auto *e = Cost_NavState_IMU_Plan_Vec::Create(p.pointOri,
                                                         p.pointProj,
                                                         Tbl,
                                                         p.sqrt_info);
            edges.push_back(e);
        }
        return;
    }

    std::vector<int> _pointSearchInd;
    std::vector<float> _pointSearchSqDis;
    std::vector<int> _pointSearchInd2;
    std::vector<float> _pointSearchSqDis2;

    Eigen::Matrix<double, 5, 3> _matA0;
    _matA0.setZero();
    Eigen::Matrix<double, 5, 1> _matB0;
    _matB0.setOnes();
    _matB0 *= -1;
    Eigen::Matrix<double, 3, 1> _matX0;
    _matX0.setZero();

    int laserCloudSurfStackNum = laserCloudSurf->points.size();
    //  TODO: 多线程处理
    for (int i = 0; i < laserCloudSurfStackNum; i++)
    {
        PointType _pointOri, _pointSel, _coeff;
        vector<float> pointSearchSqDis(NUM_MATCH_POINTS);
        auto &points_near = Nearest_Points_surf[i];

        _pointOri = laserCloudSurf->points[i];
        MAP_MANAGER::pointAssociateToMap(&_pointOri, &_pointSel, m4d);

        ikdtree_surf.Nearest_Search(_pointSel, NUM_MATCH_POINTS, points_near, pointSearchSqDis);

        if (points_near.size() >= NUM_MATCH_POINTS && pointSearchSqDis[NUM_MATCH_POINTS - 1] < thres_dist)
        {
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                _matA0(j, 0) = points_near[j].x;
                _matA0(j, 1) = points_near[j].y;
                _matA0(j, 2) = points_near[j].z;
            }
            _matX0 = _matA0.colPivHouseholderQr().solve(_matB0);

            float pa = _matX0(0, 0);
            float pb = _matX0(1, 0);
            float pc = _matX0(2, 0);
            float pd = 1;

            float ps = std::sqrt(pa * pa + pb * pb + pc * pc);
            pa /= ps;
            pb /= ps;
            pc /= ps;
            pd /= ps;

            bool planeValid = true;
            for (int j = 0; j < NUM_MATCH_POINTS; j++)
            {
                if (std::fabs(pa * points_near[j].x +
                              pb * points_near[j].y +
                              pc * points_near[j].z + pd) > 0.2)
                {
                    planeValid = false;
                    break;
                }
            }

            if (planeValid)
            {
                double dist = pa * _pointSel.x +
                              pb * _pointSel.y +
                              pc * _pointSel.z + pd;
                Eigen::Vector3d omega(pa, pb, pc);
                Eigen::Vector3d point_proj = Eigen::Vector3d(_pointSel.x, _pointSel.y, _pointSel.z) - (dist * omega);
                Eigen::Vector3d e1(1, 0, 0);
                Eigen::Matrix3d J = e1 * omega.transpose();

                Eigen::JacobiSVD<Eigen::MatrixXd> svd(J, Eigen::ComputeThinU | Eigen::ComputeThinV);
                Eigen::Matrix3d R_svd = svd.matrixV() * svd.matrixU().transpose();
                Eigen::Matrix3d info = (1.0 / IMUIntegrator::lidar_m) * Eigen::Matrix3d::Identity();
                info(1, 1) *= plan_weight_tan;
                info(2, 2) *= plan_weight_tan;
                Eigen::Matrix3d sqrt_info = info * R_svd.transpose();
                auto *e = Cost_NavState_IMU_Plan_Vec::Create(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                                             point_proj,
                                                             Tbl,
                                                             sqrt_info);
                edges.push_back(e);
                vPlanFeatures.emplace_back(Eigen::Vector3d(_pointOri.x, _pointOri.y, _pointOri.z),
                                           point_proj,
                                           sqrt_info);
                vPlanFeatures.back().ComputeError(m4d);
            }
        }
    }
}

void LIO::vector2double(const std::list<LidarFrame> &lidarFrameList)
{
    int i = 0;
    for (const auto &l : lidarFrameList)
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
        PR.segment<3>(0) = l.P;
        PR.segment<3>(3) = Sophus::SO3d(l.Q).log();

        Eigen::Map<Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
        VBias.segment<3>(0) = l.V;
        VBias.segment<3>(3) = l.bg;
        VBias.segment<3>(6) = l.ba;
        i++;
    }
}

void LIO::double2vector(std::list<LidarFrame> &lidarFrameList)
{
    int i = 0;
    for (auto &l : lidarFrameList)
    {
        Eigen::Map<const Eigen::Matrix<double, 6, 1>> PR(para_PR[i]);
        Eigen::Map<const Eigen::Matrix<double, 9, 1>> VBias(para_VBias[i]);
        l.P = PR.segment<3>(0);
        l.Q = Sophus::SO3d::exp(PR.segment<3>(3)).unit_quaternion();
        l.V = VBias.segment<3>(0);
        l.bg = VBias.segment<3>(3);
        l.ba = VBias.segment<3>(6);
        i++;
    }
}

void LIO::LasermapFovSegment_corner(Eigen::Vector3d &pos_LID)
{
    cub_needrm.clear();
    // Eigen::Vector3d pos_LID = transformTobeMapped.topRightCorner(3, 1);
    if (!Localmap_init_corner)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points_corner.vertex_min[i] = pos_LID(i) - cube_len / 2.0;
            LocalMap_Points_corner.vertex_max[i] = pos_LID(i) + cube_len / 2.0;
        }
        Localmap_init_corner = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = std::fabs(pos_LID(i) - LocalMap_Points_corner.vertex_min[i]);
        dist_to_map_edge[i][1] = std::fabs(pos_LID(i) - LocalMap_Points_corner.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points_corner;
    float mov_dist = std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points_corner;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points_corner.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points_corner.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points_corner = New_LocalMap_Points;
    // points_cache_collect();
    PointVector points_history;
    ikdtree_corner.acquire_removed_points(points_history);
    if (cub_needrm.size() > 0)
        ikdtree_corner.Delete_Point_Boxes(cub_needrm);
}

void LIO::LasermapFovSegment_surf(Eigen::Vector3d &pos_LID)
{
    cub_needrm.clear();
    // Eigen::Vector3d pos_LID = transformTobeMapped.topRightCorner(3, 1);
    if (!Localmap_init_surf)
    {
        for (int i = 0; i < 3; i++)
        {
            LocalMap_Points_surf.vertex_min[i] = pos_LID(i) - cube_len / 2.0;
            LocalMap_Points_surf.vertex_max[i] = pos_LID(i) + cube_len / 2.0;
        }
        Localmap_init_surf = true;
        return;
    }
    float dist_to_map_edge[3][2];
    bool need_move = false;
    for (int i = 0; i < 3; i++)
    {
        dist_to_map_edge[i][0] = std::fabs(pos_LID(i) - LocalMap_Points_surf.vertex_min[i]);
        dist_to_map_edge[i][1] = std::fabs(pos_LID(i) - LocalMap_Points_surf.vertex_max[i]);
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE || dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
            need_move = true;
    }
    if (!need_move)
        return;
    BoxPointType New_LocalMap_Points, tmp_boxpoints;
    New_LocalMap_Points = LocalMap_Points_surf;
    float mov_dist = std::max((cube_len - 2.0 * MOV_THRESHOLD * DET_RANGE) * 0.5 * 0.9, double(DET_RANGE * (MOV_THRESHOLD - 1)));
    for (int i = 0; i < 3; i++)
    {
        tmp_boxpoints = LocalMap_Points_surf;
        if (dist_to_map_edge[i][0] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] -= mov_dist;
            New_LocalMap_Points.vertex_min[i] -= mov_dist;
            tmp_boxpoints.vertex_min[i] = LocalMap_Points_surf.vertex_max[i] - mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
        else if (dist_to_map_edge[i][1] <= MOV_THRESHOLD * DET_RANGE)
        {
            New_LocalMap_Points.vertex_max[i] += mov_dist;
            New_LocalMap_Points.vertex_min[i] += mov_dist;
            tmp_boxpoints.vertex_max[i] = LocalMap_Points_surf.vertex_min[i] + mov_dist;
            cub_needrm.push_back(tmp_boxpoints);
        }
    }
    LocalMap_Points_surf = New_LocalMap_Points;
    // points_cache_collect();
    PointVector points_history;
    ikdtree_surf.acquire_removed_points(points_history);
    if (cub_needrm.size() > 0)
        ikdtree_surf.Delete_Point_Boxes(cub_needrm);
}

void LIO::EstimateLidarPose(std::list<LidarFrame> &lidarFrameList,
                            const Eigen::Matrix4d &exTlb,
                            const Eigen::Vector3d &gravity,
                            nav_msgs::Odometry &debugInfo)
{
    Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3, 3).transpose();
    Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3, 1);
    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
    transformTobeMapped.topLeftCorner(3, 3) = lidarFrameList.back().Q * exRbl;
    transformTobeMapped.topRightCorner(3, 1) = lidarFrameList.back().Q * exPbl + lidarFrameList.back().P;

    Eigen::Vector3d pos_LID = transformTobeMapped.topRightCorner(3, 1);

    LasermapFovSegment_corner(pos_LID);
    LasermapFovSegment_surf(pos_LID);

    int stack_count = 0;
    ros::Time tr1 = ros::Time::now();
    for (const auto &l : lidarFrameList)
    {
        laserCloudCornerLast[stack_count]->clear();
        for (const auto &p : l.laserCloud->points)
        {
            if (std::fabs(p.normal_z - 1.0) < 1e-5)
                laserCloudCornerLast[stack_count]->push_back(p);
        }
        laserCloudSurfLast[stack_count]->clear();
        for (const auto &p : l.laserCloud->points)
        {
            if (std::fabs(p.normal_z - 2.0) < 1e-5)
                laserCloudSurfLast[stack_count]->push_back(p);
        }

        laserCloudCornerStack[stack_count]->clear();
        downSizeFilterCorner.setInputCloud(laserCloudCornerLast[stack_count]);
        downSizeFilterCorner.filter(*laserCloudCornerStack[stack_count]);

        laserCloudSurfStack[stack_count]->clear();
        downSizeFilterSurf.setInputCloud(laserCloudSurfLast[stack_count]);
        downSizeFilterSurf.filter(*laserCloudSurfStack[stack_count]);
    }
    ros::Time tr2 = ros::Time::now();
    std::cout << __FUNCTION__ << ", " << __LINE__
              << ": lidarlist:" << lidarFrameList.size()
              << "," << laserCloudCornerStack[0]->size() << ", " << laserCloudSurfStack[0]->size()
              << ",takes:" << (tr2 - tr1).toSec() << std::endl;

    int laserCloudCornerStackNum = laserCloudCornerStack[0]->points.size();
    int laserCloudSurfStackNum = laserCloudSurfStack[0]->points.size();

    if (ikdtree_surf.Root_Node == nullptr)
    {
        transformTobeMapped = Eigen::Matrix4d::Identity();
        transformTobeMapped.topLeftCorner(3, 3) = lidarFrameList.front().Q * exRbl;
        transformTobeMapped.topRightCorner(3, 1) = lidarFrameList.front().Q * exPbl + lidarFrameList.front().P;

        if (laserCloudSurfStack[0]->size() > 200 || laserCloudCornerStack[0]->size() > 50)
        {
            ikdtree_surf.set_downsample_param(filter_size_map_min);
            ikdtree_corner.set_downsample_param(filter_size_map_min);

            PointType pointSel;
            PointType pointSel2;
            feats_surf_world->clear();
            feats_corner_world->clear();
            for (int i = 0; i < laserCloudCornerStackNum; i++)
            {
                MAP_MANAGER::pointAssociateToMap(&laserCloudCornerStack[0]->points[i], &pointSel, transformTobeMapped);
                feats_corner_world->push_back(pointSel);
            }
            for (int i = 0; i < laserCloudSurfStackNum; i++)
            {
                MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack[0]->points[i], &pointSel2, transformTobeMapped);
                feats_surf_world->push_back(pointSel2);
            }
            ikdtree_surf.Build(feats_surf_world->points);
            ikdtree_corner.Build(feats_corner_world->points);

            std::cout << "feature points:" << laserCloudCornerStackNum << ", " << laserCloudSurfStackNum
                      << ",ikd map num: " << ikdtree_corner.validnum() << ", " << ikdtree_surf.validnum()
                      << ",ikd size: " << ikdtree_corner.size() << ", " << ikdtree_surf.size() << std::endl;
        }
        return;
    }

    {
        Nearest_Points_corner.resize(laserCloudCornerStackNum);
        Nearest_Points_surf.resize(laserCloudSurfStackNum);

        std::cout << "[mapping]: ikd-corner:" << ikdtree_corner.validnum() << "/" << ikdtree_corner.size()
                  << ", ikd-surf: " << ikdtree_surf.validnum() << "/" << ikdtree_surf.size() << std::endl;
        ros::Time a = ros::Time::now();
        Estimate(lidarFrameList, exTlb, gravity);
        flag_inited = true;
        ros::Time b = ros::Time::now();
        std::cout << "est takes : " << (b - a).toSec() << std::endl;
    }

    transformTobeMapped = Eigen::Matrix4d::Identity();
    transformTobeMapped.topLeftCorner(3, 3) = lidarFrameList.front().Q * exRbl;
    transformTobeMapped.topRightCorner(3, 1) = lidarFrameList.front().Q * exPbl + lidarFrameList.front().P;

    std::unique_lock<std::mutex> locker(mtx_Map);
    *laserCloudCornerForMap = *laserCloudCornerStack[0];
    *laserCloudSurfForMap = *laserCloudSurfStack[0];

    std::cout << "start mapincrement" << std::endl;
    ros::Time mi1 = ros::Time::now();
    MapIncrementLocal(laserCloudCornerForMap, laserCloudSurfForMap, transformTobeMapped);
    ros::Time mi2 = ros::Time::now();
    std::cout << "map incre takes: " << (mi2 - mi1).toSec() << std::endl;
    locker.unlock();
}

void LIO::Estimate(std::list<LidarFrame> &lidarFrameList,
                   const Eigen::Matrix4d &exTlb,
                   const Eigen::Vector3d &gravity)
{

    int num_corner_map = 0;
    int num_surf_map = 0;

    static uint32_t frame_count = 0;
    int windowSize = lidarFrameList.size();
    Eigen::Matrix4d transformTobeMapped = Eigen::Matrix4d::Identity();
    Eigen::Matrix3d exRbl = exTlb.topLeftCorner(3, 3).transpose();
    Eigen::Vector3d exPbl = -1.0 * exRbl * exTlb.topRightCorner(3, 1);

    // store point to line features
    std::vector<std::vector<FeatureLine>> vLineFeatures(windowSize);
    for (auto &v : vLineFeatures)
    {
        v.reserve(2000);
    }
    // store point to plan features
    std::vector<std::vector<FeaturePlanVec>> vPlanFeatures(windowSize);
    for (auto &v : vPlanFeatures)
    {
        v.reserve(2000);
    }

    if (windowSize == SLIDEWINDOWSIZE)
    {
        plan_weight_tan = 0.0003;
        thres_dist = 1.0;
    }
    else
    {
        plan_weight_tan = 0.0;
        thres_dist = 25.0;
    }

    // excute optimize process
    const int max_iters = 5;
    for (int iterOpt = 0; iterOpt < max_iters; ++iterOpt)
    {
        vector2double(lidarFrameList);

        // create huber loss function
        ceres::LossFunction *loss_function = NULL;
        loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
        ceres::LocalParameterization *quatParameterization = new ceres::QuaternionParameterization();
        if (windowSize == SLIDEWINDOWSIZE)
        {
            loss_function = NULL;
        }
        else
        {
            loss_function = new ceres::HuberLoss(0.1 / IMUIntegrator::lidar_m);
        }

        ceres::Problem::Options problem_options;
        ceres::Problem problem(problem_options);

        for (int i = 0; i < windowSize; ++i)
        {
            problem.AddParameterBlock(para_PR[i], 6);
        }

        for (int i = 0; i < windowSize; ++i)
            problem.AddParameterBlock(para_VBias[i], 9);

        // add IMU CostFunction
        for (int f = 1; f < windowSize; ++f)
        {
            auto frame_curr = lidarFrameList.begin();
            std::advance(frame_curr, f);
            problem.AddResidualBlock(Cost_NavState_PRV_Bias::Create(frame_curr->imuIntegrator,
                                                                    const_cast<Eigen::Vector3d &>(gravity),
                                                                    Eigen::LLT<Eigen::Matrix<double, 15, 15>>(frame_curr->imuIntegrator.GetCovariance().inverse())
                                                                        .matrixL()
                                                                        .transpose()),
                                     nullptr,
                                     para_PR[f - 1],
                                     para_VBias[f - 1],
                                     para_PR[f],
                                     para_VBias[f]);
        }

        Eigen::Quaterniond q_before_opti = lidarFrameList.back().Q;
        Eigen::Vector3d t_before_opti = lidarFrameList.back().P;

        ros::Time accf1 = ros::Time::now();

        std::vector<std::vector<ceres::CostFunction *>> edgesLine(windowSize);
        std::vector<std::vector<ceres::CostFunction *>> edgesPlan(windowSize);
        std::thread threads[2];
        for (int f = 0; f < windowSize; ++f)
        {
            auto frame_curr = lidarFrameList.begin();
            std::advance(frame_curr, f);
            transformTobeMapped = Eigen::Matrix4d::Identity();
            transformTobeMapped.topLeftCorner(3, 3) = frame_curr->Q * exRbl;
            transformTobeMapped.topRightCorner(3, 1) = frame_curr->Q * exPbl + frame_curr->P;

            threads[0] = std::thread(&LIO::processPointToLine, this,
                                     std::ref(edgesLine[f]),
                                     std::ref(vLineFeatures[f]),
                                     std::ref(laserCloudCornerStack[f]),
                                     std::ref(exTlb),
                                     std::ref(transformTobeMapped));

            threads[1] = std::thread(&LIO::processPointToPlanVec, this,
                                     std::ref(edgesPlan[f]),
                                     std::ref(vPlanFeatures[f]),
                                     std::ref(laserCloudSurfStack[f]),
                                     std::ref(exTlb),
                                     std::ref(transformTobeMapped));

            threads[0].join();
            threads[1].join();
        }
        ros::Time accf2 = ros::Time::now();
        std::cout << "acc feature takes: " << (accf2 - accf1).toSec() << std::endl;

        int cntSurf = 0, inlineSurf = 0;
        int cntCorner = 0, inlineCorner = 0;
        if (windowSize == SLIDEWINDOWSIZE)
        {
            thres_dist = 1.0;

            if (iterOpt == 0)
            {
                for (int f = 0; f < windowSize; ++f)
                {
                    int cntFtu = 0;
                    for (auto &e : edgesLine[f])
                    {
                        if (std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5)
                        {
                            problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            vLineFeatures[f][cntFtu].valid = true;
                            inlineCorner++;
                        }
                        else
                        {
                            vLineFeatures[f][cntFtu].valid = false;
                        }
                        cntFtu++;
                        cntCorner++;
                    }

                    cntFtu = 0;
                    for (auto &e : edgesPlan[f])
                    {
                        if (std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5)
                        {
                            problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            vPlanFeatures[f][cntFtu].valid = true;
                            inlineSurf++;
                        }
                        else
                        {
                            vPlanFeatures[f][cntFtu].valid = false;
                        }
                        cntFtu++;
                        cntSurf++;
                    }
                }
            }
            else
            {
                for (int f = 0; f < windowSize; ++f)
                {
                    int cntFtu = 0;
                    for (auto &e : edgesLine[f])
                    {
                        if (vLineFeatures[f][cntFtu].valid)
                        {
                            problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            inlineCorner++;
                        }
                        cntFtu++;
                        cntCorner++;
                    }
                    cntFtu = 0;
                    for (auto &e : edgesPlan[f])
                    {
                        if (vPlanFeatures[f][cntFtu].valid)
                        {
                            problem.AddResidualBlock(e, loss_function, para_PR[f]);
                            inlineSurf++;
                        }
                        cntFtu++;
                        cntSurf++;
                    }
                }
            }
        }
        else
        {
            if (iterOpt == 0)
            {
                thres_dist = 10.0;
            }
            else
            {
                thres_dist = 1.0;
            }

            for (int f = 0; f < windowSize; ++f)
            {
                int cntFtu = 0;
                for (auto &e : edgesLine[f])
                {
                    if (std::fabs(vLineFeatures[f][cntFtu].error) > 1e-5)
                    {
                        problem.AddResidualBlock(e, loss_function, para_PR[f]);
                        vLineFeatures[f][cntFtu].valid = true;
                        inlineCorner++;
                    }
                    else
                    {
                        vLineFeatures[f][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntCorner++;
                }
                cntFtu = 0;
                for (auto &e : edgesPlan[f])
                {
                    if (std::fabs(vPlanFeatures[f][cntFtu].error) > 1e-5)
                    {
                        problem.AddResidualBlock(e, loss_function, para_PR[f]);
                        vPlanFeatures[f][cntFtu].valid = true;
                        inlineSurf++;
                    }
                    else
                    {
                        vPlanFeatures[f][cntFtu].valid = false;
                    }
                    cntFtu++;
                    cntSurf++;
                }
            }
        }

        std::cout << "add featrue: " << inlineCorner << "/" << cntCorner
                  << " surf: " << inlineSurf << "/" << cntSurf << std::endl;

        ros::Time solve1 = ros::Time::now();
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_SCHUR;
        options.trust_region_strategy_type = ceres::DOGLEG;
        options.max_num_iterations = 10;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = 6;
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        ros::Time solve2 = ros::Time::now();
        std::cout << "solve featrue takes: " << (solve2 - solve1).toSec() << std::endl;

        std::cout << summary.BriefReport() << std::endl;

        double2vector(lidarFrameList);

        Eigen::Quaterniond q_after_opti = lidarFrameList.back().Q;
        Eigen::Vector3d t_after_opti = lidarFrameList.back().P;
        Eigen::Vector3d V_after_opti = lidarFrameList.back().V;
        double deltaR = (q_before_opti.angularDistance(q_after_opti)) * 180.0 / M_PI;
        double deltaT = (t_before_opti - t_after_opti).norm();

        if (deltaR < 0.05 && deltaT < 0.05 || (iterOpt + 1) == max_iters)
        // if (0)
        {
            ROS_INFO("Frame: %d\n", frame_count++);
            if (windowSize != SLIDEWINDOWSIZE)
                break;

            break;
        }

        if (windowSize != SLIDEWINDOWSIZE)
        {
            for (int f = 0; f < windowSize; ++f)
            {
                edgesLine[f].clear();
                edgesPlan[f].clear();
                vLineFeatures[f].clear();
                vPlanFeatures[f].clear();
            }
        }
    }
}
void LIO::MapIncrementLocal(const pcl::PointCloud<PointType>::Ptr &laserCloudCornerStack,
                            const pcl::PointCloud<PointType>::Ptr &laserCloudSurfStack,
                            const Eigen::Matrix4d &transformTobeMapped)
{
    int laserCloudCornerStackNum = laserCloudCornerStack->points.size();
    int laserCloudSurfStackNum = laserCloudSurfStack->points.size();
    PointVector PointToAdd_surf, PointToAdd_corner;
    PointVector PointNoNeedDownsample_surf, PointNoNeedDownsample_corner;
    PointToAdd_surf.reserve(laserCloudSurfStackNum), PointToAdd_corner.reserve(laserCloudCornerStackNum);
    PointNoNeedDownsample_surf.reserve(laserCloudSurfStackNum), PointNoNeedDownsample_corner.reserve(laserCloudCornerStackNum);

    PointType pointSel, pointSel2;
    for (int i = 0; i < laserCloudCornerStackNum; i++)
    {
        MAP_MANAGER::pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel, transformTobeMapped);
        if (!Nearest_Points_corner[i].empty() && flag_inited)
        {
            const PointVector &points_near = Nearest_Points_corner[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(pointSel.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(pointSel.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(pointSel.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(pointSel, mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample_corner.push_back(pointSel);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd_corner.push_back(pointSel);
        }
        else
            PointToAdd_corner.push_back(pointSel);
    }

    int add_point_size = ikdtree_corner.Add_Points(PointToAdd_corner, true);
    int add_point_nds = ikdtree_corner.Add_Points(PointNoNeedDownsample_corner, false);

    for (int i = 0; i < laserCloudSurfStackNum; i++)
    {
        MAP_MANAGER::pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel2, transformTobeMapped);
        if (!Nearest_Points_surf[i].empty() && flag_inited)
        {
            const PointVector &points_near = Nearest_Points_surf[i];
            bool need_add = true;
            BoxPointType Box_of_Point;
            PointType downsample_result, mid_point;
            mid_point.x = floor(pointSel2.x / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.y = floor(pointSel2.y / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            mid_point.z = floor(pointSel2.z / filter_size_map_min) * filter_size_map_min + 0.5 * filter_size_map_min;
            float dist = calc_dist(pointSel2, mid_point);
            if (fabs(points_near[0].x - mid_point.x) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].y - mid_point.y) > 0.5 * filter_size_map_min &&
                fabs(points_near[0].z - mid_point.z) > 0.5 * filter_size_map_min)
            {
                PointNoNeedDownsample_surf.push_back(pointSel2);
                continue;
            }
            for (int readd_i = 0; readd_i < NUM_MATCH_POINTS; readd_i++)
            {
                if (points_near.size() < NUM_MATCH_POINTS)
                    break;
                if (calc_dist(points_near[readd_i], mid_point) < dist)
                {
                    need_add = false;
                    break;
                }
            }
            if (need_add)
                PointToAdd_surf.push_back(pointSel2);
        }
        else
            PointToAdd_surf.push_back(pointSel2);
    }

    add_point_size = ikdtree_surf.Add_Points(PointToAdd_surf, true);
    add_point_nds = ikdtree_surf.Add_Points(PointNoNeedDownsample_surf, false);
}