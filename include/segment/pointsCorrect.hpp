#ifndef _COMMON_HPP
#define _CONNON_HPP

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Core>
#include <pcl/common/transforms.h>
#include <pcl/features/normal_3d.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/common/common.h>
#include <Eigen/Dense>

#include <vector>

using namespace std;

typedef struct
{
    Eigen::Matrix3f eigenVectorsPCA;
    Eigen::Vector3f eigenValuesPCA;
    std::vector<int> neibors;
} SNeiborPCA_cor;

int GetNeiborPCA_cor(SNeiborPCA_cor &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius);
int FilterGndForPos_cor(float* outPoints,float*inPoints,int inNum);
int CalGndPos_cor(float *gnd, float *fPoints,int pointNum,float fSearchRadius);
int GetRTMatrix_cor(float *RTM, float *v0, float *v1);
int CorrectPoints_cor(float *fPoints,int pointNum,float *gndPos);
int GetGndPos(float *pos, float *fPoints,int pointNum);
#endif
