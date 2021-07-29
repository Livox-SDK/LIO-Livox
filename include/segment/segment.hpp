#ifndef _SEGMENT_HPP
#define _SEGMENT_HPP

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

#include "pointsCorrect.hpp"

using namespace std;

#define SELF_CALI_FRAMES 20

#define GND_IMG_NX 150
#define GND_IMG_NY 400
#define GND_IMG_DX 0.2
#define GND_IMG_DY 0.2
#define GND_IMG_OFFX 40
#define GND_IMG_OFFY 40

#define GND_IMG_NX1 24
#define GND_IMG_NY1 20
#define GND_IMG_DX1 4
#define GND_IMG_DY1 4
#define GND_IMG_OFFX1 40
#define GND_IMG_OFFY1 40

#define DN_SAMPLE_IMG_NX 600 //(GND_IMG_NX)
#define DN_SAMPLE_IMG_NY 200 //(GND_IMG_NY)
#define DN_SAMPLE_IMG_NZ 100
#define DN_SAMPLE_IMG_DX 0.4 //(GND_IMG_DX)
#define DN_SAMPLE_IMG_DY 0.4 //(GND_IMG_DY)
#define DN_SAMPLE_IMG_DZ 0.2
#define DN_SAMPLE_IMG_OFFX 40 //(GND_IMG_OFFX)
#define DN_SAMPLE_IMG_OFFY 40 //(GND_IMG_OFFY)
#define DN_SAMPLE_IMG_OFFZ 2.5//2.5

#define FREE_ANG_NUM 360
#define FREE_PI (3.14159265)
#define FREE_DELTA_ANG (FREE_PI*2/FREE_ANG_NUM)

typedef struct
{
    Eigen::Matrix3f eigenVectorsPCA;
    Eigen::Vector3f eigenValuesPCA;
    std::vector<int> neibors;
} SNeiborPCA;

typedef struct
{
    // basic
    int pnum;
    float xmin;
    float xmax;
    float ymin;
    float ymax;
    float zmin;
    float zmax;
    float zmean;

    // pca
    float d0[3];
    float d1[3];

    float center[3];

    float obb[8];

    int cls;//类别

} SClusterFeature;

int FilterGndForPos(float* outPoints,float*inPoints,int inNum);
int CalNomarls(float *nvects, float *fPoints,int pointNum,float fSearchRadius);

int CalGndPos(float *gnd, float *fPoints,int pointNum,float fSearchRadius);

int GetNeiborPCA(SNeiborPCA &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud,
                    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius);


int CorrectPoints(float *fPoints,int pointNum,float *gndPos);

// 地面上物体分割
int AbvGndSeg(int *pLabel, float *fPoints, int pointNum);
int SegBG(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);

SClusterFeature FindACluster(int *pLabel, int seedId, int labelId, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius, float thrHeight);
int SegObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);

int CalFreeRegion(float *pFreeDis, float *fPoints,int *pLabel,int pointNum);
int FreeSeg(float *fPoints,int *pLabel,int pointNum);

int CompleteObjects(int *pLabel, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> &kdtree, float fSearchRadius);
int ExpandObjects(int *pLabel, float* fPoints, int pointNum, float fSearchRadius);
int ExpandBG(int *pLabel, float* fPoints, int pointNum, float fSearchRadius);

// 地面分割
int GndSeg(int* pLabel,float *fPoints,int pointNum,float fSearchRadius);


class PCSeg
{
    public:
    PCSeg();
    // functions
    int DoSeg(int *pLabel, float* fPoints1, int pointNum);
    int GetMainVectors(float*fPoints, int* pLabel, int pointNum);
    int EncodeFeatures(float *pFeas);
    int DoBoundaryDetection(float* fPoints1,int *pLabel1,int pointNum);
    int DoTrafficLineDet(float *fPoints1,int *pLabel1,int pointNum);

    int CorrectPoints(float *fPoints,int pointNum,float *gndPos);

    float *PrePoints;
    int numPrePoints = 0;

    float gnd_pos[100*6];
    int gnd_vet_len = 0;

    int laneType=0;
    float lanePosition[2] = {0};

    // vars
    unsigned char *pVImg;
    float gndPos[6];
    int posFlag;

    // cluster features
    float pVectors[256*3];
    float pCenters[256*3];//重心
    int pnum[256];

    int objClass[256];
    float zs[256];
    float pOBBs[256*8];

    float CBBox[256*7];//（a,x,y,z,l,w,h）

    int clusterNum;
    
    float *corPoints;
    int corNum;

    ~PCSeg();

};

SClusterFeature CalBBox(float *fPoints,int pointNum);
SClusterFeature CalOBB(float *fPoints,int pointNum);

#endif
