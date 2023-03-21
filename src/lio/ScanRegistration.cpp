#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

typedef pcl::PointXYZINormal PointType;

ros::Publisher pubFullLaserCloud;
ros::Publisher pubSharpCloud;
ros::Publisher pubFlatCloud;
ros::Publisher pubNonFeature;

LidarFeatureExtractor* lidarFeatureExtractor;
pcl::PointCloud<PointType>::Ptr laserCloud;
pcl::PointCloud<PointType>::Ptr laserConerCloud;
pcl::PointCloud<PointType>::Ptr laserSurfCloud;
pcl::PointCloud<PointType>::Ptr laserNonFeatureCloud;
int Lidar_Type = 0;
int N_SCANS = 6;
bool Feature_Mode = false;
bool Use_seg = false;

void lidarCallBackHorizon(const livox_ros_driver::CustomMsgConstPtr &msg) {

  sensor_msgs::PointCloud2 msg2;

  if(Use_seg){
    lidarFeatureExtractor->FeatureExtract_with_segment(msg, laserCloud, laserConerCloud, laserSurfCloud, laserNonFeatureCloud, msg2,N_SCANS);
  }
  else{
    lidarFeatureExtractor->FeatureExtract(msg, laserCloud, laserConerCloud, laserSurfCloud,N_SCANS,Lidar_Type);
  } 

  sensor_msgs::PointCloud2 laserCloudMsg;
  pcl::toROSMsg(*laserCloud, laserCloudMsg);
  laserCloudMsg.header = msg->header;
  laserCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
  pubFullLaserCloud.publish(laserCloudMsg);

}

void lidarCallBackHAP(const livox_ros_driver::CustomMsgConstPtr &msg) {

  sensor_msgs::PointCloud2 msg2;

  if(Use_seg){
    lidarFeatureExtractor->FeatureExtract_with_segment_hap(msg, laserCloud, laserConerCloud, laserSurfCloud, laserNonFeatureCloud, msg2,N_SCANS);
  }
  else{
    lidarFeatureExtractor->FeatureExtract_hap(msg, laserCloud, laserConerCloud, laserSurfCloud, laserNonFeatureCloud, N_SCANS);
  } 

  sensor_msgs::PointCloud2 laserCloudMsg;
  pcl::toROSMsg(*laserCloud, laserCloudMsg);
  laserCloudMsg.header = msg->header;
  laserCloudMsg.header.stamp.fromNSec(msg->timebase+msg->points.back().offset_time);
  pubFullLaserCloud.publish(laserCloudMsg);

}

void lidarCallBackPc2(const sensor_msgs::PointCloud2ConstPtr &msg) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr laser_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::PointCloud<pcl::PointXYZINormal>::Ptr laser_cloud_custom(new pcl::PointCloud<pcl::PointXYZINormal>());

    pcl::fromROSMsg(*msg, *laser_cloud);

    for (uint64_t i = 0; i < laser_cloud->points.size(); i++)
    {
        auto p=laser_cloud->points.at(i);
        pcl::PointXYZINormal p_custom;
        if(Lidar_Type == 0||Lidar_Type == 1)
        {
            if(p.x < 0.01) continue;
        }
        else if(Lidar_Type == 2)
        {
            if(std::fabs(p.x) < 0.01) continue;
        }
        p_custom.x=p.x;
        p_custom.y=p.y;
        p_custom.z=p.z;
        p_custom.intensity=p.intensity;
        p_custom.normal_x=float (i)/float(laser_cloud->points.size());
        p_custom.normal_y=i%4;
        laser_cloud_custom->points.push_back(p_custom);
    }

    lidarFeatureExtractor->FeatureExtract_Mid(laser_cloud_custom, laserConerCloud, laserSurfCloud);

    sensor_msgs::PointCloud2 laserCloudMsg;
    pcl::toROSMsg(*laser_cloud_custom, laserCloudMsg);
    laserCloudMsg.header = msg->header;
    pubFullLaserCloud.publish(laserCloudMsg);

}

int main(int argc, char** argv)
{
  ros::init(argc, argv, "ScanRegistration");
  ros::NodeHandle nodeHandler("~");

  ros::Subscriber customCloud,pc2Cloud;

  std::string config_file;
  int msg_type=0;
  nodeHandler.getParam("config_file", config_file);
  nodeHandler.getParam("msg_type", msg_type);

  cv::FileStorage fsSettings(config_file, cv::FileStorage::READ);
  if (!fsSettings.isOpened()) {
    std::cout << "config_file error: cannot open " << config_file << std::endl;
    return false;
  }
  Lidar_Type = static_cast<int>(fsSettings["Lidar_Type"]);
  N_SCANS = static_cast<int>(fsSettings["Used_Line"]);
  Feature_Mode = static_cast<int>(fsSettings["Feature_Mode"]);
  Use_seg = static_cast<int>(fsSettings["Use_seg"]);

  int NumCurvSize = static_cast<int>(fsSettings["NumCurvSize"]);
  float DistanceFaraway = static_cast<float>(fsSettings["DistanceFaraway"]);
  int NumFlat = static_cast<int>(fsSettings["NumFlat"]);
  int PartNum = static_cast<int>(fsSettings["PartNum"]);
  float FlatThreshold = static_cast<float>(fsSettings["FlatThreshold"]);
  float BreakCornerDis = static_cast<float>(fsSettings["BreakCornerDis"]);
  float LidarNearestDis = static_cast<float>(fsSettings["LidarNearestDis"]);
  float KdTreeCornerOutlierDis = static_cast<float>(fsSettings["KdTreeCornerOutlierDis"]);


  laserCloud.reset(new pcl::PointCloud<PointType>);
  laserConerCloud.reset(new pcl::PointCloud<PointType>);
  laserSurfCloud.reset(new pcl::PointCloud<PointType>);
  laserNonFeatureCloud.reset(new pcl::PointCloud<PointType>);

  if (Lidar_Type == 0)
  {
    customCloud = nodeHandler.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, &lidarCallBackHorizon);
  }
  else if (Lidar_Type == 1)
  {
    customCloud = nodeHandler.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, &lidarCallBackHAP);
  }
  else if(Lidar_Type==2){
      if (msg_type==0)
          customCloud = nodeHandler.subscribe<livox_ros_driver::CustomMsg>("/livox/lidar", 100, &lidarCallBackHorizon);
      else if(msg_type==1)
          pc2Cloud=nodeHandler.subscribe<sensor_msgs::PointCloud2>("/livox/lidar", 100, &lidarCallBackPc2);
  }
  pubFullLaserCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_full_cloud", 10);
  pubSharpCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_less_sharp_cloud", 10);
  pubFlatCloud = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_less_flat_cloud", 10);
  pubNonFeature = nodeHandler.advertise<sensor_msgs::PointCloud2>("/livox_nonfeature_cloud", 10);

  lidarFeatureExtractor = new LidarFeatureExtractor(N_SCANS,NumCurvSize,DistanceFaraway,NumFlat,PartNum,
                                                    FlatThreshold,BreakCornerDis,LidarNearestDis,KdTreeCornerOutlierDis);

  ros::spin();

  return 0;
}

