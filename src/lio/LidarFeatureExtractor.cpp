#include "LidarFeatureExtractor/LidarFeatureExtractor.h"

LidarFeatureExtractor::LidarFeatureExtractor(int n_scans,int NumCurvSize,float DistanceFaraway,int NumFlat,
                                             int PartNum,float FlatThreshold,float BreakCornerDis,float LidarNearestDis,float KdTreeCornerOutlierDis)
                                             :N_SCANS(n_scans),
                                              thNumCurvSize(NumCurvSize),
                                              thDistanceFaraway(DistanceFaraway),
                                              thNumFlat(NumFlat),
                                              thPartNum(PartNum),
                                              thFlatThreshold(FlatThreshold),
                                              thBreakCornerDis(BreakCornerDis),
                                              thLidarNearestDis(LidarNearestDis){
  vlines.resize(N_SCANS);
  for(auto & ptr : vlines){
    ptr.reset(new pcl::PointCloud<PointType>());
  }
  vcorner.resize(N_SCANS);
  vsurf.resize(N_SCANS);
}

bool LidarFeatureExtractor::plane_judge(const std::vector<PointType>& point_list,const int plane_threshold)
{
  int num = point_list.size();
  float cx = 0;
  float cy = 0;
  float cz = 0;
  for (int j = 0; j < num; j++) {
    cx += point_list[j].x;
    cy += point_list[j].y;
    cz += point_list[j].z;
  }
  cx /= num;
  cy /= num;
  cz /= num;
  //mean square error
  float a11 = 0;
  float a12 = 0;
  float a13 = 0;
  float a22 = 0;
  float a23 = 0;
  float a33 = 0;
  for (int j = 0; j < num; j++) {
    float ax = point_list[j].x - cx;
    float ay = point_list[j].y - cy;
    float az = point_list[j].z - cz;

    a11 += ax * ax;
    a12 += ax * ay;
    a13 += ax * az;
    a22 += ay * ay;
    a23 += ay * az;
    a33 += az * az;
  }
  a11 /= num;
  a12 /= num;
  a13 /= num;
  a22 /= num;
  a23 /= num;
  a33 /= num;

  Eigen::Matrix< double, 3, 3 > _matA1;
  _matA1.setZero();
  Eigen::Matrix< double, 3, 1 > _matD1;
  _matD1.setZero();
  Eigen::Matrix< double, 3, 3 > _matV1;
  _matV1.setZero();

  _matA1(0, 0) = a11;
  _matA1(0, 1) = a12;
  _matA1(0, 2) = a13;
  _matA1(1, 0) = a12;
  _matA1(1, 1) = a22;
  _matA1(1, 2) = a23;
  _matA1(2, 0) = a13;
  _matA1(2, 1) = a23;
  _matA1(2, 2) = a33;

  Eigen::JacobiSVD<Eigen::Matrix3d> svd(_matA1, Eigen::ComputeThinU | Eigen::ComputeThinV);
  _matD1 = svd.singularValues();
  _matV1 = svd.matrixU();
  if (_matD1(0, 0) < plane_threshold * _matD1(1, 0)) {
    return true;
  }
  else{
    return false;
  }
}

void LidarFeatureExtractor::detectFeaturePoint(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp,
                                                std::vector<int>& pointsLessFlat){
  int CloudFeatureFlag[20000];
  float cloudCurvature[20000];
  float cloudDepth[20000];
  int cloudSortInd[20000];
  float cloudReflect[20000];
  int reflectSortInd[20000];
  int cloudAngle[20000];

  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;

  int cloudSize = laserCloudIn->points.size();

  PointType point;
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
#ifdef UNDISTORT
    point.normal_x = laserCloudIn.points[i].normal_x;
#else
    point.normal_x = 1.0;
#endif
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }

  cloudSize = _laserCloud->size();

  int debugnum1 = 0;
  int debugnum2 = 0;
  int debugnum3 = 0;
  int debugnum4 = 0;
  int debugnum5 = 0;

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //---------------------------------------- surf feature extract ---------------------------------------------
  int scanStartInd = 5;
  int scanEndInd = cloudSize - 6;

  int thDistanceFaraway_fea = 0;

  for (int i = 5; i < cloudSize - 5; i ++ ) {

    float diffX = 0;
    float diffY = 0;
    float diffZ = 0;

    float dis = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                     _laserCloud->points[i].y * _laserCloud->points[i].y +
                     _laserCloud->points[i].z * _laserCloud->points[i].z);

    Eigen::Vector3d pt_last(_laserCloud->points[i-1].x, _laserCloud->points[i-1].y, _laserCloud->points[i-1].z);
    Eigen::Vector3d pt_cur(_laserCloud->points[i].x, _laserCloud->points[i].y, _laserCloud->points[i].z);
    Eigen::Vector3d pt_next(_laserCloud->points[i+1].x, _laserCloud->points[i+1].y, _laserCloud->points[i+1].z);

    double angle_last = (pt_last-pt_cur).dot(pt_cur) / ((pt_last-pt_cur).norm()*pt_cur.norm());
    double angle_next = (pt_next-pt_cur).dot(pt_cur) / ((pt_next-pt_cur).norm()*pt_cur.norm());
 
    if (dis > thDistanceFaraway || (fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966)) {
      thNumCurvSize = 2;
    } else {
      thNumCurvSize = 3;
    }

    if(fabs(angle_last) > 0.966 && fabs(angle_next) > 0.966) {
      cloudAngle[i] = 1;
    }

    float diffR = -2 * thNumCurvSize * _laserCloud->points[i].intensity;
    for (int j = 1; j <= thNumCurvSize; ++j) {
      diffX += _laserCloud->points[i - j].x + _laserCloud->points[i + j].x;
      diffY += _laserCloud->points[i - j].y + _laserCloud->points[i + j].y;
      diffZ += _laserCloud->points[i - j].z + _laserCloud->points[i + j].z;
      diffR += _laserCloud->points[i - j].intensity + _laserCloud->points[i + j].intensity;
    }
    diffX -= 2 * thNumCurvSize * _laserCloud->points[i].x;
    diffY -= 2 * thNumCurvSize * _laserCloud->points[i].y;
    diffZ -= 2 * thNumCurvSize * _laserCloud->points[i].z;

    cloudDepth[i] = dis;
    cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;// / (2 * thNumCurvSize * dis + 1e-3);
    cloudSortInd[i] = i;
    cloudReflect[i] = diffR;
    reflectSortInd[i] = i;

  }

  for (int j = 0; j < thPartNum; j++) {
    int sp = scanStartInd + (scanEndInd - scanStartInd) * j / thPartNum;
    int ep = scanStartInd + (scanEndInd - scanStartInd) * (j + 1) / thPartNum - 1;

    // sort the curvatures from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudCurvature[cloudSortInd[l]] <
            cloudCurvature[cloudSortInd[l - 1]]) {
          int temp = cloudSortInd[l - 1];
          cloudSortInd[l - 1] = cloudSortInd[l];
          cloudSortInd[l] = temp;
        }
      }
    }

    // sort the reflectivity from small to large
    for (int k = sp + 1; k <= ep; k++) {
      for (int l = k; l >= sp + 1; l--) {
        if (cloudReflect[reflectSortInd[l]] <
            cloudReflect[reflectSortInd[l - 1]]) {
          int temp = reflectSortInd[l - 1];
          reflectSortInd[l - 1] = reflectSortInd[l];
          reflectSortInd[l] = temp;
        }
      }
    }

    int smallestPickedNum = 1;
    int sharpestPickedNum = 1;
    for (int k = sp; k <= ep; k++) {
      int ind = cloudSortInd[k];

      if (CloudFeatureFlag[ind] != 0) continue;

      if (cloudCurvature[ind] < thFlatThreshold * cloudDepth[ind] * thFlatThreshold * cloudDepth[ind]) {
        
        CloudFeatureFlag[ind] = 3;

        for (int l = 1; l <= thNumCurvSize; l++) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l - 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l - 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l - 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }

          CloudFeatureFlag[ind + l] = 1;
        }
        for (int l = -1; l >= -thNumCurvSize; l--) {
          float diffX = _laserCloud->points[ind + l].x -
                        _laserCloud->points[ind + l + 1].x;
          float diffY = _laserCloud->points[ind + l].y -
                        _laserCloud->points[ind + l + 1].y;
          float diffZ = _laserCloud->points[ind + l].z -
                        _laserCloud->points[ind + l + 1].z;
          if (diffX * diffX + diffY * diffY + diffZ * diffZ > 0.02 || cloudDepth[ind] > thDistanceFaraway) {
            break;
          }

          CloudFeatureFlag[ind + l] = 1;
        }
      }
    }
    
    for (int k = sp; k <= ep; k++) {
      int ind = cloudSortInd[k];
      if(((CloudFeatureFlag[ind] == 3) && (smallestPickedNum <= thNumFlat)) || 
          ((CloudFeatureFlag[ind] == 3) && (cloudDepth[ind] > thDistanceFaraway)) ||
          cloudAngle[ind] == 1){
        smallestPickedNum ++;
        CloudFeatureFlag[ind] = 2;
        if(cloudDepth[ind] > thDistanceFaraway) {
          thDistanceFaraway_fea++;
        }
      }

      int idx = reflectSortInd[k];
      if(cloudCurvature[idx] < 0.7 * thFlatThreshold * cloudDepth[idx] * thFlatThreshold * cloudDepth[idx]
         && sharpestPickedNum <= 3 && cloudReflect[idx] > 20.0){
        sharpestPickedNum ++;
        CloudFeatureFlag[idx] = 300;
      }
    }
    
  }

  //---------------------------------------- line feature where surfaces meet -------------------------------------
  for (int i = 5; i < cloudSize - 5; i += count_num ) {
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);
    //left curvature
    float ldiffX =
            _laserCloud->points[i - 4].x + _laserCloud->points[i - 3].x
            - 4 * _laserCloud->points[i - 2].x
            + _laserCloud->points[i - 1].x + _laserCloud->points[i].x;

    float ldiffY =
            _laserCloud->points[i - 4].y + _laserCloud->points[i - 3].y
            - 4 * _laserCloud->points[i - 2].y
            + _laserCloud->points[i - 1].y + _laserCloud->points[i].y;

    float ldiffZ =
            _laserCloud->points[i - 4].z + _laserCloud->points[i - 3].z
            - 4 * _laserCloud->points[i - 2].z
            + _laserCloud->points[i - 1].z + _laserCloud->points[i].z;

    float left_curvature = ldiffX * ldiffX + ldiffY * ldiffY + ldiffZ * ldiffZ;

    if(left_curvature < thFlatThreshold * depth){

      std::vector<PointType> left_list;

      for(int j = -4; j < 0; j++){
        left_list.push_back(_laserCloud->points[i + j]);
      }

      left_surf_flag = true;
    }
    else{
      left_surf_flag = false;
    }

    //right curvature
    float rdiffX =
            _laserCloud->points[i + 4].x + _laserCloud->points[i + 3].x
            - 4 * _laserCloud->points[i + 2].x
            + _laserCloud->points[i + 1].x + _laserCloud->points[i].x;

    float rdiffY =
            _laserCloud->points[i + 4].y + _laserCloud->points[i + 3].y
            - 4 * _laserCloud->points[i + 2].y
            + _laserCloud->points[i + 1].y + _laserCloud->points[i].y;

    float rdiffZ =
            _laserCloud->points[i + 4].z + _laserCloud->points[i + 3].z
            - 4 * _laserCloud->points[i + 2].z
            + _laserCloud->points[i + 1].z + _laserCloud->points[i].z;

    float right_curvature = rdiffX * rdiffX + rdiffY * rdiffY + rdiffZ * rdiffZ;

    if(right_curvature < thFlatThreshold * depth){
      std::vector<PointType> right_list;

      for(int j = 1; j < 5; j++){
        right_list.push_back(_laserCloud->points[i + j]);
      }
      count_num = 4;
      right_surf_flag = true;
    }
    else{
      count_num = 1;
      right_surf_flag = false;
    }

    //calculate the included angle
    if(left_surf_flag && right_surf_flag){
      debugnum4 ++;

      Eigen::Vector3d norm_left(0,0,0);
      Eigen::Vector3d norm_right(0,0,0);
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_left += (k/10.0)* tmp;
      }
      for(int k = 1;k<5;k++){
        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        norm_right += (k/10.0)* tmp;
      }

      //calculate the angle between this group and the previous group
      double cc = fabs( norm_left.dot(norm_right) / (norm_left.norm()*norm_right.norm()) );
      //calculate the maximum distance, the distance cannot be too small
      Eigen::Vector3d last_tmp = Eigen::Vector3d(_laserCloud->points[i - 4].x - _laserCloud->points[i].x,
                                                 _laserCloud->points[i - 4].y - _laserCloud->points[i].y,
                                                 _laserCloud->points[i - 4].z - _laserCloud->points[i].z);
      Eigen::Vector3d current_tmp = Eigen::Vector3d(_laserCloud->points[i + 4].x - _laserCloud->points[i].x,
                                                    _laserCloud->points[i + 4].y - _laserCloud->points[i].y,
                                                    _laserCloud->points[i + 4].z - _laserCloud->points[i].z);
      double last_dis = last_tmp.norm();
      double current_dis = current_tmp.norm();

      if(cc < 0.5 && last_dis > 0.05 && current_dis > 0.05 ){ //
        debugnum5 ++;
        CloudFeatureFlag[i] = 150;
      }
    }

  }

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);
    
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool left_is_plane = plane_judge(left_list,100);

        if( cc < 0.95 ){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        bool right_is_plane = plane_judge(right_list,100);

        if( cc < 0.95){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      debugnum2++;
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      if(cc < 0.95){
        debugnum3++;
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  //push_back feature

  for(int i = 5; i < cloudSize - 5; i ++){

    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x
            + _laserCloud->points[i].y * _laserCloud->points[i].y
            + _laserCloud->points[i].z * _laserCloud->points[i].z;

    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

    if(CloudFeatureFlag[i] == 2){
      pointsLessFlat.push_back(i);
      num_surf++;
      continue;
    }

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 150){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }

  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}

void LidarFeatureExtractor::FeatureExtract_with_segment(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                        pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                        pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                        pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                        sensor_msgs::PointCloud2 &msg_seg,
                                                        const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  int *idtrans = (int*)calloc(dnum, sizeof(int));
  float *data=(float*)calloc(dnum*4,sizeof(float));
  int point_num = 0;

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }

  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
    int line_idx = _float_as_int(laserCloud->points[i].normal_y);
    laserCloud->points[i].normal_z = _int_as_float(i);
    vlines[line_idx]->push_back(laserCloud->points[i]);
  }

  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
    threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint3, this, std::ref(vlines[i]),std::ref(vcorner[i]));
  }

  for(int i=0; i<N_SCANS; ++i){
    threads[i].join();
  }

  int num_corner = 0;
  for(int i=0; i<N_SCANS; ++i){
    for(int j=0; j<vcorner[i].size(); ++j){
      laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0; 
      num_corner++;
    }
  }

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}

void LidarFeatureExtractor::FeatureExtract_with_segment_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                                            pcl::PointCloud<PointType>::Ptr& laserCloud,
                                                            pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                                            pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                                            sensor_msgs::PointCloud2 &msg_seg,
                                                            const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  int *idtrans = (int*)calloc(dnum, sizeof(int));
  float *data=(float*)calloc(dnum*4,sizeof(float));
  int point_num = 0;

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);

    data[point_num*4+0] = point.x;
    data[point_num*4+1] = point.y;
    data[point_num*4+2] = point.z;
    data[point_num*4+3] = point.intensity;


    point_num++;
  }

  PCSeg pcseg;
  pcseg.DoSeg(idtrans,data,dnum);

  std::size_t cloud_num = laserCloud->size();

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  for(std::size_t i=0; i<cloud_num; ++i){
    float dis = laserCloud->points[i].x * laserCloud->points[i].x
                + laserCloud->points[i].y * laserCloud->points[i].y
                + laserCloud->points[i].z * laserCloud->points[i].z;
    if( idtrans[i] > 9 && dis < 50*50){
      laserCloud->points[i].normal_z = 0;
    }
  }

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }

}


void LidarFeatureExtractor::detectFeaturePoint2(pcl::PointCloud<PointType>::Ptr& cloud,
                                                pcl::PointCloud<PointType>::Ptr& pointsLessFlat,
                                                pcl::PointCloud<PointType>::Ptr& pointsNonFeature){

  int cloudSize = cloud->points.size();

  pointsLessFlat.reset(new pcl::PointCloud<PointType>());
  pointsNonFeature.reset(new pcl::PointCloud<PointType>());

  pcl::KdTreeFLANN<PointType>::Ptr KdTreeCloud;
  KdTreeCloud.reset(new pcl::KdTreeFLANN<PointType>);
  KdTreeCloud->setInputCloud(cloud);

  std::vector<int> _pointSearchInd;
  std::vector<float> _pointSearchSqDis;

  int num_near = 10;
  int stride = 1;
  int interval = 4;

  for(int i = 5; i < cloudSize - 5; i = i+stride) {
    if(fabs(cloud->points[i].normal_z - 1.0) < 1e-5) {
      continue;
    }

    double thre1d = 0.5;
    double thre2d = 0.8;
    double thre3d = 0.5;
    double thre3d2 = 0.13;

    double disti = sqrt(cloud->points[i].x * cloud->points[i].x + 
                        cloud->points[i].y * cloud->points[i].y + 
                        cloud->points[i].z * cloud->points[i].z);

    if(disti < 30.0) {
      thre1d = 0.5;
      thre2d = 0.8;
      thre3d2 = 0.07;
      stride = 14;
      interval = 4;
    } else if(disti < 60.0) {
      stride = 10;
      interval = 3;
    } else {
      stride = 1;
      interval = 0;
    }

    if(disti > 100.0) {
      num_near = 6;

      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
      continue;
    } else if(disti > 60.0) {
      num_near = 8;
    } else {
      num_near = 10;
    }

    KdTreeCloud->nearestKSearch(cloud->points[i], num_near, _pointSearchInd, _pointSearchSqDis);

    if (_pointSearchSqDis[num_near-1] > 5.0 && disti < 90.0) {
      continue;
    }

    Eigen::Matrix< double, 3, 3 > _matA1;
    _matA1.setZero();

    float cx = 0;
    float cy = 0;
    float cz = 0;
    for (int j = 0; j < num_near; j++) {
      cx += cloud->points[_pointSearchInd[j]].x;
      cy += cloud->points[_pointSearchInd[j]].y;
      cz += cloud->points[_pointSearchInd[j]].z;
    }
    cx /= num_near;
    cy /= num_near;
    cz /= num_near;

    float a11 = 0;
    float a12 = 0;
    float a13 = 0;
    float a22 = 0;
    float a23 = 0;
    float a33 = 0;
    for (int j = 0; j < num_near; j++) {
      float ax = cloud->points[_pointSearchInd[j]].x - cx;
      float ay = cloud->points[_pointSearchInd[j]].y - cy;
      float az = cloud->points[_pointSearchInd[j]].z - cz;

      a11 += ax * ax;
      a12 += ax * ay;
      a13 += ax * az;
      a22 += ay * ay;
      a23 += ay * az;
      a33 += az * az;
    }
    a11 /= num_near;
    a12 /= num_near;
    a13 /= num_near;
    a22 /= num_near;
    a23 /= num_near;
    a33 /= num_near;

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
    double a1d = (sqrt(saes.eigenvalues()[2]) - sqrt(saes.eigenvalues()[1])) / sqrt(saes.eigenvalues()[2]);
    double a2d = (sqrt(saes.eigenvalues()[1]) - sqrt(saes.eigenvalues()[0])) / sqrt(saes.eigenvalues()[2]);
    double a3d = sqrt(saes.eigenvalues()[0]) / sqrt(saes.eigenvalues()[2]);

    if(a2d > thre2d || (a3d < thre3d2 && a1d < thre1d)) {
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 2.0;
        pointsLessFlat->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 2.0;
      pointsLessFlat->points.push_back(cloud->points[i]);
    } else if(a3d > thre3d) {
      for(int k = 1; k < interval; k++) {
        cloud->points[i-k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i-k]);
        cloud->points[i+k].normal_z = 3.0;
        pointsNonFeature->points.push_back(cloud->points[i+k]);
      }
      cloud->points[i].normal_z = 3.0;
      pointsNonFeature->points.push_back(cloud->points[i]);
    }
  }  
}


void LidarFeatureExtractor::detectFeaturePoint3(pcl::PointCloud<PointType>::Ptr& cloud,
                                                std::vector<int>& pointsLessSharp){
  int CloudFeatureFlag[20000];
  float cloudCurvature[20000];
  float cloudDepth[20000];
  int cloudSortInd[20000];
  float cloudReflect[20000];
  int reflectSortInd[20000];
  int cloudAngle[20000];

  pcl::PointCloud<PointType>::Ptr& laserCloudIn = cloud;

  int cloudSize = laserCloudIn->points.size();

  PointType point;
  pcl::PointCloud<PointType>::Ptr _laserCloud(new pcl::PointCloud<PointType>());
  _laserCloud->reserve(cloudSize);

  for (int i = 0; i < cloudSize; i++) {
    point.x = laserCloudIn->points[i].x;
    point.y = laserCloudIn->points[i].y;
    point.z = laserCloudIn->points[i].z;
    point.normal_x = 1.0;
    point.intensity = laserCloudIn->points[i].intensity;

    if (!pcl_isfinite(point.x) ||
        !pcl_isfinite(point.y) ||
        !pcl_isfinite(point.z)) {
      continue;
    }

    _laserCloud->push_back(point);
    CloudFeatureFlag[i] = 0;
  }

  cloudSize = _laserCloud->size();

  int count_num = 1;
  bool left_surf_flag = false;
  bool right_surf_flag = false;

  //--------------------------------------------------- break points ---------------------------------------------
  for(int i = 5; i < cloudSize - 5; i ++){
    float diff_left[2];
    float diff_right[2];
    float depth = sqrt(_laserCloud->points[i].x * _laserCloud->points[i].x +
                       _laserCloud->points[i].y * _laserCloud->points[i].y +
                       _laserCloud->points[i].z * _laserCloud->points[i].z);

    for(int count = 1; count < 3; count++ ){
      float diffX1 = _laserCloud->points[i + count].x - _laserCloud->points[i].x;
      float diffY1 = _laserCloud->points[i + count].y - _laserCloud->points[i].y;
      float diffZ1 = _laserCloud->points[i + count].z - _laserCloud->points[i].z;
      diff_right[count - 1] = sqrt(diffX1 * diffX1 + diffY1 * diffY1 + diffZ1 * diffZ1);

      float diffX2 = _laserCloud->points[i - count].x - _laserCloud->points[i].x;
      float diffY2 = _laserCloud->points[i - count].y - _laserCloud->points[i].y;
      float diffZ2 = _laserCloud->points[i - count].z - _laserCloud->points[i].z;
      diff_left[count - 1] = sqrt(diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2);
    }

    float depth_right = sqrt(_laserCloud->points[i + 1].x * _laserCloud->points[i + 1].x +
                             _laserCloud->points[i + 1].y * _laserCloud->points[i + 1].y +
                             _laserCloud->points[i + 1].z * _laserCloud->points[i + 1].z);
    float depth_left = sqrt(_laserCloud->points[i - 1].x * _laserCloud->points[i - 1].x +
                            _laserCloud->points[i - 1].y * _laserCloud->points[i - 1].y +
                            _laserCloud->points[i - 1].z * _laserCloud->points[i - 1].z);

    
    if(fabs(diff_right[0] - diff_left[0]) > thBreakCornerDis){
      if(diff_right[0] > diff_left[0]){

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i - 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i - 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i - 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double left_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> left_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){   //TODO: change the plane window size and add thin rod support
          left_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i - j].x - _laserCloud->points[i - j - 1].x,
                                                        _laserCloud->points[i - j].y - _laserCloud->points[i - j - 1].y,
                                                        _laserCloud->points[i - j].z - _laserCloud->points[i - j - 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool left_is_plane = plane_judge(left_list,0.3);

        if(cc < 0.93){//(max_dis < 2*min_dis) && left_surf_dis < 0.05 * depth  && left_is_plane &&
          if(depth_right > depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_right == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
      else{

        Eigen::Vector3d surf_vector = Eigen::Vector3d(_laserCloud->points[i + 1].x - _laserCloud->points[i].x,
                                                      _laserCloud->points[i + 1].y - _laserCloud->points[i].y,
                                                      _laserCloud->points[i + 1].z - _laserCloud->points[i].z);
        Eigen::Vector3d lidar_vector = Eigen::Vector3d(_laserCloud->points[i].x,
                                                       _laserCloud->points[i].y,
                                                       _laserCloud->points[i].z);
        double right_surf_dis = surf_vector.norm();
        //calculate the angle between the laser direction and the surface
        double cc = fabs( surf_vector.dot(lidar_vector) / (surf_vector.norm()*lidar_vector.norm()) );

        std::vector<PointType> right_list;
        double min_dis = 10000;
        double max_dis = 0;
        for(int j = 0; j < 4; j++){ //TODO: change the plane window size and add thin rod support
          right_list.push_back(_laserCloud->points[i - j]);
          Eigen::Vector3d temp_vector = Eigen::Vector3d(_laserCloud->points[i + j].x - _laserCloud->points[i + j + 1].x,
                                                        _laserCloud->points[i + j].y - _laserCloud->points[i + j + 1].y,
                                                        _laserCloud->points[i + j].z - _laserCloud->points[i + j + 1].z);

          if(j == 3) break;
          double temp_dis = temp_vector.norm();
          if(temp_dis < min_dis) min_dis = temp_dis;
          if(temp_dis > max_dis) max_dis = temp_dis;
        }
        // bool right_is_plane = plane_judge(right_list,0.3);

        if(cc < 0.93){ //right_is_plane  && (max_dis < 2*min_dis) && right_surf_dis < 0.05 * depth &&

          if(depth_right < depth_left){
            CloudFeatureFlag[i] = 100;
          }
          else{
            if(depth_left == 0) CloudFeatureFlag[i] = 100;
          }
        }
      }
    }

    // break points select
    if(CloudFeatureFlag[i] == 100){
      std::vector<Eigen::Vector3d> front_norms;
      Eigen::Vector3d norm_front(0,0,0);
      Eigen::Vector3d norm_back(0,0,0);

      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i - k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i - k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i - k].z - _laserCloud->points[i].z);
        tmp.normalize();
        front_norms.push_back(tmp);
        norm_front += (k/6.0)* tmp;
      }
      std::vector<Eigen::Vector3d> back_norms;
      for(int k = 1;k<4;k++){

        float temp_depth = sqrt(_laserCloud->points[i - k].x * _laserCloud->points[i - k].x +
                        _laserCloud->points[i - k].y * _laserCloud->points[i - k].y +
                        _laserCloud->points[i - k].z * _laserCloud->points[i - k].z);

        if(temp_depth < 1){
          continue;
        }

        Eigen::Vector3d tmp = Eigen::Vector3d(_laserCloud->points[i + k].x - _laserCloud->points[i].x,
                                              _laserCloud->points[i + k].y - _laserCloud->points[i].y,
                                              _laserCloud->points[i + k].z - _laserCloud->points[i].z);
        tmp.normalize();
        back_norms.push_back(tmp);
        norm_back += (k/6.0)* tmp;
      }
      double cc = fabs( norm_front.dot(norm_back) / (norm_front.norm()*norm_back.norm()) );
      if(cc < 0.93){
      }else{
        CloudFeatureFlag[i] = 101;
      }

    }

  }

  pcl::PointCloud<PointType>::Ptr laserCloudCorner(new pcl::PointCloud<PointType>());
  pcl::PointCloud<PointType> cornerPointsSharp;

  std::vector<int> pointsLessSharp_ori;

  int num_surf = 0;
  int num_corner = 0;

  for(int i = 5; i < cloudSize - 5; i ++){
    Eigen::Vector3d left_pt = Eigen::Vector3d(_laserCloud->points[i - 1].x,
                                              _laserCloud->points[i - 1].y,
                                              _laserCloud->points[i - 1].z);
    Eigen::Vector3d right_pt = Eigen::Vector3d(_laserCloud->points[i + 1].x,
                                               _laserCloud->points[i + 1].y,
                                               _laserCloud->points[i + 1].z);

    Eigen::Vector3d cur_pt = Eigen::Vector3d(_laserCloud->points[i].x,
                                             _laserCloud->points[i].y,
                                             _laserCloud->points[i].z);

    float dis = _laserCloud->points[i].x * _laserCloud->points[i].x +
                _laserCloud->points[i].y * _laserCloud->points[i].y +
                _laserCloud->points[i].z * _laserCloud->points[i].z;

    double clr = fabs(left_pt.dot(right_pt) / (left_pt.norm()*right_pt.norm()));
    double cl = fabs(left_pt.dot(cur_pt) / (left_pt.norm()*cur_pt.norm()));
    double cr = fabs(right_pt.dot(cur_pt) / (right_pt.norm()*cur_pt.norm()));

    if(clr < 0.999){
      CloudFeatureFlag[i] = 200;
    }

    if(dis < thLidarNearestDis*thLidarNearestDis) continue;

    if(CloudFeatureFlag[i] == 100 || CloudFeatureFlag[i] == 200){ //
      pointsLessSharp_ori.push_back(i);
      laserCloudCorner->push_back(_laserCloud->points[i]);
    }
  }

  for(int i = 0; i < laserCloudCorner->points.size();i++){
      pointsLessSharp.push_back(pointsLessSharp_ori[i]);
      num_corner++;
  }

}


void LidarFeatureExtractor::FeatureExtract(const livox_ros_driver::CustomMsgConstPtr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserCloud,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                           const int Used_Line,const int lidar_type){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
  ptr->clear();
  }
  for(auto & v : vcorner){
  v.clear();
  }
  for(auto & v : vsurf){
  v.clear();
  }
  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){
  int line_num = (int)p.line;
  if(line_num > Used_Line-1) continue;
  if(lidar_type == 0||lidar_type == 1)
  {
      if(p.x < 0.01) continue;
  }
  else if(lidar_type == 2)
  {
      if(std::fabs(p.x) < 0.01) continue;
  }
//  if(p.x < 0.01) continue;
  point.x = p.x;
  point.y = p.y;
  point.z = p.z;
  point.intensity = p.reflectivity;
  point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
  point.normal_y = _int_as_float(line_num);
  laserCloud->push_back(point);
  }
  std::size_t cloud_num = laserCloud->size();
  for(std::size_t i=0; i<cloud_num; ++i){
  int line_idx = _float_as_int(laserCloud->points[i].normal_y);
  laserCloud->points[i].normal_z = _int_as_float(i);
  vlines[line_idx]->push_back(laserCloud->points[i]);
  laserCloud->points[i].normal_z = 0;
  }
  std::thread threads[N_SCANS];
  for(int i=0; i<N_SCANS; ++i){
  threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                     std::ref(vcorner[i]), std::ref(vsurf[i]));
  }
  for(int i=0; i<N_SCANS; ++i){
  threads[i].join();
  }
  for(int i=0; i<N_SCANS; ++i){
  for(int j=0; j<vcorner[i].size(); ++j){
  laserCloud->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
  }
  for(int j=0; j<vsurf[i].size(); ++j){
  laserCloud->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
  }
  }

  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 1.0) < 1e-5)
  laserConerFeature->push_back(p);
  }
  for(const auto& p : laserCloud->points){
  if(std::fabs(p.normal_z - 2.0) < 1e-5)
  laserSurfFeature->push_back(p);
  }
}

void LidarFeatureExtractor::FeatureExtract_hap(const livox_ros_driver::CustomMsgConstPtr &msg,
                                               pcl::PointCloud<PointType>::Ptr& laserCloud,
                                               pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserSurfFeature,
                                               pcl::PointCloud<PointType>::Ptr& laserNonFeature,
                                               const int Used_Line){
  laserCloud->clear();
  laserConerFeature->clear();
  laserSurfFeature->clear();
  laserCloud->clear();
  laserCloud->reserve(15000*N_SCANS);
  for(auto & ptr : vlines){
    ptr->clear();
  }
  for(auto & v : vcorner){
    v.clear();
  }
  for(auto & v : vsurf){
    v.clear();
  }

  int dnum = msg->points.size();

  double timeSpan = ros::Time().fromNSec(msg->points.back().offset_time).toSec();
  PointType point;
  for(const auto& p : msg->points){

    int line_num = (int)p.line;
    if(line_num > Used_Line-1) continue;
    if(p.x < 0.01) continue;
    if (!pcl_isfinite(p.x) ||
        !pcl_isfinite(p.y) ||
        !pcl_isfinite(p.z)) {
      continue;
    }
    point.x = p.x;
    point.y = p.y;
    point.z = p.z;
    point.intensity = p.reflectivity;
    point.normal_x = ros::Time().fromNSec(p.offset_time).toSec() /timeSpan;
    point.normal_y = _int_as_float(line_num);
    laserCloud->push_back(point);
  }

  detectFeaturePoint2(laserCloud, laserSurfFeature, laserNonFeature);

  pcl::PointCloud<PointType>::Ptr laserConerFeature_filter;
  laserConerFeature_filter.reset(new pcl::PointCloud<PointType>());
  laserConerFeature.reset(new pcl::PointCloud<PointType>());
  laserSurfFeature.reset(new pcl::PointCloud<PointType>());
  laserNonFeature.reset(new pcl::PointCloud<PointType>());
  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 1.0) < 1e-5)
      laserConerFeature->push_back(p);
  }

  for(const auto& p : laserCloud->points){
    if(std::fabs(p.normal_z - 2.0) < 1e-5)
      laserSurfFeature->push_back(p);
    if(std::fabs(p.normal_z - 3.0) < 1e-5)
      laserNonFeature->push_back(p);
  }
}

void LidarFeatureExtractor::FeatureExtract_Mid(pcl::PointCloud<pcl::PointXYZINormal>::Ptr &msg,
                                           pcl::PointCloud<PointType>::Ptr& laserConerFeature,
                                           pcl::PointCloud<PointType>::Ptr& laserSurfFeature){
    laserConerFeature->clear();
    laserSurfFeature->clear();
    for(auto & ptr : vlines){
        ptr->clear();
    }
    for(auto & v : vcorner){
        v.clear();
    }
    for(auto & v : vsurf){
        v.clear();
    }
    int cloud_num= msg->points.size();
    for(int i=0; i<cloud_num; ++i){
        int line_idx = std::round(msg->points[i].normal_y);
        msg->points[i].normal_z = _int_as_float(i);

        vlines[line_idx]->push_back(msg->points[i]);

        msg->points[i].normal_z = 0;
    }
    std::thread threads[N_SCANS];
    for(int i=0; i<N_SCANS; ++i){
        threads[i] = std::thread(&LidarFeatureExtractor::detectFeaturePoint, this, std::ref(vlines[i]),
                                 std::ref(vcorner[i]), std::ref(vsurf[i]));
    }
    for(int i=0; i<N_SCANS; ++i){
        threads[i].join();
    }
    for(int i=0; i<N_SCANS; ++i){
        for(int j=0; j<vcorner[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vcorner[i][j]].normal_z)].normal_z = 1.0;
        }
        for(int j=0; j<vsurf[i].size(); ++j){
            msg->points[_float_as_int(vlines[i]->points[vsurf[i][j]].normal_z)].normal_z = 2.0;
        }
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 1.0) < 1e-5)
            laserConerFeature->push_back(p);
    }
    for(const auto& p : msg->points){
        if(std::fabs(p.normal_z - 2.0) < 1e-5)
            laserSurfFeature->push_back(p);
    }
}