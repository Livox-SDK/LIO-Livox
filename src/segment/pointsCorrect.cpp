#include "segment/pointsCorrect.hpp"

float gnd_pos[6];
int frame_count = 0;
int frame_lenth_threshold = 5;//5 frames update

int GetNeiborPCA_cor(SNeiborPCA_cor &npca, pcl::PointCloud<pcl::PointXYZ>::Ptr cloud, pcl::KdTreeFLANN<pcl::PointXYZ> kdtree, pcl::PointXYZ searchPoint, float fSearchRadius)
{
    std::vector<float> k_dis;
    pcl::PointCloud<pcl::PointXYZ>::Ptr subCloud(new pcl::PointCloud<pcl::PointXYZ>);

    if(kdtree.radiusSearch(searchPoint,fSearchRadius,npca.neibors,k_dis)>5)
    {
        subCloud->width=npca.neibors.size();
        subCloud->height=1;
        subCloud->points.resize(subCloud->width*subCloud->height);

        for (int pid=0;pid<subCloud->points.size();pid++)//搜索半径内的地面点云 sy
        {
            subCloud->points[pid].x=cloud->points[npca.neibors[pid]].x;
            subCloud->points[pid].y=cloud->points[npca.neibors[pid]].y;
            subCloud->points[pid].z=cloud->points[npca.neibors[pid]].z;
        }
        //利用PCA主元分析法获得点云的三个主方向，获取质心，计算协方差，获得协方差矩阵，求取协方差矩阵的特征值和特长向量，特征向量即为主方向。 sy
        Eigen::Vector4f pcaCentroid;
    	pcl::compute3DCentroid(*subCloud, pcaCentroid);
	    Eigen::Matrix3f covariance;
	    pcl::computeCovarianceMatrixNormalized(*subCloud, pcaCentroid, covariance);
	    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	    npca.eigenVectorsPCA = eigen_solver.eigenvectors();
	    npca.eigenValuesPCA = eigen_solver.eigenvalues();
        float vsum=npca.eigenValuesPCA(0)+npca.eigenValuesPCA(1)+npca.eigenValuesPCA(2);
        npca.eigenValuesPCA(0)=npca.eigenValuesPCA(0)/(vsum+0.000001);//单位化 sy
        npca.eigenValuesPCA(1)=npca.eigenValuesPCA(1)/(vsum+0.000001);
        npca.eigenValuesPCA(2)=npca.eigenValuesPCA(2)/(vsum+0.000001);
    }
    else
    {
        npca.neibors.clear();
    }
    //std::cout << "in PCA2\n";
    return npca.neibors.size();
}

int FilterGndForPos_cor(float* outPoints,float*inPoints,int inNum)
{
    int outNum=0;
    float dx=2;
    float dy=2;
    int x_len = 20;
    int y_len = 10;
    int nx=2*x_len/dx; //80
    int ny=2*y_len/dy; //10
    float offx=-20,offy=-10;
    float THR=0.4;
    

    float *imgMinZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgMaxZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgSumZ=(float*)calloc(nx*ny,sizeof(float));
    float *imgMeanZ=(float*)calloc(nx*ny,sizeof(float));
    int *imgNumZ=(int*)calloc(nx*ny,sizeof(int));
    int *idtemp = (int*)calloc(inNum,sizeof(int));
    for(int ii=0;ii<nx*ny;ii++)
    {
        imgMinZ[ii]=10;
        imgMaxZ[ii]=-10;
        imgMeanZ[ii] = -10;
        imgSumZ[ii]=0;
        imgNumZ[ii]=0;
    }

    for(int pid=0;pid<inNum;pid++)
    {
        idtemp[pid] = -1;
        if((inPoints[pid*4] > -x_len) && (inPoints[pid*4]<x_len)&&(inPoints[pid*4+1]>-y_len)&&(inPoints[pid*4+1]<y_len))
        {
            int idx=(inPoints[pid*4]-offx)/dx;
            int idy=(inPoints[pid*4+1]-offy)/dy;
            idtemp[pid] = idx+idy*nx;
            if (idtemp[pid] >= nx*ny)
                continue;
            imgSumZ[idx+idy*nx] += inPoints[pid*4+2];
            imgNumZ[idx+idy*nx] +=1;
            if(inPoints[pid*4+2]<imgMinZ[idx+idy*nx])
            {
                imgMinZ[idx+idy*nx]=inPoints[pid*4+2];
            }
            if(inPoints[pid*4+2]>imgMaxZ[idx+idy*nx]){
                imgMaxZ[idx+idy*nx]=inPoints[pid*4+2];
            }
        }
    }
    for(int pid=0;pid<inNum;pid++)
    {
        if (outNum >= 60000)
            break;
        if(idtemp[pid] > 0 && idtemp[pid] < nx*ny)
        {
            imgMeanZ[idtemp[pid]] = float(imgSumZ[idtemp[pid]]/(imgNumZ[idtemp[pid]] + 0.0001));
            //最高点与均值高度差小于阈值；点数大于3；均值高度小于1 
            if((imgMaxZ[idtemp[pid]] - imgMeanZ[idtemp[pid]]) < THR && imgNumZ[idtemp[pid]] > 3 && imgMeanZ[idtemp[pid]] < 2)
            {// imgMeanZ[idtemp[pid]]<0&&
                outPoints[outNum*4]=inPoints[pid*4];
                outPoints[outNum*4+1]=inPoints[pid*4+1];
                outPoints[outNum*4+2]=inPoints[pid*4+2];
                outPoints[outNum*4+3]=inPoints[pid*4+3];
                outNum++;
            }
        }
    }

    free(imgMinZ);
    free(imgMaxZ);
    free(imgSumZ);
    free(imgMeanZ);
    free(imgNumZ);
    free(idtemp);
    return outNum;
}

int CalGndPos_cor(float *gnd, float *fPoints,int pointNum,float fSearchRadius)
{
    // 初始化gnd
    for(int ii=0;ii<6;ii++)
    {
        gnd[ii]=0;
    }
    // 转换点云到pcl的格式
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

    //去除异常点
    if (pointNum <= 3)
    {
        return 0;
    }
    cloud->width=pointNum;
    cloud->height=1;
    cloud->points.resize(cloud->width*cloud->height);

    for (int pid=0;pid<cloud->points.size();pid++)
    {
        cloud->points[pid].x=fPoints[pid*4];
        cloud->points[pid].y=fPoints[pid*4+1];
        cloud->points[pid].z=fPoints[pid*4+2];
    }
    pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
    kdtree.setInputCloud (cloud);
    int nNum=0;
    unsigned char* pLabel = (unsigned char*)calloc(pointNum,sizeof(unsigned char));
    for(int pid=0;pid<pointNum;pid++)
    {
        if ((nNum<1000)&&(pLabel[pid]==0))
        {
            SNeiborPCA_cor npca;
            pcl::PointXYZ searchPoint;
            searchPoint.x=cloud->points[pid].x;
            searchPoint.y=cloud->points[pid].y;
            searchPoint.z=cloud->points[pid].z;

            if(GetNeiborPCA_cor(npca,cloud,kdtree,searchPoint,fSearchRadius)>0)
            {
                for(int ii=0;ii<npca.neibors.size();ii++)
                {
                    pLabel[npca.neibors[ii]]=1;
                }

                if(npca.eigenValuesPCA[1]/(npca.eigenValuesPCA[0] + 0.00001)>5000){ //指的是主方向与次方向差异较大。即这一小块接近平面 sy

                        if(npca.eigenVectorsPCA(2,0)>0) //垂直向上？
                        {
                            gnd[0]+=npca.eigenVectorsPCA(0,0);
                            gnd[1]+=npca.eigenVectorsPCA(1,0);
                            gnd[2]+=npca.eigenVectorsPCA(2,0);

                            gnd[3]+=searchPoint.x;
                            gnd[4]+=searchPoint.y;
                            gnd[5]+=searchPoint.z;
                        }
                        else
                        {
                            gnd[0]+=-npca.eigenVectorsPCA(0,0);
                            gnd[1]+=-npca.eigenVectorsPCA(1,0);
                            gnd[2]+=-npca.eigenVectorsPCA(2,0);

                            gnd[3]+=searchPoint.x;
                            gnd[4]+=searchPoint.y;
                            gnd[5]+=searchPoint.z;
                        }
                        nNum++;

                }
            }
        }
    }
    free(pLabel);
    if(nNum>0)
    {
        for(int ii=0;ii<6;ii++)
        {
            gnd[ii]/=nNum; //平均法向量 & 地面点云的中心
        }
        if(abs(gnd[0])<0.1){
            gnd[0]=gnd[0]*(1-abs(gnd[0]))*4.5;
        }
        else if(abs(gnd[0])<0.2){
            gnd[0]=gnd[0]*(1-abs(gnd[0]))*3.2;
        }
        else{
            gnd[0]=gnd[0]*(1-abs(gnd[0]))*2.8;
        }
        gnd[1] = gnd[1]*2.3;
        
    }
    return nNum;
}

int GetRTMatrix_cor(float *RTM, float *v0, float *v1)
{
    // 归一化
    float nv0=sqrt(v0[0]*v0[0]+v0[1]*v0[1]+v0[2]*v0[2]);
    v0[0]/=(nv0+0.000001);
    v0[1]/=(nv0+0.000001);
    v0[2]/=(nv0+0.000001);

    float nv1=sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2]);
    v1[0]/=(nv1+0.000001);
    v1[1]/=(nv1+0.000001);
    v1[2]/=(nv1+0.000001);

    // 叉乘
    float v2[3];
    v2[0]=v0[1]*v1[2]-v0[2]*v1[1];
    v2[1]=v0[2]*v1[0]-v0[0]*v1[2];
    v2[2]=v0[0]*v1[1]-v0[1]*v1[0];

    // 正余弦
    float cosAng=0,sinAng=0;
    cosAng=v0[0]*v1[0]+v0[1]*v1[1]+v0[2]*v1[2];
    sinAng=sqrt(1-cosAng*cosAng);

    // 计算旋转矩阵
    RTM[0]=v2[0]*v2[0]*(1-cosAng)+cosAng;
    RTM[4]=v2[1]*v2[1]*(1-cosAng)+cosAng;
    RTM[8]=v2[2]*v2[2]*(1-cosAng)+cosAng;

    RTM[1]=RTM[3]=v2[0]*v2[1]*(1-cosAng);
    RTM[2]=RTM[6]=v2[0]*v2[2]*(1-cosAng);
    RTM[5]=RTM[7]=v2[1]*v2[2]*(1-cosAng);

    RTM[1]+=(v2[2])*sinAng;
    RTM[2]+=(-v2[1])*sinAng;
    RTM[3]+=(-v2[2])*sinAng;

    RTM[5]+=(v2[0])*sinAng;
    RTM[6]+=(v2[1])*sinAng;
    RTM[7]+=(-v2[0])*sinAng;

    return 0;
}

int CorrectPoints_cor(float *fPoints,int pointNum,float *gndPos)
{
    float RTM[9];
    float gndHeight=0;
    float znorm[3]={0,0,1};
    float tmp[3];

    GetRTMatrix_cor(RTM,gndPos,znorm);

    gndHeight = RTM[2]*gndPos[3]+RTM[5]*gndPos[4]+RTM[8]*gndPos[5];

    for(int pid=0;pid<pointNum;pid++)
    {
        tmp[0]=RTM[0]*fPoints[pid*4]+RTM[3]*fPoints[pid*4+1]+RTM[6]*fPoints[pid*4+2];
        tmp[1]=RTM[1]*fPoints[pid*4]+RTM[4]*fPoints[pid*4+1]+RTM[7]*fPoints[pid*4+2];
        tmp[2]=RTM[2]*fPoints[pid*4]+RTM[5]*fPoints[pid*4+1]+RTM[8]*fPoints[pid*4+2]-gndHeight;

        fPoints[pid*4]=tmp[0];
        fPoints[pid*4+1]=tmp[1];
        fPoints[pid*4+2]=tmp[2];
    }
    return 0;
}


int GetGndPos(float *pos, float *fPoints,int pointNum){
    float *fPoints3=(float*)calloc(60000*4,sizeof(float));//地面点
    int pnum3 = FilterGndForPos_cor(fPoints3,fPoints,pointNum);
    float tmpPos[6];
    if (pnum3 < 3)
    {
        std::cout << "too few ground points!\n";
    }
    int gndnum = CalGndPos_cor(tmpPos,fPoints3,pnum3,1.0);//用法向量判断，获取到法向量 & 地面搜索点，放到tmppos
    if(gnd_pos[5]==0){
        memcpy(gnd_pos,tmpPos,sizeof(tmpPos));
    }
    else{

        if(frame_count<frame_lenth_threshold&&tmpPos[5]!=0){
            if(gndnum>0&&abs(gnd_pos[0]-tmpPos[0])<0.1&&abs(gnd_pos[1]-tmpPos[1])<0.1){//更新法向量            
                for(int i = 0;i<6;i++){
                    gnd_pos[i] = (gnd_pos[i]+tmpPos[i])*0.5;
                }
                frame_count = 0;
            }
            else{
                frame_count++;
            }
        }
        else if(tmpPos[5]!=0){
            memcpy(gnd_pos,tmpPos,sizeof(tmpPos));
            frame_count = 0;
        }
    }
   
    memcpy(pos,gnd_pos,sizeof(float)*6);

    free(fPoints3);
    
    return 0;
}



