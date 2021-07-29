cmake_minimum_required(VERSION 2.8.3)
project(lio_livox)

SET(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")

#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

find_package(catkin REQUIRED COMPONENTS
  	     message_generation
  	     geometry_msgs
  	     nav_msgs
  	     sensor_msgs
  	     roscpp
  	     rospy
  	     std_msgs
  	     tf
  	     tf_conversions
  	     rosbag
  	     livox_ros_driver
  	     eigen_conversions
  	     pcl_conversions
  	     pcl_ros
  	     message_filters
  	     std_srvs)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(Ceres REQUIRED)
find_package(OpenCV REQUIRED)
find_package(SuiteSparse REQUIRED)

include_directories(include
        	    ${catkin_INCLUDE_DIRS}
		    ${EIGEN3_INCLUDE_DIR}
		    ${PCL_INCLUDE_DIRS}
        	    ${CERES_INCLUDE_DIRS}
		    ${OpenCV_INCLUDE_DIRS}
        	    ${SUITESPARSE_INCLUDE_DIRS})

##################
## ROS messages ##
##################

generate_messages(DEPENDENCIES std_msgs)

catkin_package(CATKIN_DEPENDS geometry_msgs nav_msgs roscpp rospy std_msgs message_runtime 
  	       pcl_conversions pcl_ros message_filters std_srvs tf_conversions tf 
	       eigen_conversions DEPENDS PCL OpenCV INCLUDE_DIRS include)

add_executable (ScanRegistration 
		src/lio/ScanRegistration.cpp
		src/lio/LidarFeatureExtractor.cpp
        	src/segment/segment.cpp
        	src/segment/pointsCorrect.cpp)
target_link_libraries(ScanRegistration ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable (PoseEstimation 
		src/lio/PoseEstimation.cpp
		src/lio/Estimator.cpp
		src/lio/IMUIntegrator.cpp
		src/lio/ceresfunc.cpp 
		src/lio/Map_Manager.cpp)
target_link_libraries(PoseEstimation ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${CERES_LIBRARIES})



