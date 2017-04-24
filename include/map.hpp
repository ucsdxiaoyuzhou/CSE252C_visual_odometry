#ifndef map_hpp
#define map_hpp

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "frame.hpp"

using namespace pcl;
using namespace std;

class map{

public:
	PointCloud<PointXYZ> entireMap;

	map();
	void jointToMap(PointCloud<PointXYZ> frameMap, Eigen::Isometry3d& trans);
	void pointToPointCloud(vector<Point3f> scenePts;)
};














#endif /* map_hpp */
