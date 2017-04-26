#ifndef map_hpp
#define map_hpp

#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/common/transforms.h>

#include <stdio.h>
#include <iostream>
#include <fstream>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "frame.hpp"

using namespace pcl;
using namespace std;

class MAP{

public:
	PointCloud<PointXYZ> entireMap;// (new PointCloud<PointXYZ>);
	bool initialized = false;


	MAP();
	void jointToMap(PointCloud<PointXYZ> frameMap, Eigen::Isometry3d& trans);
	PointCloud<PointXYZ> pointToPointCloud(vector<Point3f> scenePts);
	// void showMap(visualization::CloudViewer viewer);
};














#endif /* map_hpp */
