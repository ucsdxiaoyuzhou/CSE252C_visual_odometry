#include "map.hpp"


MAP::MAP(){
}


PointCloud<PointXYZ> MAP::pointToPointCloud(vector<Point3f> scenePts){
	int ptsNum = scenePts.size();
	PointCloud<PointXYZ> result;
	for(int n = 0; n < ptsNum; n++){
		PointXYZ pt;
		pt.x = scenePts[n].x;
		pt.y = scenePts[n].y;
		pt.z = scenePts[n].z;
		result.points.push_back(pt);
	}

	result.height = 1;
	result.width = result.points.size();

	return result;
}


void MAP::jointToMap(PointCloud<PointXYZ> frameMap, Eigen::Isometry3d& trans){

	if(false == initialized){
		initialized = true;
		cout << "map initializing! " << endl;
		entireMap = frameMap;
		cout << "map initialized!" << endl;
	}
	PointCloud<PointXYZ>::Ptr tempCloud (new PointCloud<PointXYZ>());

	transformPointCloud(entireMap, *tempCloud, trans.matrix());

	*tempCloud += frameMap;
	entireMap = *tempCloud;


	// static pcl::VoxelGrid<PointXYZ> voxel;
	// voxel.setLeafSize(2.5, 2.5, 2.5);
	// voxel.setInputCloud(tempCloud);

	// voxel.filter(entireMap);

}

// void MAP::showMap(visualization::CloudViewer viewer){
	
// 	*cloud = entireMap;
	
// 	cout << "map points: " << cloud->points.size() << endl;
// 	viewer.showCloud(cloud);

// 	if(waitKey(1000) == 27){
// 		exit;
// 	}


// }