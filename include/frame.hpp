//
//  frame.hpp
//  sceneReconstruct
//
//  Created by 周晓宇 on 4/3/17.
//  Copyright © 2017 xiaoyu. All rights reserved.
//

#ifndef frame_hpp
#define frame_hpp

#include <stdio.h>
#include <iostream>
#include <iomanip>

#include <opencv2/core/core.hpp>
//#include <opencv2/core/eigen.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/calib3d/calib3d.hpp"

/*
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
// #include <pcl/common/projection_matrix.h>
*/
#include "draw.hpp"

using namespace cv;
using namespace std;
// using namespace pcl;

class Frame{

public:
    Mat imgL, imgR;
    vector<KeyPoint> keypointL, keypointR;
    vector<Point2f> p_keypointL, p_keypointR;
    vector<Point3f> scenePts;
    Mat despL, despR;
    Mat P1, P2;
    Mat rvec, tvec;

    
    Frame(string filenameL,
          string filenameR,
          Mat _P1,
          Mat _P2);
    void detectFeatures();
    void getFrame(string filenameL, string filenameR);
    void matchFrame(Frame frame);
    void getMotion(vector<DMatch> matches, Frame frame);
};




#endif /* frame_hpp */
