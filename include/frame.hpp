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
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include "opencv2/calib3d/calib3d.hpp"

#include "ORBextractor.h"

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/kdtree/kdtree_flann.h>

/*
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/common/transforms.h>
// #include <pcl/common/projection_matrix.h>
*/
#include "draw.hpp"

using namespace cv;
using namespace std;
using namespace ORB_SLAM2;
using namespace pcl;

class Frame{

public:
    int frameID;
    bool success = true;
    Mat imgL, imgR;
    vector<KeyPoint> keypointL, keypointR;
    vector<Point2f> p_keypointL, p_keypointR;
    vector<Point3f> scenePts;
    vector<bool> farPtsIdx;
    vector<Point2f> forDrawL;

    KdTreeFLANN<PointXY> kdtree_keypointL;


    int matchedNumWithCurrentFrame = 0;
    // vector<int> farIdx;

    Mat despL, despR;
    Mat P1, P2;
    Mat rvec, tvec;

    
    Frame(string filenameL,
          string filenameR,
          Mat _P1,
          Mat _P2,
          int id);
    
    Frame(string filenameL, Mat _P1, int id);
    void setKeyframe(string filenameR, Mat _P2);
    void detectFeatures();
    void stereoMatchKLT(const vector<Point2f>& p1, //keypoint in the previous frame
                            const vector<Point2f>& p2, //keypoint in the current frame
                               vector<Point3f>& obj_pts,
                               vector<Point2f>& img_pts,
                               vector<int>& farIdx);

    void stereoMatchFeature(const vector<Point2f>& p1, //keypoint in the previous frame
                            const vector<Point2f>& p2, //keypoint in the current frame
                            vector<Point3f>& obj_pts,
                            vector<Point2f>& img_pts,
                            vector<int>& farIdx);

    void PnP(vector<Point3f> obj_pts, 
             vector<Point2f> img_pts,
             Mat& inliers);

    void getFrame(string filenameL, string filenameR);
    void matchFrame(Frame frame);
    void matchFrameNN(Frame frame);
    void getMotion(vector<DMatch> matches, Frame frame);
    void matchFeature(const Mat& desp1, const Mat& desp2, 
                         const vector<Point2f>& p_keypoint1, 
                         const vector<Point2f>& p_keypoint2,
                         vector<Point2f>& matched_keypoint1,
                         vector<Point2f>& matched_keypoint2);
    void releaseMemory();
};




#endif /* frame_hpp */
