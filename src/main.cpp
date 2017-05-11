//
//  main.cpp
//  sceneReconstruct
//
//  Created by 周晓宇 on 3/16/17.
//  Copyright © 2017 xiaoyu. All rights reserved.
//
//  use strict standard to track features and camera, and decide should I add a new keyframe by according to baseline distance and number of tracked features. then use a more tolerant standard to judge should I do a triangulation.
//
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <fstream>
 
#include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
// #include <pcl/registration/icp.h>

#include "frame.hpp"
#include "draw.hpp"
#include "map.hpp"
#include "optimizer.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>


using namespace cv;
using namespace std;

//used to store stereo camera parameters
//for current project, because images are rectified
// only P1, P2 are used here.
struct STEREO_RECTIFY_PARAMS{
    Mat P1, P2;
    Mat R1, R2;
};

//compute the extent of motion by taking both rotation and translation
//into account
double normofTransform( cv::Mat rvec, cv::Mat tvec ){
    return fabs(MIN(norm(rvec), 2*M_PI-norm(rvec)))+ fabs(norm(tvec));
}
//used to obtain image file name vector
void LoadImages(const string &strPathLeft,
                const string &strPathRight,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight);
//used to find nearby loop
void checkNearbyLoop(vector<Frame>& keyframe, 
                     Frame& currFrame,
                     Optimizer& opt,
                     int step);
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );

int main(int argc, const char * argv[]) {
    if(argc != 4){
        cerr << endl<<"usage: ./path_to_left_camera_image_directory ./path_to_right_camera_image_directory ./path_to_camera_setting_file" << endl;
        return 1;
    }
    
//=================load data==========================================
    vector<string> leftImgName;
    vector<string> rightImgName;
    STEREO_RECTIFY_PARAMS srp; // used to store 

    LoadImages(string(argv[1]), string(argv[2]),
               leftImgName, rightImgName);
    
    cv::FileStorage fsSettings(argv[3], cv::FileStorage::READ);
    if(!fsSettings.isOpened()){
        cerr << "ERROR: Wrong path to settings" << endl;
        return -1;
    }
    
    fsSettings["LEFT.P"] >> srp.P1;
    fsSettings["RIGHT.P"] >> srp.P2;

    double lowerMovementThres = fsSettings["lower move threshold"];
    double upperMovementThres = fsSettings["upper move threshold"];
    int matchedThres = fsSettings["feature number threshold"];
    int frameThres = fsSettings["frame threshold"];

    int row = fsSettings["height"];
    int col = fsSettings["width"];
    int fullTimes = fsSettings["global optimization times"];
    int localTimes = fsSettings["local optimization times"];
    int nearbyLoopStep = fsSettings["nearby loop step"];

    string TrajectoryFile = fsSettings["trajectory file"];

    cout << "system overview: " << endl;
    cout << " lower movement threshold:  " << lowerMovementThres << endl;
    cout << " upper movement threshold:  " << upperMovementThres << endl;
    cout << " feature number threshold:  " << matchedThres << endl;
    cout << "          frame threshold:  " << frameThres << endl;
    cout << "         nearby loop step:  " << nearbyLoopStep << endl;
    cout << "global optimization times:  " << fullTimes << endl;
    cout << " local optimization times:  " << localTimes << endl;

    if(srp.P1.empty() || srp.P2.empty() || row==0 || col==0){
        cerr << "ERROR: Calibration parameters to rectify stereo are missing!" << endl;
        return -1;
    }
    cout << "P1: " << endl;
    cout << srp.P1 << endl;
    cout << "P2: " << endl;
    cout << srp.P2 << endl;

    Size imageSize = Size(col, row);
    clock_t t;
//=============== initialize system ============================================
    MAP myMap;
    vector<Frame> keyframe; // keyframe

    visualization::CloudViewer viewer("Cloud Viewer");
    PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

    Frame lastframe(leftImgName[0], rightImgName[0], srp.P1, srp.P2, 0);
    keyframe.push_back(lastframe);


    // Frame lastframe(leftImgName[0], srp.P1, 0);
    // lastframe.setKeyframe(rightImgName[0], srp.P2);
    // keyframe.push_back(lastframe);

    ofstream poseFileOut;
    poseFileOut.open(TrajectoryFile.c_str(), std::ofstream::out | std::ofstream::trunc);

    int intervalFrame = 0;
    double accumMove = 0.0;

    Optimizer optimizer(fullTimes, localTimes);

    int deleteKeyframe = -20;
    int countKeyframe = 1;

    Mat prevRvec, prevTvec;

//============== main loop ============================================================
    for(int count = 1;count < leftImgName.size()-3; count++){
        //grap the current frame
        intervalFrame++;
        Frame currframe(leftImgName[count], rightImgName[count], srp.P1, srp.P2, count);
        // Frame currframe(leftImgName[count], srp.P1, count);

        keyframe.back().matchFrame(currframe);
        accumMove = normofTransform(keyframe.back().rvec, keyframe.back().tvec);

        if(((accumMove > lowerMovementThres) && 
            (accumMove < upperMovementThres)) || 
            intervalFrame >= frameThres ||
            ((keyframe.back().matchedNumWithCurrentFrame < matchedThres) &&
             (accumMove < upperMovementThres))){
            
            cout << endl<<endl<<endl<<"========================================================="<<endl;
            cout <<"insert keyframe." << endl;
            optimizer.addNewNodeEdge(keyframe.back().frameID, 
            						 currframe.frameID,
            						 keyframe.back().rvec, 
            						 keyframe.back().tvec,
                                     true);

            //add new keyframe
            accumMove = 0.0;
            intervalFrame = 0;
            
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< keyframe.back().rvec.at<double>(n,0) <<" ";
                cout<< setw(15) << keyframe.back().rvec.at<double>(n,0) <<" ";
            }
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< keyframe.back().tvec.at<double>(n,0) << " ";
                cout<< setw(15) << keyframe.back().tvec.at<double>(n,0) << " ";
            }
            poseFileOut << endl;
            cout << endl;

            //check nearby loop
            checkNearbyLoop(keyframe, currframe, optimizer, nearbyLoopStep);
            
            if(keyframe.back().success == true){
                Eigen::Isometry3d curTrans =  cvMat2Eigen(keyframe.back().rvec, keyframe.back().tvec);
                myMap.jointToMap(myMap.pointToPointCloud(keyframe.back().scenePts), 
                                curTrans);
            }
            else{
                cout << "failed" << endl;
                Eigen::Isometry3d curTrans =  cvMat2Eigen(keyframe[countKeyframe-1].rvec, keyframe[countKeyframe-1].tvec);
                myMap.jointToMap(myMap.pointToPointCloud(keyframe[countKeyframe-1].scenePts), 
                                curTrans);
            }

            *cloud = myMap.entireMap;
            viewer.showCloud(cloud);
            if(waitKey(5) == 27){};

            // currframe.setKeyframe(rightImgName[count], srp.P2);
            keyframe.push_back(currframe);

            countKeyframe++; 
            deleteKeyframe++;
            if(deleteKeyframe > 0){
                keyframe[deleteKeyframe].releaseMemory();
            }  
        }

    }
    

    optimizer.fullOptimize();

/*
    for(int i = 0; i < keyframe.size(); i ++){
        g2o::VertexSE3* vertex = dynamic_cast<g2o::VertexSE3*>(optimizer.globalOptimizer.vertex(keyframe[i].frameID));
        Eigen::Isometry3d pose = vertex->estimate(); //该帧优化后的位姿
        for(int r = 0; r < 3; r++){
            for(int c = 0; c < 4; c++){
                poseFileOut << setw(15)<< pose(r,c) <<" ";
                cout<< setw(15) << pose(r,c) <<" ";
            }
        } 
        poseFileOut << endl;
        cout << endl;
    }*/
    poseFileOut.close();




    cloud->height = 1;
    cloud->width = cloud->points.size();
    pcl::io::savePCDFileASCII("traj.pcd", *cloud);
    cout << "trajectory saved!" << endl;
    
//=========== main loop ends ===========================================================
    
    waitKey(0);
    return 0;
}

void checkNearbyLoop(vector<Frame>& keyframe, 
                     Frame& currFrame,
                     Optimizer& opt,
                     int step){
    //always check from the last keyframe
    int frameSize = keyframe.size();
    for(int n = 0; n < min(step-1, frameSize-1); n++){
        int idx = frameSize - n - 2;
        keyframe[idx].matchFrame(currFrame);

        double move = normofTransform(keyframe[idx].rvec, keyframe[idx].tvec);
        if(keyframe[idx].matchedNumWithCurrentFrame > 25 &&
           move < 10.0){

            opt.addNewNodeEdge(keyframe[idx].frameID,
                               currFrame.frameID,
                               keyframe[idx].rvec,
                               keyframe[idx].tvec,
                               false);
            // cout << "found a nearby loop" << endl;
            for(int n = 0; n < 3; n++){
                // poseFileOut << setw(15)<< keyframe.back().rvec.at<double>(n,0) <<" ";
                cout<< setw(15) << keyframe[idx].rvec.at<double>(n,0) <<" ";
            }
            for(int n = 0; n < 3; n++){
                // poseFileOut << setw(15)<< keyframe.back().tvec.at<double>(n,0) << " ";
                cout<< setw(15) << keyframe[idx].tvec.at<double>(n,0) << " ";
            }
            cout << endl;
        }
    }
}




Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec )
{
    cv::Mat R;
    cv::Rodrigues( rvec, R );
    Eigen::Matrix3d r;
    for ( int i=0; i<3; i++ )
        for ( int j=0; j<3; j++ ) 
            r(i,j) = R.at<double>(i,j);
  
    // convert translation vector to translation matrix
    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    T = angle;
    T(0,3) = tvec.at<double>(0,0); 
    T(1,3) = tvec.at<double>(1,0); 
    T(2,3) = tvec.at<double>(2,0);
    return T;
}


void LoadImages(const string &strPathLeft,
                const string &strPathRight,
                vector<string> &vstrImageLeft,
                vector<string> &vstrImageRight)
{
    char *cstr = new char[strPathLeft.length()+1];
    strcpy(cstr, strPathLeft.c_str());
    DIR *dp;
    int i = 0;
    struct dirent *ep;
    dp = opendir(cstr);
    
    if(dp != NULL){
        while(ep = readdir(dp)){
            i++;
        }
        closedir(dp);
    }
    else
        cerr <<"could not open the directory! " << endl;
    
    cout << "total " << i << " images!" << endl;
    
    for(int n = 0; n < i; n++){
        
        stringstream ss;
        ss.fill('0');
        ss.width(6);
        ss << n;
        vstrImageLeft.push_back(strPathLeft + "/" + ss.str() + ".png");
        vstrImageRight.push_back(strPathRight + "/" + ss.str() + ".png");
        
    }
    delete cstr;
}

