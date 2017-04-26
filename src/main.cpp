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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/video/video.hpp>

#include <pcl/io/pcd_io.h>
// #include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/cloud_viewer.h>
// #include <pcl/registration/icp.h>


#include "frame.hpp"
#include "draw.hpp"
#include "map.hpp"

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
    Mat Q;
    Mat map11, map12;
    Mat map21, map22;
    Point2f leftup;
    Point2f rightbottom;
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

void accumulateTransformation(Eigen::Isometry3d& accumT, Mat currR, Mat currT);
Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );
int main(int argc, const char * argv[]) {
    if(argc != 4){
        cerr << endl<<"usage: ./path_to_left_camera_image_directory ./path_to_right_camera_image_directory ./path_to_camera_setting_file" << endl;
        return 1;
    }
    
    // load data
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

    int row = fsSettings["height"];
    int col = fsSettings["width"];

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
    int count = 1;
    // Mat accumTranslation = Mat::eye(4,4,CV_32F);
    Eigen::Isometry3d accumTranslation = Eigen::Isometry3d::Identity();
    visualization::CloudViewer viewer("Cloud Viewer");
    PointCloud<PointXYZ>::Ptr cloud (new PointCloud<PointXYZ>);

    //grap the first image, set it to lastframe, actually, this can be called
    //prevous frame.
    Frame lastframe(leftImgName[0], rightImgName[0], srp.P1, srp.P2);
    
    string filePath = "../pose08.txt";
    ofstream poseFileOut;
    poseFileOut.open(filePath.c_str(), std::ofstream::out | std::ofstream::trunc);

//============== main loop ============================================================
    for(;count < leftImgName.size()-1; count++){
        //grap the current frame
        Frame currframe(leftImgName[count], rightImgName[count], srp.P1, srp.P2);
        
        lastframe.matchFrame(currframe);

        //compute the extent of motion
        double move = normofTransform(lastframe.rvec, lastframe.tvec);
        //if the motion is small, it means it is correct, because the motion
        //should be smooth, there cannot ba any sudden change.
        if(move < 3.0){
            accumulateTransformation(accumTranslation, lastframe.rvec, lastframe.tvec);
            // cout <<endl << accumTranslation << endl;
            //if motion is small, then set the current frame as the previous 
            //frame
            /*
            

            accumTranslation.row(0).copyTo(forshow.colRange(0,4));
            accumTranslation.row(1).copyTo(forshow.colRange(4,8));
            accumTranslation.row(2).copyTo(forshow.colRange(8,12));
*/
            // Mat forshow = Mat::zeros(1,12,CV_32F);

            // cout << "Frame number: " << count <<endl;
            // for(int n = 0; n < 11; n++){
            //     poseFileOut << accumTranslation.at<float>(0,n) << " " ;
            // }
            Eigen::Isometry3d curTrans =  cvMat2Eigen(lastframe.rvec, lastframe.tvec);
            myMap.jointToMap(myMap.pointToPointCloud(lastframe.scenePts), 
                             curTrans);

            // myMap.showMap(viewer);
            *cloud = myMap.entireMap;
            viewer.showCloud(cloud);
            if(waitKey(5) == 27){
                exit;
            }

            
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< lastframe.rvec.at<double>(n,0) <<" ";
                cout<< setw(15) << lastframe.rvec.at<double>(n,0) <<" ";
            }
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< lastframe.tvec.at<double>(n,0) << " ";
                cout<< setw(15) << lastframe.tvec.at<double>(n,0) << " ";
            }
            poseFileOut << endl;
            cout << "   move:" << move  <<endl;

            /*for(int r = 0; r < 3; r++){
                for(int c = 0; c < 4; c++){
                    poseFileOut << accumTranslation(r,c) << " ";
                    cout << accumTranslation(r,c) << " ";
                }
            }
            poseFileOut <<"\n" ;
            cout << "\n" ;
*/

          //  cout << forshow.at<float>(0,3)<<" "<< forshow.at<float>(0,7)<<" "<< forshow.at<float>(0,11)<<" " << endl;
            lastframe = currframe;
        }
        else{
            //if motion is too large, then assume the current estimation is wrong
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< 0.0 <<" ";
                cout<< setw(15) << 0.0 <<" ";
            }
            for(int n = 0; n < 3; n++){
                poseFileOut << setw(15)<< 0.0 << " ";
                cout<< setw(15) << 0.0 << " ";
            }
            poseFileOut << endl;
            cout << "   move:" << move;
            cout << "bad frame!" << endl;
        }

    }
    poseFileOut.close();
    cloud->height = 1;
    cloud->width = cloud->points.size();
    
    pcl::io::savePCDFileASCII("traj.pcd", *cloud);
    cout << "trajectory saved!" << endl;
    
//=========== main loop ends ===========================================================
    
    waitKey(0);
    return 0;
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
void accumulateTransformation(Eigen::Isometry3d& accumT, Mat currR, Mat currT){
    cv::Mat R;
    cv::Rodrigues( currR, R );
    Eigen::Matrix3d r;

    r(0,0) = R.at<double>(0,0);
    r(0,1) = R.at<double>(0,1);
    r(0,2) = R.at<double>(0,2);
    r(1,0) = R.at<double>(1,0);
    r(1,1) = R.at<double>(1,1);
    r(1,2) = R.at<double>(1,2);
    r(2,0) = R.at<double>(2,0);
    r(2,1) = R.at<double>(2,1);
    r(2,2) = R.at<double>(2,2);

    Eigen::Isometry3d T = Eigen::Isometry3d::Identity();

    Eigen::AngleAxisd angle(r);
    Eigen::Translation<double,3> trans(currT.at<double>(0,0), 
                                       currT.at<double>(1,0), 
                                       currT.at<double>(2,0));
    T = angle;
    T(0,3) = currT.at<double>(0,0); 
    T(1,3) = currT.at<double>(1,0); 
    T(2,3) = currT.at<double>(2,0);

    Eigen::Isometry3d invT = T.inverse();
    accumT = invT * accumT;

    // cout << accumT.matrix() << endl<<endl;
    // accumT = T;
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
}






