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
#include <time.h>
#include <string.h>
#include <dirent.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d/features2d.hpp>
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/nonfree.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include <opencv2/video/video.hpp>

#include "frame.hpp"
#include "draw.hpp"

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
    int count = 1;
    // Mat accumTranslation;
    double accumTX = 0, accumTY = 0, accumTZ = 0;

    //grap the first image, set it to lastframe, actually, this can be called
    //prevous frame.
    Frame lastframe(leftImgName[0], rightImgName[0], srp.P1, srp.P2);
    
    cout << accumTX << endl;

//============== main loop ============================================================
    while(1){
        //grap the current frame
        Frame currframe(leftImgName[count], rightImgName[count], srp.P1, srp.P2);
        lastframe.matchFrame(currframe);

        //compute the extent of motion
        double move = normofTransform(lastframe.rvec, lastframe.tvec);
        //if the motion is small, it means it is correct, because the motion
        //should be smooth, there cannot ba any sudden change.
        if(move < 3.0){
            accumTX += lastframe.tvec.at<double>(0,0);
            accumTY += lastframe.tvec.at<double>(1,0);
            accumTZ += lastframe.tvec.at<double>(2,0);

            cout << "accum translation: " << accumTX <<" "<<accumTY << " "<<accumTZ << " " << endl;
            //if motion is small, then set the current frame as the previous 
            //frame
            lastframe = currframe;
        }
        else{
            //if motion is too large, then assume the current estimation is wrong
            cout << "bad frame!" << endl;
        }

        count++;
    }
    
//=========== main loop ends ===========================================================
    
    waitKey(0);
    return 0;
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