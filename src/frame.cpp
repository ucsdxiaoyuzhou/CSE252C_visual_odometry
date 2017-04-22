//
//  frame.cpp
//  sceneReconstruct
//
//  Created by 周晓宇 on 4/3/17.
//  Copyright © 2017 xiaoyu. All rights reserved.
//

#include "frame.hpp"

Frame::Frame(string filenameL, string filenameR, Mat _P1, Mat _P2){
    imgL = imread(filenameL);
    imgR = imread(filenameR);
    P1 = _P1;
    P2 = _P2;
    //if use Harris or anyother corner detector, gray-scale images are needed
//    cvtColor( imgL  , imgL, CV_BGR2GRAY );
//    cvtColor( imgR  , imgR, CV_BGR2GRAY );

    //Gaussian blur image, preprocessing
    GaussianBlur(imgL, imgL, Size(7,7), 0.5);
    GaussianBlur(imgR, imgR, Size(7,7), 0.5);
    if(!imgL.data || ! imgR.data){
        cout << "image does not exist..." << endl;
        exit;
    }
    
   // OrbFeatureDetector detector(500, 1.2, 8);
    SurfFeatureDetector detector(1000);
    SurfDescriptorExtractor descriptor;
    
    detector.detect(imgL, keypointL);
    detector.detect(imgR, keypointR);
    
//    KeyPoint::convert(keypointL, p_keypointL);
//    KeyPoint::convert(keypointR, p_keypointR);
    
    descriptor.compute(imgL, keypointL, despL);
    descriptor.compute(imgR, keypointR, despR);
}


void Frame::matchFrame(Frame frame){

    // use FLANN matcher to match features between current frame and 
    // the previous frame.
    FlannBasedMatcher matcher;
    std::vector< DMatch > matches;
    matcher.match( despL, frame.despL, matches );
    
    //standard method to select good matches, provided by OpenCV
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < despL.rows; i++ ){ 
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    // cout << "min_dist: " << min_dist << endl<<endl;
    std::vector< DMatch > good_matches;

    for( int i = 0; i < despL.rows; i++ ){
        if( matches[i].distance <= max(3.5*min_dist, 0.2) ){
            good_matches.push_back( matches[i]);
        }
    }
    drawFeature(imgL, keypointL, "features");
    drawMatch(imgL, keypointL, frame.keypointL, matches, 2, "matched features");
    
    getMotion(good_matches, frame);
    
}

void Frame::getMotion(vector<DMatch> matches, 
                      Frame frame){
    vector<Point2f> temp_p_keypointL, temp_p_keypointR;
    vector<Point2f> p_keypointCurr;
    
    for(int n = 0; n < matches.size(); n++){
        //temp_p_keypointL stores inlier features in the previous left image
        temp_p_keypointL.push_back(keypointL[matches[n].queryIdx].pt);
        //p_keypoint2 stores inlier features in the current right image
        p_keypointCurr.push_back(frame.keypointL[matches[n].trainIdx].pt);
    }
    //get 3D pose
    vector<float> err;
    vector<uchar> status;
    Size winSize = Size(199,11);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
    
    //temp_p_keypointR stores features in the right image correspond 
    //to the left image after optical flow
    //these pairs of features are used for getting 3D points
    calcOpticalFlowPyrLK(imgL, imgR, 
                         temp_p_keypointL,temp_p_keypointR,
                         status, err, winSize,
                         6, termcrit, OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

    //below can be used to delete outliers from optical flow
    //========= select good matches================
    vector<int> inlierIdx;
    vector<Point2f> p_left_trackedPts;
    vector<Point2f> p_right_trackedPts;
   
    Point2f pt;
    vector<Point2f> p_featuresL, p_featuresR;
   
    for(int n = 0; n < status.size(); n++){
        if((1 == status[n]) && 
           (err[n] < 0.25) &&
           (fabs(temp_p_keypointL[n].y - temp_p_keypointR[n].y) < 2.0)){

            inlierIdx.push_back(n);//store inlier idx

            pt = temp_p_keypointL[n];
            p_featuresL.push_back(pt);

            pt = temp_p_keypointR[n];
            p_featuresR.push_back(pt);
        }
    }

    temp_p_keypointL = p_featuresL;
    temp_p_keypointR = p_featuresR;

    //for drawing
    vector<KeyPoint> temp_keypointL, temp_keypointR;
    KeyPoint::convert(temp_p_keypointL, temp_keypointL);
    KeyPoint::convert(temp_p_keypointR, temp_keypointR);
    drawMatch(imgL, temp_p_keypointL, temp_p_keypointR, 2, "stereo match");

    //====compute depth=======
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    //from paper: Vision meets Robotics: The KITTI Dataset
    //the format of P2 is:
    //  f_u,    0,    c_u,    -f_u*b_x
    //   0 ,  f_v,    c_v,        0
    //   0 ,    0,      1,        0
    // the origin is at the right camera, which is unusual

    //only compute 3D points for matched features
    Point3f pd;
    vector<Point3f> obj_pts;//3D points
    for(int n = 0; n<temp_p_keypointL.size(); n++){
        d = temp_p_keypointL[n].x - temp_p_keypointR[n].x;
        pd.x = b*(temp_p_keypointL[n].x - uc)/d;
        pd.y = b*(temp_p_keypointL[n].y - vc)/d;
        pd.z = b*f/d;
        
        obj_pts.push_back(pd);
    }


    //=====obtain tracked points in the second frame=====
    vector<Point2f> img_pts;//2D matched feature points in current image
    //according to inliers previously in optical flow, inliers need to be
    //selected again here
    for(int n = 0; n < inlierIdx.size(); n++){
        pt = p_keypointCurr[inlierIdx[n]];
        img_pts.push_back(pt);
    }
    
    //solve PnP
    Mat K = P2.colRange(0,3).clone();
    Mat inliers;

    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   rvec, tvec, false, 200, 3.0, 30, inliers);
    // Mat t_tvec;
    // transpose(tvec, t_tvec);

    // cout <<"inlier numbers: "<< inliers.rows<<", tvec: "<< t_tvec << endl;
  

    
    //below is used for visulization
    vector<KeyPoint> mKeypoint1, mKeypoint2;
    KeyPoint kp;

    vector<KeyPoint> k_img_pts;
    KeyPoint::convert(img_pts, k_img_pts);
    for(int i = 0; i < inliers.rows; i++){
        int idx = inliers.at<int>(i, 0);
        kp = temp_keypointL[idx];
        mKeypoint1.push_back(kp);
        kp = k_img_pts[idx];
        mKeypoint2.push_back(kp);
    }
    drawMatch(imgL, mKeypoint1, mKeypoint2, 2, "PnP inliers");
    
}
















