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
    if(!imgL.data || ! imgR.data){
        cout << "image does not exist..." << endl;
        exit;
    }

    GaussianBlur(imgL, imgL, Size(7,7), 0.5);
    GaussianBlur(imgR, imgR, Size(7,7), 0.5);
    
   // OrbFeatureDetector detector(500, 1.2, 8);
    SurfFeatureDetector detector(600);
    SurfDescriptorExtractor descriptor;
    
    detector.detect(imgL, keypointL);
    detector.detect(imgR, keypointR);
    
    KeyPoint::convert(keypointL, p_keypointL);
    KeyPoint::convert(keypointR, p_keypointR);
    
    descriptor.compute(imgL, keypointL, despL);
    descriptor.compute(imgR, keypointR, despR);

    // stereoMatchFeature();
    // stereoMatchKLT();
    drawFeature(imgL, p_keypointL, "features");
}

void Frame::stereoMatchFeature(const vector<Point2f>& p1, //keypoint in the previous frame
                               const vector<Point2f>& p2, //keypoint in the current frame
                               vector<Point3f>& obj_pts,
                               vector<Point2f>& img_pts,
                               vector<int>& farIdx){
    
    img_pts.clear();

    SurfDescriptorExtractor descriptor;
    Mat tempDespL;
    vector<KeyPoint> kp1;
    KeyPoint::convert(p1, kp1);

    descriptor.compute(imgL, kp1, tempDespL);
    cout << "right feature number: "<< despR.rows << endl;
    FlannBasedMatcher matcher;
    vector<DMatch> matches;
    drawFeature(imgL, p1, "left features");
    drawFeature(imgR, p_keypointR, "right features");
    //stereo match
    matcher.match(tempDespL, despR, matches);
    vector<Point2f> tempGoodFeaturesL, tempGoodFeaturesR;
    //match stereo features and discard bad matches
    double maxDist = 0, minDist = 100;
    for(int i = 0; i < matches.size(); i++){
        double dist = matches[i].distance;
        if(dist < minDist) minDist = dist;
        if(dist > maxDist) maxDist = dist;
    }
    cout << "min dist: "<< minDist << "  max dist: " << maxDist << endl;
    // vector<DMatch> goodMatches;
    for(int i = 0; i < matches.size(); i++){
        cout << matches[i].distance << endl;
        if(matches[i].distance <= max(3.0*minDist, 0.2) &&

           fabs(p1[matches[i].queryIdx].y - 
                p_keypointR[matches[i].trainIdx].y) < 1.5 &&

           fabs(p1[matches[i].queryIdx].x - 
                p_keypointR[matches[i].trainIdx].x) < 130.0){

            tempGoodFeaturesL.push_back(p1[matches[i].queryIdx]);
            tempGoodFeaturesR.push_back(p_keypointR[matches[i].trainIdx]);
            img_pts.push_back(p2[matches[i].queryIdx]);
        }
    }
    cout << "stere point number: " << tempGoodFeaturesR.size() << endl;
    //obtain 3D points
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    float thres = 40*b;

    // closeScenePts.clear();
    // farScenePts.clear();
    farIdx.clear();

    for(int n = 0; n<tempGoodFeaturesL.size(); n++){
        Point3f pd;
        d = fabs(tempGoodFeaturesL[n].x - tempGoodFeaturesR[n].x);
        pd.x = b*(tempGoodFeaturesL[n].x - uc)/d;
        pd.y = b*(tempGoodFeaturesL[n].y - vc)/d;
        pd.z = b*f/d;
        obj_pts.push_back(pd);
        if(pd.z < thres){
            farIdx.push_back(0);
        }
        else{
            farIdx.push_back(1);
        }   
    }
    // drawFarandCloseFeatures(imgL, closeFeaturesL, farFeaturesL, "far close features");
    // drawMatch(imgL, p_keypointL, p_keypointR, 1, "stereo features");
}

void Frame::stereoMatchKLT(const vector<Point2f>& p1, //keypoint in the previous frame
                            const vector<Point2f>& p2, //keypoint in the current frame
                               vector<Point3f>& obj_pts,
                               vector<Point2f>& img_pts,
                               vector<int>& farIdx){
    vector<float> err;
    vector<uchar> status;
    Size winSize = Size(199,3);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);
    
    vector<Point2f> temp_p_keypointL, temp_p_keypointR;

    calcOpticalFlowPyrLK(imgL, imgR, 
                         p1,temp_p_keypointR,
                         status, err, winSize,
                         7, termcrit, OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

    //delete outliers from optical flow
    //========= select good matches================
    vector<int> inlierIdx;
    vector<Point2f> matched_p_keypointL, matched_p_keypointR;
    for(int n = 0; n < status.size(); n++){
        if((1 == status[n]) && 
           (err[n] < 0.1) &&
           (fabs(p1[n].y - temp_p_keypointR[n].y) < 1.5)){
            Point2f pt;

            pt = p1[n];
            matched_p_keypointL.push_back(pt);

            pt = temp_p_keypointR[n];
            matched_p_keypointR.push_back(pt);

            img_pts.push_back(p2[n]);
        }
    }

    //obtain 3D points
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    float thres = 40*b;

    // closeScenePts.clear();
    // farScenePts.clear();
    farIdx.clear();

    for(int n = 0; n<matched_p_keypointL.size(); n++){
        Point3f pd;
        d = fabs(matched_p_keypointL[n].x - matched_p_keypointR[n].x);
        pd.x = b*(matched_p_keypointL[n].x - uc)/d;
        pd.y = b*(matched_p_keypointL[n].y - vc)/d;
        pd.z = b*f/d;
        obj_pts.push_back(pd);
        if(pd.z < thres){
            farIdx.push_back(0);
        }
        else{
            farIdx.push_back(1);
        }   
    }
    scenePts = obj_pts;
    forDrawL = matched_p_keypointL;
    drawMatch(imgL, matched_p_keypointL, matched_p_keypointR, 1, "stereo features");
}

void Frame::matchFrame(Frame frame){
    // use FLANN matcher to match features between current frame and 
    // the previous frame.
    FlannBasedMatcher matcher;
    vector< DMatch > matches;
    matcher.match(despL, frame.despL, matches);
    
    //standard method to select good matches, provided by OpenCV
    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < despL.rows; i++ ){ 
        double dist = matches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    
    // vector<DMatch> good_matches;
    //obtain far and close points
    vector<Point2f> matched_prev, matched_curr;

    for( int i = 0; i < matches.size(); i++ ){
        if( matches[i].distance <= max(3.0*min_dist, 0.1) ){
            int qIdx, tIdx;
            qIdx = matches[i].queryIdx; // previous, used to determine depth
            tIdx = matches[i].trainIdx; // current
            //matched features, one-to-one correspondence
            matched_prev.push_back(p_keypointL[qIdx]);
            matched_curr.push_back(frame.p_keypointL[tIdx]);
        }
    }

    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;
    vector<int> farIdx;

    // cout << "matched feature number: " << matched_prev.size() << endl;
    // matchedNumWithCurrentFrame = matched_prev.size();
    stereoMatchKLT(matched_prev, //keypoint in the previous frame
                       matched_curr, //keypoint in the current frame
                       obj_pts, img_pts, farIdx);

    int farnb = 0, closenb = 0;
    vector<Point2f> farImg_pts, closeImg_pts;
    vector<Point3f> farObj_pts, closeObj_pts;
    for(int i = 0; i < farIdx.size(); i++){
        if(farIdx[i] == 1){
            farnb++;
            farImg_pts.push_back(img_pts[i]);
            farObj_pts.push_back(obj_pts[i]);
        }
        else{
            closenb++;
            closeImg_pts.push_back(img_pts[i]);
            closeObj_pts.push_back(obj_pts[i]);
        }
    }
    drawFarandCloseFeatures(frame.imgL, img_pts, farIdx, "far close");



    PnP(obj_pts, img_pts);
    // drawMatch(imgL, matched_prev, matched_curr, 1, "matched features");
    // drawMatch(imgL, keypointL, frame.keypointL, matches,1, "matched features");
    // drawFarandCloseFeatures(imgL, matched_close_img_pts, matched_far_img_pts, "far&close");

    // getMotion(good_matches, frame);
    
}

void Frame::PnP(vector<Point3f> obj_pts, 
                vector<Point2f> img_pts){

     //solve PnP
    Mat K = P2.colRange(0,3).clone();
    Mat inliers;
    // Mat temprvec, temptvec;
    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   rvec, tvec, false, 1000, 3.0, 30, inliers);
    cout << "PnP inliers: "<< inliers.rows << endl;
    matchedNumWithCurrentFrame = inliers.rows;

    // to display
    // vector<Point2f> mL,mR;

    // for(int i = 0; i < inliers.rows; i++){
    //     Point2f pt;
    //     int idx = inliers.at<int>(i, 0);
    //     pt = forDrawL[idx];
    //     mL.push_back(pt);
    //     pt = img_pts[idx];
    //     mR.push_back(pt);
    // }
    // drawMatch(imgL, mL, mR, 1, "PnP");


}

/*
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
    vector<DMatch> good_matches;
    vector<Point2f> matched_p_keypointL;
    vector<Point2f> matched_p_keypointR;
    vector<int> inlierIdx;
    matchFeature(despL, despR, 
                 inlierIdx, 
                 good_matches,
                 p_keypointL, p_keypointR,
                 matched_p_keypointL,
                 matched_p_keypointR);

    vector<int> tempInlierIdx;
    vector<Point2f> temp_m_p_kpL, temp_m_p_kpR;
    vector<DMatch> tempMatch;
    for(int i = 0; i < good_matches.size(); i++){
        if((fabs(matched_p_keypointL[i].y - 
                matched_p_keypointR[i].y) < 1.5) &&
            (fabs(matched_p_keypointL[i].x - 
                matched_p_keypointR[i].x < 150))){
            temp_m_p_kpL.push_back(matched_p_keypointL[i]);
            temp_m_p_kpR.push_back(matched_p_keypointR[i]);
            // tempInlierIdx.push_back(inlierIdx[i]);
            tempMatch.push_back(good_matches[i]);
        }
    }

    matched_p_keypointL = temp_m_p_kpL;
    matched_p_keypointR = temp_m_p_kpR;
    good_matches = tempMatch;

    // if(waitKey(100000) == 27){
    //     exit;
    // }
    /*
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
    // vector<KeyPoint> temp_keypointL, temp_keypointR;
    // KeyPoint::convert(temp_p_keypointL, temp_keypointL);
    // KeyPoint::convert(temp_p_keypointR, temp_keypointR);
    // drawMatch(imgL, temp_p_keypointL, temp_p_keypointR, 2, "stereo match");

    //====compute depth=======
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    float thres = 40*b;
    // cout << "far threshold is: " << thres << endl;
    //from paper: Vision meets Robotics: The KITTI Dataset
    //the format of P2 is:
    //  f_u,    0,    c_u,    -f_u*b_x
    //   0 ,  f_v,    c_v,        0
    //   0 ,    0,      1,        0
    // the origin is at the right camera, which is unusual

    //only compute 3D points for matched features
    
    vector<Point3f> obj_pts;//3D points
    // farScenePts.clear();
    vector<int> reIdx;
    for(int n = 0; n<matched_p_keypointL.size(); n++){
        Point3f pd;
        d = fabs(matched_p_keypointL[n].x - matched_p_keypointR[n].x);
        pd.x = b*(matched_p_keypointL[n].x - uc)/d;
        pd.y = b*(matched_p_keypointL[n].y - vc)/d;
        pd.z = b*f/d;
        if(pd.z < thres){
            obj_pts.push_back(pd);
            reIdx.push_back(1);
        }
        else{
            reIdx.push_back(0);
            farScenePts.push_back(pd);
        }
    }

    scenePts = obj_pts;
    //=====obtain tracked points in the second frame=====
    vector<Point2f> img_pts;//2D matched feature points in current image
    vector<Point2f> farImg_pts;
    //according to inliers previously in optical flow, inliers need to be
    //selected again here
    for(int n = 0; n < matched_p_keypointL.size(); n++){
        Point2f pt;
        if(reIdx[n] == 1){
            pt = matched_p_keypointL[n];
            img_pts.push_back(pt);
        }
        else{
            pt = matched_p_keypointL[n];
            farImg_pts.push_back(pt);
        }

    }
    cout << "far points: " << farImg_pts.size() << endl;
    cout << "close points: " << img_pts.size() << endl;
    cout << "total feature points: " << matched_p_keypointL.size() << endl;
    
    drawFeature(imgL, matched_p_keypointL, "left features");
    drawFeature(imgR, matched_p_keypointR, "right features");
    drawFarandCloseFeatures(imgL, img_pts, farImg_pts, "far & close");
    drawMatch(imgL, matched_p_keypointL, matched_p_keypointR, 2, "stereo match");
    // if(waitKey(100000) == 27){
    //     return;
    // }
    //solve PnP
    Mat K = P2.colRange(0,3).clone();
    Mat inliers;
    Mat temprvec, temptvec;
    //firstly use close points to estimate translation
    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   temprvec, temptvec, false, 200, 3.0, 40, inliers);
    tvec = temptvec;
    cout << "close inlier: " << inliers.rows << "   ";
    //then use far points to estimate rotation
    solvePnPRansac(farScenePts, farImg_pts, K, Mat(),
                   temprvec, temptvec, false, 200, 3.0, 30, inliers);
    rvec = temprvec;
    cout << "far inlier: " << inliers.rows << "   "<< endl;
    // cout << tvec << endl;
    // cout << rvec << endl;


    // if(waitKey(100000) == 27){
    //     return;
    // }
    //below is used for visulization
    vector<KeyPoint> mKeypoint1, mKeypoint2;
    KeyPoint kp;
    Point3f obj_pt;
    vector<Point3f> tempScenePts;
    // scenePts.clear();
    // scenePts.reserve(inliers.rows);
    /*
    vector<KeyPoint> k_img_pts;
    KeyPoint::convert(img_pts, k_img_pts);
    for(int i = 0; i < inliers.rows; i++){
        int idx = inliers.at<int>(i, 0);
        kp = temp_keypointL[idx];
        mKeypoint1.push_back(kp);
        kp = k_img_pts[idx];
        mKeypoint2.push_back(kp);
        obj_pt = obj_pts[idx];
        tempScenePts.push_back(obj_pt);
    }
    // scenePts = tempScenePts;
    // drawMatch(imgL, mKeypoint1, mKeypoint2, 2, "PnP inliers");
    // drawFeature(imgL, mKeypoint2, "PnP");
    
    
}


*/
void Frame::matchFeature(const Mat& desp1, const Mat& desp2, 
           vector<int>& inlierIdx,
           vector<DMatch>& good_matches,
           const vector<Point2f>& p_keypoint1, 
           const vector<Point2f>& p_keypoint2,
           vector<Point2f>& result_p_keypoint1,
           vector<Point2f>& result_p_keypoint2){
    
    FlannBasedMatcher matcher;
    vector<DMatch> matches_12;
    matcher.match(desp1, desp2, matches_12);
    inlierIdx.clear();

    double max_dist = 0, min_dist = 100;
    for(int i = 0; i < desp1.rows; i++){
        double dist = matches_12[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }

    for( int i = 0; i < desp1.rows; i++ ){
        if( matches_12[i].distance <= max(2*min_dist, 0.2) ){
            good_matches.push_back(matches_12[i]);
            inlierIdx.push_back(i);
        }
    }

    Point2f pt;
    for(int i = 0; i < good_matches.size(); i++){
        pt = p_keypoint1[good_matches[i].queryIdx];
        result_p_keypoint1.push_back(pt);
        pt = p_keypoint2[good_matches[i].trainIdx];
        result_p_keypoint2.push_back(pt);
    }
}












