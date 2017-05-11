//
//  frame.cpp
//  sceneReconstruct
//
//  Created by 周晓宇 on 4/3/17.
//  Copyright © 2017 xiaoyu. All rights reserved.
//

#include "frame.hpp"


Frame::Frame(string filenameL, string filenameR, Mat _P1, Mat _P2, int id){
    imgL = imread(filenameL);
    imgR = imread(filenameR);
    P1 = _P1;
    P2 = _P2;
    frameID = id;
    Mat imgLgray, imgRgray;
    //if use Harris or anyother corner detector, gray-scale images are needed
    // cvtColor( imgL  , imgLgray, CV_BGR2GRAY );
    // cvtColor( imgR  , imgRgray, CV_BGR2GRAY );

    //Gaussian blur image, preprocessing
    if(!imgL.data || ! imgR.data){
        cout << "image does not exist..." << endl;
        exit;
    }

    // GaussianBlur(imgL, imgL, Size(7,7), 0.5);
    // GaussianBlur(imgR, imgR, Size(7,7), 0.5);
    /*
    ORBextractor* detector = new ORBextractor(1000,1.2,3,20,10);

    (*detector)(imgLgray, Mat(), keypointL, despL);
    (*detector)(imgRgray, Mat(), keypointR, despR);

    KeyPoint::convert(keypointL, p_keypointL);
    KeyPoint::convert(keypointR, p_keypointR);*/


    SurfFeatureDetector detector(10);
    SurfDescriptorExtractor descriptor;
    
    detector.detect(imgL, keypointL);
    detector.detect(imgR, keypointR);
    
    KeyPoint::convert(keypointL, p_keypointL);
    KeyPoint::convert(keypointR, p_keypointR);
    
    descriptor.compute(imgL, keypointL, despL);
    descriptor.compute(imgR, keypointR, despR);

    //create kd-tree for keypoints in the left image.
    /*
    PointCloud<PointXY>::Ptr cloud (new PointCloud<PointXY>);
    cloud->height = 1;
    cloud->width = p_keypointL.size();
    cloud->points.resize(cloud->height * cloud->width);
    for(int n = 0; n < p_keypointL.size(); n++){
        cloud->points[n].x = p_keypointL[n].x;
        cloud->points[n].y = p_keypointL[n].y;
    }

    kdtree_keypointL.setInputCloud(cloud);*/

    // drawFeature(imgL, p_keypointL, "features");
}

Frame::Frame(string filenameL, Mat _P1, int id){
    imgL = imread(filenameL);
    P1 = _P1;
    frameID = id;
    Mat imgLgray;
    cvtColor( imgL  , imgLgray, CV_BGR2GRAY );

    ORBextractor* detector = new ORBextractor(1000,1.1,6,15,5);

    (*detector)(imgLgray, Mat(), keypointL, despL);

    KeyPoint::convert(keypointL, p_keypointL);

    SurfDescriptorExtractor descriptor;
    descriptor.compute(imgL, keypointL, despL);
    //create kd-tree for keypoints in the left image.
    PointCloud<PointXY>::Ptr cloud (new PointCloud<PointXY>);
    cloud->height = 1;
    cloud->width = p_keypointL.size();
    cloud->points.resize(cloud->height * cloud->width);
    for(int n = 0; n < p_keypointL.size(); n++){
        cloud->points[n].x = p_keypointL[n].x;
        cloud->points[n].y = p_keypointL[n].y;
    }

    kdtree_keypointL.setInputCloud(cloud);
    drawFeature(imgL, p_keypointL, "features");
}

//once set to keyframe, only store points that can be found depth
void Frame::setKeyframe(string filenameR, Mat _P2){
    imgR = imread(filenameR);
    P2 = _P2;

    vector<float> err;
    vector<uchar> status;
    Size winSize = Size(199,3);
    TermCriteria termcrit(TermCriteria::COUNT|TermCriteria::EPS,50,0.01);

    calcOpticalFlowPyrLK(imgL, imgR, 
                         p_keypointL,p_keypointR,
                         status, err, winSize,
                         7, termcrit, OPTFLOW_LK_GET_MIN_EIGENVALS, 0.001);

    //delete outliers from optical flow
    //========= select good matches================
    vector<Point2f> matched_p_keypointL, matched_p_keypointR;
    for(int n = 0; n < status.size(); n++){
        if((1 == status[n]) && 
           (err[n] < 0.1)&&
           (fabs(p_keypointL[n].y - p_keypointR[n].y) < 1.5)){
            Point2f pt;

            pt = p_keypointL[n];
            matched_p_keypointL.push_back(pt);

            pt = p_keypointR[n];
            matched_p_keypointR.push_back(pt);
        }
    }
    p_keypointL = matched_p_keypointL;
    p_keypointR = matched_p_keypointR;

    //=========obtain 3D points====================
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    float thres = 40*b;

    farPtsIdx.clear();
    farPtsIdx.reserve(p_keypointL.size());

    for(int n = 0; n<p_keypointL.size(); n++){
        Point3f pd;
        d = fabs(p_keypointL[n].x - p_keypointR[n].x);
        pd.x = b*(p_keypointL[n].x - uc)/d;
        pd.y = b*(p_keypointL[n].y - vc)/d;
        pd.z = b*f/d;
        scenePts.push_back(pd);
        if(pd.z < thres){
            farPtsIdx.push_back(false);
        }
        else{
            farPtsIdx.push_back(true);
        }   
    }
    //========initialize rvec and tvec to zeros==============
    rvec = Mat::zeros(3,1,CV_64F);
    tvec = Mat::zeros(3,1,CV_64F);
    drawMatch(imgL, p_keypointL, p_keypointR, 1, "stereo features");

}

void Frame::matchFrameNN(Frame frame){
    //===project current frame's 3D points to the new frame
    //  1. project features to the new frame
    vector<Point2f> proj_p_keypointL;
    projectPoints(scenePts, rvec, tvec, P1.colRange(0,3), Mat(), proj_p_keypointL);
    drawFeature(frame.imgL, proj_p_keypointL, "projected features");
    //  2. delete out-of-boundary points
    vector<bool> inBoundIdx(proj_p_keypointL.size(), false);
    for(int n = 0; n < proj_p_keypointL.size(); n++){
        Point2f pt = proj_p_keypointL[n];
        if(pt.x > 0 && pt.x < imgL.cols&&
           pt.y > 0 && pt.y < imgL.rows){
            inBoundIdx[n] = true;
        }
    }
    // 3. search NN for projected in bound features
    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;
    vector<bool> curFarIdx;
    vector<Point2f> matchedPrev, matchedCurr;
    //============try kdtree match=================
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv:: DescriptorExtractor::create( "SURF" );
    for(int n = 0; n < proj_p_keypointL.size(); n++){

        if(true == inBoundIdx[n]){
            // BFMatcher matcher(NORM_HAMMING);
            FlannBasedMatcher matcher;
            vector<DMatch> matches;
            //search around projected points
            PointXY searchPt;
            searchPt.x = proj_p_keypointL[n].x;
            searchPt.y = proj_p_keypointL[n].y;

            vector<int> pointIdxRadiusSearch;
            vector<float> pointRadiusSquaredDistance;

            if(frame.kdtree_keypointL.radiusSearch (searchPt, 
                                              200, 
                                              pointIdxRadiusSearch, 
                                              pointRadiusSquaredDistance) > 0 ){
             
                //get descriptors for matched features
                // vector<Mat> tempDesp;
                // for(int i = 0; i < pointIdxRadiusSearch.size(); i++){
                //     Mat td;
                //     td = frame.despL.row(pointIdxRadiusSearch[i]).clone();
                //     tempDesp.push_back(td);
                // }
                //==============================================
                vector<KeyPoint> tempkeypoint;
                for(int k = 0; k < pointIdxRadiusSearch.size(); k++){
                    tempkeypoint.push_back(frame.keypointL[pointIdxRadiusSearch[k]]);
                }
                Mat surroundKeypointDesp;// = Mat(tempDesp, true);
                Mat imgGray;
                // cvtColor( frame.imgL  , imgGray, CV_BGR2GRAY );
                descriptor->compute(frame.imgL, tempkeypoint, surroundKeypointDesp);
                //==============================================


                matcher.match(despL.row(n), surroundKeypointDesp, matches);
                if(matches.size() > 0){
                    //find the original index
                    int originalIdx = pointIdxRadiusSearch[matches[0].trainIdx];
                    //get obj_pts, img_pts
                    obj_pts.push_back(scenePts[n]);
                    img_pts.push_back(frame.p_keypointL[originalIdx]);
                    curFarIdx.push_back(farPtsIdx[n]);
                    matchedPrev.push_back(p_keypointL[n]);
                    matchedCurr.push_back(frame.p_keypointL[originalIdx]);
                }
            }
        }
    }
    drawMatch(imgL, matchedPrev, matchedCurr, 1, "matched features");
    // 4. solve PnP
    int farnb = 0, closenb = 0;
    vector<Point2f> farImg_pts, closeImg_pts;
    vector<Point3f> farObj_pts, closeObj_pts;
    for(int i = 0; i < curFarIdx.size(); i++){
        if(curFarIdx[i] == true){
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
    // drawFarandCloseFeatures(frame.imgL, img_pts, curFarIdx, "far close");
    Mat inliers;
    PnP(obj_pts, img_pts, inliers);
    cout << "PnP inliers: " << matchedNumWithCurrentFrame << endl;

    vector<Point2f> finalP1, finalP2;
    for(int n = 0; n < inliers.rows; n++){
        finalP1.push_back(matchedPrev[inliers.at<int>(n,0)]);
        finalP2.push_back(matchedCurr[inliers.at<int>(n,0)]);
    }
    drawMatch(imgL, finalP1, finalP2, 1, "pnp inliers");
    // 5. update translation and rotation
        //updating done in PnP
    // drawMatch(imgL, matchedPrev, matchedCurr, 1, "matched features");
    // drawMatch(imgL, keypointL, frame.keypointL, matches,1, "matched features");
    // drawFarandCloseFeatures(imgL, matched_close_img_pts, matched_far_img_pts, "far&close");

}

void Frame::matchFrame(Frame frame){
    // use FLANN matcher to match features between current frame and 
    // the previous frame.
    vector<KeyPoint> matchedPrev, matchedCurr;
    vector<DMatch> matches;
    matchFeatureKNN(despL, frame.despL, 
                    keypointL, frame.keypointL,
                    matchedPrev, matchedCurr,
                    matches, 0.6);

    vector<Point3f> obj_pts;
    vector<Point2f> img_pts;
    vector<int> farIdx;
    // cout << "matched feature number: " << matchedPrev.size() << endl;
    // matchedNumWithCurrentFrame = matchedPrev.size();
    stereoMatchFeature(matchedPrev, //keypoint in the previous frame
                       matchedCurr, //keypoint in the current frame
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
    scenePts = closeObj_pts;
    // drawFarandCloseFeatures(frame.imgL, img_pts, farIdx, "far close");
    Mat inliers;
    PnP(obj_pts, img_pts, inliers);
    drawMatch(imgL, matchedPrev, matchedCurr, 1, "matched features");
    cout << "PnP inliers: " << img_pts.size() << endl;
    vector<Point2f> finalP1, finalP2;
    for(int n = 0; n < inliers.rows; n++){
        finalP1.push_back(matchedPrev[inliers.at<int>(n,0)].pt);
        finalP2.push_back(matchedCurr[inliers.at<int>(n,0)].pt);
    }
    drawMatch(imgL, finalP1, finalP2, 1, "pnp inliers");    
}

void Frame::PnP(vector<Point3f> obj_pts, 
                vector<Point2f> img_pts,
                Mat& inliers){

     //solve PnP
    Mat K = P2.colRange(0,3).clone();
    // Mat temprvec, temptvec;
    if(obj_pts.size() == 0){
        cout << "points for PnP is 0, cannot solve." << endl;
        success = false;
        return;
    }
    solvePnPRansac(obj_pts, img_pts, K, Mat(),
                   rvec, tvec, false, 2000, 3.0, 90, inliers);
    // cout << "PnP inliers: "<< inliers.rows << endl;
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



void Frame::stereoMatchFeature(vector<KeyPoint>& p1, //keypoint usd to get obj_pts
                               vector<KeyPoint>& p2, //keypoint usd to get img_pts
                               vector<Point3f>& obj_pts,
                               vector<Point2f>& img_pts,
                               vector<int>& farIdx){
    obj_pts.clear();
    img_pts.clear();
    farIdx.clear();

    vector<KeyPoint> tGoodFeaturesLK, tGoodFeaturesRK;
    // vector<KeyPoint
    Mat tempDespL;

    SurfDescriptorExtractor descriptor;
    descriptor.compute(imgL, p1, tempDespL);

    vector<DMatch> matches;
    matchFeatureKNN(tempDespL, despR, 
                    p1, keypointR,
                    tGoodFeaturesLK,
                    tGoodFeaturesRK,
                    matches, 0.6);

    vector<Point2f> tGoodFeaturesL, tGoodFeaturesR;

    KeyPoint::convert(tGoodFeaturesLK, tGoodFeaturesL);
    KeyPoint::convert(tGoodFeaturesRK, tGoodFeaturesR);

    // drawFeature(imgL, p1, "left");
    // drawFeature(imgR, p_keypointR, "right");
    // drawMatch(imgL, tempGoodFeaturesL, tempGoodFeaturesR, 1, "stereo features");
    // if(waitKey(100000) == 27){};

    //obtain obj_pts
    float uc = P2.at<float>(0,2);
    float vc = P2.at<float>(1,2);
    float f = P2.at<float>(0,0);
    float b = -P2.at<float>(0,3)/f;
    float d = 0;
    float thres = 40*b;

    for(int n = 0; n<tGoodFeaturesL.size(); n++){
        Point3f pd;
        d = fabs(tGoodFeaturesL[n].x - tGoodFeaturesR[n].x);
        pd.x = b*(tGoodFeaturesL[n].x - uc)/d;
        pd.y = b*(tGoodFeaturesL[n].y - vc)/d;
        pd.z = b*f/d;
        obj_pts.push_back(pd);
        if(pd.z < thres){
            farIdx.push_back(0);
        }
        else{
            farIdx.push_back(1);
        }   
    }

    //obtain img_pts
    for(auto m:matches){
        img_pts.push_back(p2[m.queryIdx].pt);
    }
}

void symmetryTest(const std::vector<cv::DMatch> &matches1,const std::vector<cv::DMatch> &matches2,std::vector<cv::DMatch>& symMatches)
{
    symMatches.clear();
    for (vector<DMatch>::const_iterator matchIterator1= matches1.begin();matchIterator1!= matches1.end(); ++matchIterator1)
    {
        for (vector<DMatch>::const_iterator matchIterator2= matches2.begin();matchIterator2!= matches2.end();++matchIterator2)
        {
            if ((*matchIterator1).queryIdx ==(*matchIterator2).trainIdx &&(*matchIterator2).queryIdx ==(*matchIterator1).trainIdx)
            {
                symMatches.push_back(DMatch((*matchIterator1).queryIdx,(*matchIterator1).trainIdx,(*matchIterator1).distance));
                break;
            }
        }
    }
}

void Frame::matchFeature(const Mat& desp1, const Mat& desp2, 
                         const vector<Point2f>& p_keypoint1, 
                         const vector<Point2f>& p_keypoint2,
                         vector<Point2f>& matchedKeypoint1,
                         vector<Point2f>& matchedKeypoint2){

    matchedKeypoint1.clear();
    matchedKeypoint2.clear();
    float imgThres = 0.2 * sqrt(pow(imgL.rows, 2)+pow(imgL.cols, 2));

    FlannBasedMatcher matcher;
    vector<DMatch> matches1to2, matches2to1;
    matcher.match(desp1, desp2, matches1to2);
    matcher.match(desp2, desp1, matches2to1);

    vector<DMatch> mutualMatches;
    symmetryTest(matches1to2, matches2to1, mutualMatches);

    double max_dist = 0; double min_dist = 100;
    for( int i = 0; i < mutualMatches.size(); i++ ){ 
        double dist = mutualMatches[i].distance;
        if( dist < min_dist ) min_dist = dist;
        if( dist > max_dist ) max_dist = dist;
    }
    for( int i = 0; i < mutualMatches.size(); i++ ){

        Point2f pt1 = p_keypoint1[mutualMatches[i].queryIdx];
        Point2f pt2 = p_keypoint2[mutualMatches[i].trainIdx];

        float ptdist = sqrt(pow(pt1.x-pt2.x, 2) + pow(pt1.y-pt2.y, 2));

        if( mutualMatches[i].distance <= max(3.0*min_dist, 0.1) &&
            ptdist < imgThres){
            int qIdx, tIdx;
            qIdx = mutualMatches[i].queryIdx; // previous, used to determine depth
            tIdx = mutualMatches[i].trainIdx; // current
            //matched features, one-to-one correspondence
            matchedKeypoint1.push_back(p_keypoint1[qIdx]);
            matchedKeypoint2.push_back(p_keypoint2[tIdx]);
        }
    }
}


void Frame::matchFeatureKNN(const Mat& desp1, const Mat& desp2, 
                            const vector<KeyPoint>& keypoint1, 
                            const vector<KeyPoint>& keypoint2,
                            vector<KeyPoint>& matchedKeypoint1,
                            vector<KeyPoint>& matchedKeypoint2,
                            vector<DMatch>& matches,
                            double knn_match_ratio){

    matchedKeypoint1.clear();
    matchedKeypoint2.clear();
    matches.clear();

    float imgThres = 0.2 * sqrt(pow(imgL.rows, 2)+pow(imgL.cols, 2));

    cv::Ptr<cv::DescriptorMatcher>  matcher = cv::DescriptorMatcher::create("BruteForce");

    vector< vector<cv::DMatch> > matches_knn;
    matcher->knnMatch( desp1, desp2, matches_knn, 2 );
    vector< cv::DMatch > tMatches;

    for ( size_t i=0; i<matches_knn.size(); i++ )
    {
        if (matches_knn[i][0].distance < knn_match_ratio * matches_knn[i][1].distance )
            tMatches.push_back( matches_knn[i][0] );
    }
    
    if (tMatches.size() <= 20) //too few matches
        return;

    vector<KeyPoint> tMatchedKeypoint1, tMatchedKeypoint2;
    for ( auto m:tMatches )
    {
        Point2f pt1, pt2;
        pt1 = keypoint1[m.queryIdx].pt;
        pt2 = keypoint2[m.trainIdx].pt;
        float ptdist = sqrt(pow(pt1.x-pt2.x, 2) + pow(pt1.y-pt2.y, 2));
        if(ptdist < imgThres){
            matchedKeypoint1.push_back(keypoint1[m.queryIdx]);
            matchedKeypoint2.push_back(keypoint2[m.trainIdx]);
            matches.push_back(m);
        } 
    }  
}

void Frame::releaseMemory(){
    imgL.release();
    imgR.release();
    despL.release();
    despR.release();

}








