#include "optimizer.hpp"


Optimizer::Optimizer(int _fullTimes, int _localTimes){
	fullTimes = _fullTimes;
	localTimes = _localTimes;

	SlamLinearSolver* linearSolver = new SlamLinearSolver();
    linearSolver->setBlockOrdering( false );
    SlamBlockSolver* blockSolver = new SlamBlockSolver( linearSolver );
    g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg( blockSolver );
    //==============================================
    globalOptimizer.setAlgorithm( solver ); //the one will be used all the time
    globalOptimizer.setVerbose( false );

    //add the first vertex
    g2o::VertexSE3* v = new g2o::VertexSE3();
    v->setId(0);
    v->setEstimate(Eigen::Isometry3d::Identity());
    v->setFixed( true ); //fix the first vertex
    globalOptimizer.addVertex(v);

}

void Optimizer::addNewNodeEdge(int prevId, int currId, Mat rvec, Mat tvec){
	Eigen::Isometry3d T = cvMat2Eigen(rvec, tvec);

	//add new node
	g2o::VertexSE3 *v = new g2o::VertexSE3();
    v->setId( currId );
    v->setEstimate( Eigen::Isometry3d::Identity() );
    globalOptimizer.addVertex(v);

    //add edge
    g2o::EdgeSE3* edge = new g2o::EdgeSE3();
  	//id between connected nodes
    edge->vertices() [0] = globalOptimizer.vertex(currId);
    edge->vertices() [1] = globalOptimizer.vertex(prevId);
    edge->setRobustKernel( robustKernel );

    //information matrix
    Eigen::Matrix<double, 6, 6> information = Eigen::Matrix< double, 6,6 >::Identity();
    // 信息矩阵是协方差矩阵的逆，表示我们对边的精度的预先估计
    // 因为pose为6D的，信息矩阵是6*6的阵，假设位置和角度的估计精度均为0.1且互相独立
    // 那么协方差则为对角为0.01的矩阵，信息阵则为100的矩阵
    information(0,0) = information(1,1) = information(2,2) = 100;
    information(3,3) = information(4,4) = information(5,5) = 100;
    // 也可以将角度设大一些，表示对角度的估计更加准确
    edge->setInformation( information );
    // estimation of edge is the result of PnP
    edge->setMeasurement(T.inverse());
    // 将此边加入图中
    globalOptimizer.addEdge(edge);

}

void Optimizer::fullOptimize(){
	cout << "optimizing pose graph, vertice numbers: " << globalOptimizer.vertices().size() << endl;
	globalOptimizer.save("../pose_graph/before_full_optimize.g2o");
	globalOptimizer.initializeOptimization();
	globalOptimizer.optimize(fullTimes);
	globalOptimizer.save("../pose_graph/after_full_optimize.g2o");
	cout<<"Optimization done."<<endl;
}

void Optimizer::localOptimize(){



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