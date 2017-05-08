#ifndef optimizer_hpp
#define optimizer_hpp

#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <time.h>
#include <string.h>
#include <dirent.h>
#include <fstream>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/factory.h>
#include <g2o/core/optimization_algorithm_factory.h>
#include <g2o/core/optimization_algorithm_gauss_newton.h>
#include <g2o/solvers/csparse/linear_solver_csparse.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_factory.h>
#include <g2o/core/optimization_algorithm_levenberg.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <opencv2/core/core.hpp>
#include "opencv2/calib3d/calib3d.hpp"

typedef g2o::BlockSolver_6_3 SlamBlockSolver; 
typedef g2o::LinearSolverCSparse< SlamBlockSolver::PoseMatrixType > SlamLinearSolver; 

using namespace std;
using namespace cv;

class Optimizer{
private:
	int fullTimes;
	int localTimes;  
    g2o::RobustKernel* robustKernel = g2o::RobustKernelFactory::instance()->construct( "Welsch" );

public:
	Optimizer(int _fullTimes, int _localTimes);
	void addNewNodeEdge(int prevId, int currId, Mat rvec, Mat tvec, bool newNode);
	void fullOptimize();
	void localOptimize();
	
    g2o::SparseOptimizer globalOptimizer;

};

Eigen::Isometry3d cvMat2Eigen( cv::Mat& rvec, cv::Mat& tvec );







#endif