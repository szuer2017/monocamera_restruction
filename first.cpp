#include "opencv2/core/core.hpp"  
#include "opencv2/imgproc/imgproc.hpp"  
#include "opencv2/calib3d/calib3d.hpp"  
#include "opencv2/highgui/highgui.hpp"  
#include <opencv2/opencv.hpp>  
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/traits.hpp>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include <iostream>  
#include <fstream>
#include <pcl/visualization/cloud_viewer.h>
#include<pcl/point_types.h>  
#include <pcl/io/pcd_io.h>


using namespace cv;
using namespace std;


int main()
{
	Matx33d K(2484.3, 0, 653.6668,0, 2484.3, 517.0802,0, 0, 1);
	//Create SIFT class pointer
	Ptr<Feature2D> f2d = xfeatures2d::SIFT::create();
	//读入图片
	Mat img_1 = imread("o9.bmp");
	Mat img_2 = imread("o3.bmp");
	//Detect the keypoints
	vector<KeyPoint> keypoints_1, keypoints_2;
	f2d->detect(img_1, keypoints_1);
	f2d->detect(img_2, keypoints_2);
	//Calculate descriptors (feature vectors)
	Mat descriptors_1, descriptors_2;
	f2d->compute(img_1, keypoints_1, descriptors_1);
	f2d->compute(img_2, keypoints_2, descriptors_2);
	//Matching descriptor vector using BFMatcher
	BFMatcher matcher;
	vector<DMatch> matches;
	matcher.match(descriptors_1, descriptors_2, matches);

	int ptCount = (int)matches.size();

	Mat p1(ptCount, 2, CV_32F);
	Mat p2(ptCount, 2, CV_32F);
	// 把Keypoint转换为Mat


	Point2f pt;
	for (int i = 0; i<ptCount; i++)
	{
		pt = keypoints_1[matches[i].queryIdx].pt;
		p1.at<float>(i, 0) = pt.x;
		p1.at<float>(i, 1) = pt.y;


		pt = keypoints_2[matches[i].trainIdx].pt;
		p2.at<float>(i, 0) = pt.x;
		p2.at<float>(i, 1) = pt.y;
	}
	Mat F;
	F = findFundamentalMat(p1, p2, FM_RANSAC);
	cout << "基础矩阵为：" << F << endl;
	//绘制匹配出的关键点
	Mat img_matches;
	drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_matches);
	namedWindow("【match图】", 0);
	resizeWindow("【match图】", 1280, 480);
	imshow("【match图】", img_matches);

	Mat_<double> E = Mat(K.t()) * F * Mat(K);
	SVD svd(E);
	Matx33d W(0, -1, 0,//HZ 9.13  
		1, 0, 0,
		0, 0, 1);
	Mat_<double> R = svd.u * Mat(W) * svd.vt; //HZ 9.19  
	Mat_<double> t = svd.u.col(2); //u3 

	Matx34d P(1, 0, 0, 0,
		0, 1, 0, 0,
		0, 0, 1, 0);
	Matx34d P1(R(0, 0), R(0, 1), R(0, 2), t(0),
		R(1, 0), R(1, 1), R(1, 2), t(1),
		R(2, 0), R(2, 1), R(2, 2), t(2));

	//cout <<P << P1 << endl;
	int points_size = (int)keypoints_1.size();
	vector<KeyPoint> points_1, points_2;
	int ptCount_1 = ptCount;
	points_1 = keypoints_1;
	points_2 = keypoints_2;

	Mat p_1(ptCount_1, 2, CV_32F);
	Mat p_2(ptCount_1, 2, CV_32F);
	// 把Keypoint转换为Mat
	for (int i = 0; i<ptCount_1; i++)
	{
		pt = points_1[matches[i].queryIdx].pt;
		p_1.at<float>(i, 0) = pt.x;
		p_1.at<float>(i, 1) = pt.y;


		pt = points_2[matches[i].trainIdx].pt;
		p_2.at<float>(i, 0) = pt.x;
		p_2.at<float>(i, 1) = pt.y;
	}




	Mat Kinv;
	Kinv = invert(K, Kinv);
	MatSize length = Kinv.size;


	vector<Point3d> u_1, u_2;
	for (int i = 0; i < ptCount_1; i++)
	{
		u_1.push_back(Point3f(p_1.at<float>(i, 0), p_1.at<float>(i, 1), 1));
		u_2.push_back(Point3f(p_2.at<float>(i, 0), p_2.at<float>(i, 1), 1));


	}
	Mat_<double> LinearLSTriangulation(
		Point3d u,//homogenous image point (u,v,1)  
		Matx34d P,//camera 1 matrix  
		Point3d u1,//homogenous image point in 2nd camera  
		Matx34d P1//camera 2 matrix  
		);
	
	vector<Point3d> pointcloud;
	Mat_<double> X_1, X;
	for (int i = 0; i < ptCount_1; i++)
	{
		X_1 = LinearLSTriangulation(u_1[i], P, u_2[i], P1);
		X.push_back(X_1);
		pointcloud.push_back(Point3d(X_1(0), X_1(1), X_1(2)));

	}
	cout << "点云结果为：" << pointcloud << endl;
	

	pcl::PointCloud<pcl::PointXYZ> pointcloud1;
	pointcloud1.width = ptCount_1;
	pointcloud1.height = 1;
	pointcloud1.is_dense = false;
	pointcloud1.points.resize(pointcloud1.width * pointcloud1.height);
	for (size_t i = 0; i < ptCount_1; i++)
	{
		pointcloud1[i].x = pointcloud[i].x;
		pointcloud1[i].y = pointcloud[i].y;
		pointcloud1[i].z = pointcloud[i].z;

	}
	pcl::io::savePCDFileASCII("pointcloud.pcd", pointcloud1);


	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>); // 创建点云（指针）  
	pcl::io::loadPCDFile("pointcloud.pcd", *cloud1);
	pcl::visualization::PCLVisualizer viewer;
	cout << "点云结果如图所示" << endl;
	viewer.addPointCloud(cloud1, "cloud1");
	viewer.spin();


	system("pause");

}
Mat_<double> LinearLSTriangulation(
	Point3d u,//homogenous image point (u,v,1)  
	Matx34d P,//camera 1 matrix  
	Point3d u1,//homogenous image point in 2nd camera  
	Matx34d P1//camera 2 matrix  
	)
{
	//build A matrix  
	Matx43d A(u.x*P(2, 0) - P(0, 0), u.x*P(2, 1) - P(0, 1), u.x*P(2, 2) - P(0, 2),
		u.y*P(2, 0) - P(1, 0), u.y*P(2, 1) - P(1, 1), u.y*P(2, 2) - P(1, 2),
		u1.x*P1(2, 0) - P1(0, 0), u1.x*P1(2, 1) - P1(0, 1), u1.x*P1(2, 2) - P1(0, 2),
		u1.y*P1(2, 0) - P1(1, 0), u1.y*P1(2, 1) - P1(1, 1), u1.y*P1(2, 2) - P1(1, 2)
		);
	//build B vector  
	Matx41d B(-(u.x*P(2, 3) - P(0, 3)),
		-(u.y*P(2, 3) - P(1, 3)),
		-(u1.x*P1(2, 3) - P1(0, 3)),
		-(u1.y*P1(2, 3) - P1(1, 3)));
	//solve for X  
	Mat_<double> X;
	solve(A, B, X, DECOMP_SVD);
	return X;
}
