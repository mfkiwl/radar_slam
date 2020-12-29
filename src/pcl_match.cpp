//
// Created by gao on 2020/12/28.
//
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include <eigen3/Eigen/Core>
#include <queue>
#include "association.hpp"
//#include <opencv2/imgproc.hpp>

#define HIGHT 640
#define WIDTH 480

using namespace std;




vector<Eigen::Vector3d> read_csv(string path){
    vector<Eigen::Vector3d> pcl;
    ifstream fin(path);
    string line;
    getline(fin, line);
    while(getline(fin, line)){
        istringstream sin(line);
        Eigen::Vector3d point;
        vector<double> fields;
        string field;
        while(getline(sin, field, ',')){
            fields.push_back(atof(field.c_str()));
        }
        point.x() = fields[0];
        point.y() = fields[1];
        point.z() = fields[2];
        pcl.push_back(point);
    }
    return pcl;
}

cv::Mat Points2Image(vector<Eigen::Vector3d> _pcl){
    cv::Mat img = cv::Mat::zeros(HIGHT, WIDTH, CV_8UC3);
    int x,y;
    cv::Point2f center;
    for(int i = 0; i < _pcl.size(); i++){
        x = (int)(_pcl[i].x() + WIDTH/2);
        y = (int)(_pcl[i].y() + HIGHT/2);
        center = cv::Point(x, y);
        cv::circle(img, center, 2, cv::Scalar(0,255,255), cv::FILLED);
    }
    cv::imshow("pcl_img", img);
    cv::waitKey(0);
    return img;
}


bool AddKeyPoints(std::vector<cv::KeyPoint>& _kp, const vector<Eigen::Vector3d> _pcl, int patch_size){
    double u,v;
    for(int i = 0; i < _pcl.size(); i++){
        u = _pcl[i].x() + WIDTH/2;
        v = _pcl[i].y() + HIGHT/2;
        _kp.push_back(cv::KeyPoint(u,v,patch_size));
    }
    return true;
}


int main(int argc, char** argv){
    std::string datadir = "/home/gao/Dopper_radar/catkin_ws/src/beginner_tutorial/data";
    std::vector<std::string> radar_files;
    string radar_path1 = "/home/gao/Dopper_radar/catkin_ws/src/beginner_tutorial/data/pcl_0.csv";
    string radar_path2 = "/home/gao/Dopper_radar/catkin_ws/src/beginner_tutorial/data/pcl_1.csv";

    vector<Eigen::Vector3d> pcl_1, pcl_2;
    cv::Mat img1, img2;
    pcl_1 = read_csv(radar_path1);
    pcl_2 = read_csv(radar_path2);

    img1 = Points2Image(pcl_1);
    img2 = Points2Image(pcl_2);

    int patch_size = 2;
    cv::Mat desc1, desc2;//descriptor
    std::vector<cv::KeyPoint> kp1, kp2;//key points
    AddKeyPoints(kp1, pcl_1, patch_size);
    AddKeyPoints(kp2, pcl_2, patch_size);

    cv::Ptr<cv::ORB> detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    detector->compute(img1, kp1, desc1);
    detector->compute(img2, kp2, desc2);

    std::vector<std::vector<cv::DMatch>> knn_matches;//queryIdx indicate the first image, trainIdx indicate the second image
    matcher->knnMatch(desc1, desc2, knn_matches, 1);

//    float nndr = 0.80;
    std::vector<cv::DMatch> good_matches;
    for (uint j = 0; j < knn_matches.size(); ++j) {
        if (!knn_matches[j].size())
            continue;
        if (knn_matches[j][0].distance == 0) {
            good_matches.push_back(knn_matches[j][0]); // distance the smaller the better
        }
    }

    cv::Mat img_match;
    cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_match);
    cv::imshow("match_img", img_match);
    cv::waitKey(0);
    std::cout << "hello world" << std::endl;

    return 0;
}
