//
// Created by gao on 2020/12/30.
//
#include <iostream>
#include <string>
#include <fstream>
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
//#include <opencv2/features2d.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <opencv2/core/base.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include <eigen3/Eigen/Core>
#include <time.h>
#include <stdlib.h>
#include "association.hpp"

#define IMAGE_NUM 45

using namespace std;


class FeatureTracker{
public:

    FeatureTracker(cv::Mat _img){
        _img.copyTo(imgPre);
    }

    bool Match(cv::Mat _curImg){

        vector<cv::KeyPoint> kp1, kp2;
        detector->detectAndCompute(imgPre, cv::Mat(), kp1, descPre);
        detector->detectAndCompute(_curImg, cv::Mat(), kp2, descCur);

        matcher.match(descPre, descCur, matches); //DMatch里边定义了小于号，比较distance，所以可以进行默认排序
        sort(matches.begin(), matches.end());
        int ptsPairs = min(30, (int)(matches.size()*0.2));//只要距离最小的前20%的点，且点的数量不超过30个
        good_matches.clear();
        for(int i = 0; i < ptsPairs; i++){
            good_matches.push_back(matches[i]);
        }

        refineMatch(kp1, kp2, good_matches);
        cv::Mat img_match;
        cv::drawMatches(imgPre, kp1, _curImg, kp2, good_matches, img_match);
        cv::imshow("raw_match", img_match);
        cv::waitKey(0);
        _curImg.copyTo(imgPre);
    }

private:

    void refineMatch(vector<cv::KeyPoint> &_kp1, vector<cv::KeyPoint> &_kp2, vector<cv::DMatch> &matches){
        vector<cv::KeyPoint> kp1_new, kp2_new;
        vector<cv::DMatch> matches_new;
        vector<cv::Point2f> p1, p2;

        for(int i = 0; i < matches.size(); i++){
            p1.emplace_back(_kp1[matches[i].queryIdx].pt.x, _kp1[matches[i].queryIdx].pt.y);
            p2.emplace_back(_kp2[matches[i].trainIdx].pt.x, _kp2[matches[i].trainIdx].pt.y);
        }
        cv::Mat mask;
        cv::findHomography(p1, p2, mask,cv::RANSAC, 2.0);

        int j = 0;
        for(int i = 0; i < mask.rows; i++){
            if(mask.at<uchar>(i, 0) == 1){
                kp1_new.push_back(_kp1[matches[i].queryIdx]);
                kp2_new.push_back(_kp2[matches[i].trainIdx]);
                matches_new.emplace_back(j, j, matches[i].imgIdx, matches[i].distance);
                j++;
            }
        }
        _kp1 = kp1_new;
        _kp2 = kp2_new;
        matches = matches_new;
    }


    int minHessian = 100;
    cv::Ptr<cv::xfeatures2d::SURF> detector = cv::xfeatures2d::SURF::create(minHessian, 4, 3, true, true);
    cv::BFMatcher matcher = cv::BFMatcher();
    std::vector<cv::DMatch> matches;
    vector<cv::DMatch> good_matches;

    //ransac parameters
    double ransac_threshold = 1.0;
    double inlier_ratio = 0.50;
    int max_iterations = 2000;
    cv::Mat imgPre;
    cv::Mat descPre, descCur;
    vector<Eigen::Vector3d> preFeatures;
};


int main(){
    string dir_path = "/home/gao/Dopper_radar/catkin_ws/src/beginner_tutorial/data/oxford_cartesian_image/img_";
    cv::Mat img_read, img;

    img_read = cv::imread(dir_path + to_string(0) + ".jpg");
    cv::cvtColor(img_read, img, cv::COLOR_BGR2GRAY);
    FeatureTracker tracker(img);

    for(int i = 1; i < IMAGE_NUM; i++){
        img_read = cv::imread(dir_path + to_string(i) + ".jpg");
        cv::cvtColor(img_read, img, cv::COLOR_BGR2GRAY);
        tracker.Match(img);
    }
}
