#include <iostream>
#include <string>
//#include <fstream>
#include <vector>
#include <eigen3/Eigen/Core>
#include <chrono>
#include <opencv2/core.hpp>
//#include <opencv2/highgui.hpp>
//#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include "radar_utils.hpp"
#include "features.hpp"
#include "association.hpp"
#include <ros/ros.h>
#include <geometry_msgs/Point32.h>
#include <sensor_msgs/PointCloud.h>


std::string root = "/home/gao/dataset/Oxford_Radar/oxford_radar_robotcar_dataset_sample_small/";
std::string sequence = "2019-01-10-14-36-48-radar-oxford-10k-partial";

std::string datadir = root + sequence + "/radar";
std::string gt = root + sequence + "/gt/radar_odometry.csv";

int min_range = 58;                 // min range of radar points (bin)
float radar_resolution = 0.0432;    // resolution of radar bins in meters per bin
float cart_resolution = 0.2592;     // meters per pixel
int cart_pixel_width = 964;         // height and width of cartesian image in pixels
bool interp = true;
int keypoint_extraction = 0;        // 0: cen2018, 1: cen2019, 2: orb
// cen2018 parameters
float zq = 3.0;
int sigma_gauss = 17;
// cen2019 parameters
int max_points = 10000;
// ORB descriptor / matching parameters
int patch_size = 21;                // width of patch in pixels in cartesian radar image
float nndr = 0.80;                  // Nearest neighbor distance ratio
// RANSAC
double ransac_threshold = 0.35;
double inlier_ratio = 0.90;
int max_iterations = 100;
// MDRANSAC
int max_gn_iterations = 10;
double md_threshold = pow(ransac_threshold, 2);
// DOPPLER
double beta = 0.049;

std::vector<std::string> radar_files;
std::ofstream ofs;

cv::Mat img1, img2, desc1, desc2;//descriptor
std::vector<cv::KeyPoint> kp1, kp2;//key points
Eigen::MatrixXd targets, cart_targets1, cart_targets2;
std::vector<int64_t> t1, t2;

std::vector<int64_t> times;
std::vector<double> azimuths;
std::vector<bool> valid;
cv::Mat fft_data;

cv::Ptr<cv::ORB> detector;
cv::Ptr<cv::DescriptorMatcher> matcher;



void PclCallback(const sensor_msgs::PointCloudConstPtr &pcl_msg){

    std::vector<Eigen::Vector3d> pcl;
    for(int i = 0; i < pcl_msg->points.size(); i++){
        Eigen::Vector3d point;
        point.x() = pcl_msg->points[i].x;
        point.y() = pcl_msg->points[i].y;
        point.z() = pcl_msg->points[i].z;
        pcl.push_back(point);
    }


    for (uint i = 0; i < radar_files.size() - 1; ++i) {
        std::cout << i << "/" << radar_files.size() << std::endl;
        if (i > 0) {
            t1 = t2; desc1 = desc2.clone(); cart_targets1 = cart_targets2;
            kp1 = kp2; img2.copyTo(img1);
        }
        load_radar(datadir + "/" + radar_files[i], times, azimuths, valid, fft_data);
        if (keypoint_extraction == 0)
            cen2018features(fft_data, zq, sigma_gauss, min_range, targets);
        if (keypoint_extraction == 1)
            cen2019features(fft_data, max_points, min_range, targets);
        if (keypoint_extraction == 0 || keypoint_extraction == 1) {
            radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, cart_resolution, cart_pixel_width, interp, img2, CV_8UC1);  // NOLINT
            polar_to_cartesian_points(azimuths, times, targets, radar_resolution, cart_targets2, t2);
            convert_to_bev(cart_targets2, cart_resolution, cart_pixel_width, patch_size, kp2, t2);
            // cen2019descriptors(azimuths, cv::Size(fft_data.cols, fft_data.rows), targets, cart_targets2,
            // radar_resolution, 0.3456, 722, desc2);
            detector->compute(img2, kp2, desc2);
//            cv::imshow("cart_image", img2);
//            cv::waitKey(0);
        }
        if (keypoint_extraction == 2) {
            detector->detect(img2, kp2);
            detector->compute(img2, kp2, desc2);
            convert_from_bev(kp2, cart_resolution, cart_pixel_width, cart_targets2);
            getTimes(cart_targets2, azimuths, times, t2);
        }
        if (i == 0)
            continue;
        // Match keypoint descriptors
        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(desc1, desc2, knn_matches, 2);

        // Filter matches using nearest neighbor distance ratio (Lowe, Szeliski)
        std::vector<cv::DMatch> good_matches;
        for (uint j = 0; j < knn_matches.size(); ++j) {
            if (!knn_matches[j].size())
                continue;
            if (knn_matches[j][0].distance < nndr * knn_matches[j][1].distance) {
                good_matches.push_back(knn_matches[j][0]);
            }
        }
        // Convert the good key point matches to Eigen matrices
        Eigen::MatrixXd p1 = Eigen::MatrixXd::Zero(2, good_matches.size());
        Eigen::MatrixXd p2 = p1;
        std::vector<int64_t> t1prime = t1, t2prime = t2;
        for (uint j = 0; j < good_matches.size(); ++j) {
            p1(0, j) = cart_targets1(0, good_matches[j].queryIdx);
            p1(1, j) = cart_targets1(1, good_matches[j].queryIdx);
            p2(0, j) = cart_targets2(0, good_matches[j].trainIdx);
            p2(1, j) = cart_targets2(1, good_matches[j].trainIdx);
            t1prime[j] = t1[good_matches[j].queryIdx];
            t2prime[j] = t2[good_matches[j].trainIdx];
        }
        t1prime.resize(good_matches.size());
        t2prime.resize(good_matches.size());

        std::vector<std::string> parts;
        boost::split(parts, radar_files[i], boost::is_any_of("."));
        int64 time1 = std::stoll(parts[0]);
        boost::split(parts, radar_files[i + 1], boost::is_any_of("."));
        int64 time2 = std::stoll(parts[0]);
        double delta_t = (time2 - time1) / 1000000.0;

        // Compute the transformation using RANSAC
        Ransac ransac(p2, p1, ransac_threshold, inlier_ratio, max_iterations);
        srand(i);
        ransac.computeModel();
        Eigen::MatrixXd T;  // T_1_2
        ransac.getTransform(T);

        // Compute the transformation using motion-distorted RANSAC
        MotionDistortedRansac mdransac(p2, p1, t2prime, t1prime, md_threshold, inlier_ratio, max_iterations);
        mdransac.setMaxGNIterations(max_gn_iterations);
        mdransac.correctForDoppler(false);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd;
        mdransac.getTransform(delta_t, Tmd);
        Tmd = Tmd.inverse();

        Eigen::VectorXd wbar;
        mdransac.getMotion(wbar);

        // MDRANSAC + Doppler
        mdransac.correctForDoppler(true);
        mdransac.setDopplerParameter(beta);
        srand(i);
        mdransac.computeModel();
        Eigen::MatrixXd Tmd2 = Eigen::MatrixXd::Zero(4, 4);
        mdransac.getTransform(delta_t, Tmd2);
        Tmd2 = Tmd2.inverse();

        // Retrieve the ground truth to calculate accuracy
        std::vector<float> gtvec;
        if (!get_groundtruth_odometry(gt, time1, time2, gtvec)) {
            std::cout << "ground truth odometry for " << time1 << " " << time2 << " not found... exiting." << std::endl;
            break;
        }
        float yaw = -1 * asin(T(0, 1));
        float yaw2 = -1 * asin(Tmd(0, 1));
        float yaw3 = -1 * asin(Tmd2(0, 1));

        // Write estimated and ground truth transform to the csv file
        ofs << T(0, 2) << "," << T(1, 2) << "," << yaw << ",";
        ofs << gtvec[0] << "," << gtvec[1] << "," << gtvec[5] << ",";
        ofs << time1 << "," << time2 << "," << Tmd(0, 3) << "," << Tmd(1, 3) << "," <<  yaw2 << ",";
        ofs << Tmd2(0, 3) << "," << Tmd2(1, 3) << "," << yaw3 << "\n";

        // cv::Mat img_matches;
        // cv::drawMatches(img1, kp1, img2, kp2, good_matches, img_matches, cv::Scalar::all(-1),
        //          cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        // cv::imshow("good", img_matches);
        // cv::waitKey(0);
    }
}

int main(int argc, char *argv[]) {
    omp_set_num_threads(8);

    // Get file names of the radar images
    get_file_names(datadir, radar_files);
// File for storing the results of estimation on each frame (and the accuracy)
    ofs.open("/home/gao/Dopper_radar/catkin_ws/src/yeti_radar_odometry-master/results/accuracy.csv", std::ios::out);
    ofs << "x,y,yaw,gtx,gty,gtyaw,time1,time2,xmd,ymd,yawmd,xdopp,ydopp,yawdopp\n";
    // Create ORB feature detector
    detector = cv::ORB::create();
    detector->setPatchSize(patch_size);
    detector->setEdgeThreshold(patch_size);
    // BRUTEFORCE_HAMMING for ORB descriptors FLANNBASED for cen2019 descriptors
    matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

    ros::init(argc, argv, "odometry_node");
    ros::NodeHandle nh;
    ros::Subscriber pcl_sub = nh.subscribe("radar_pcl", 100, PclCallback);
    ros::spin();

    return 0;
}
