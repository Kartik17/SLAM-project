#include <iostream>
#include <list>
#include <string>
#include <fstream>
#include <vector>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <sensor_msgs/image_encodings.h>

using namespace std;
/**
 * argv -  .txt file which contains list of all images similar to what we give ORB SLAM non ROS version  
 * 
 * This node will publish a topic = /camera/image_raw publishing the image
 * */

void LoadImages(const string &strFile, vector<string> &vstrImageFilenames, vector<double> &vTimestamps)
{
    ifstream f;
    f.open(strFile.c_str());

    // skip first three lines
    string s0;
    getline(f,s0);
    getline(f,s0);
    getline(f,s0);

    while(!f.eof())
    {
        string s;
        getline(f,s);
        if(!s.empty())
        {
            stringstream ss;
            ss << s;
            double t;
            string sRGB;
            ss >> t;
            vTimestamps.push_back(t);
            ss >> sRGB;
            vstrImageFilenames.push_back(sRGB);
        }
    }
}

int main(int argc, char** argv)
{
    if(argc<2)
    {
        std::cout<<"no image names given"<<std::endl;
        return -1;
    }
    ros::init(argc,argv,"image_publisher");
    ros::NodeHandle nh;
    ros::Publisher image_pub = nh.advertise<sensor_msgs::Image>("/camera/image_raw",5);


    // Retrive path to images 
    std::vector<string> vstrImageFilenames;
    std::vector<double> vTimestamps;
    std::string strFile = string(argv[1])+"/rgb.txt";
    LoadImages(strFile, vstrImageFilenames, vTimestamps);

    int nImages = vstrImageFilenames.size();

    try
    {
        cv::Mat image;
        cv_bridge::CvImage img_bridge;
        sensor_msgs::Image img_msg; // >> message to be sent
        std_msgs::Header header;

        ROS_INFO("loading images");
        int rate = 10;
        ROS_INFO_STREAM("publishing images at "<<rate <<" hz ");
        ros::Rate ros_rate(rate);
        for(int i=0;i<nImages;++i)
        {
            // Read image from file
            image = cv::imread(string(argv[1])+"/"+vstrImageFilenames[i],CV_LOAD_IMAGE_UNCHANGED);
            header.seq = i;
            header.stamp = ros::Time::now();//vTimestamps[i];
            img_bridge = cv_bridge::CvImage(header,sensor_msgs::image_encodings::RGB8,image);
            img_bridge.toImageMsg(img_msg);
            image_pub.publish(img_msg);
            ros::spinOnce();
            ros_rate.sleep();

        }
    }
    catch(const std::exception& e)
    {
        std::cout<<"Exception in reading the images"<< e.what() << '\n';
    }
    return 0;
    
}