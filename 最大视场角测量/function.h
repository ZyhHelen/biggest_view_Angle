#pragma once
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/core/core.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/imgproc/imgproc.hpp> // 图像预处理头文件
#include<iostream>
#include<map>
#include<string>

using namespace cv;

Mat preProcess(const char *image);
Mat getROI(Mat image, const char *srcImage);
int getCounterNum(Mat image);
float getPrecision(const char *image);
float getResult(const char *image);

void Threshold_Demo(int, void*);
void binaryZation(const char * image);
Point getCircleCenter(const char *image);
void filter(const char *image);
int ImageBinarization(Mat src);
