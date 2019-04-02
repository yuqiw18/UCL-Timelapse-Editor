#include <iostream>
#include <opencv2/core.hpp>
#include "custom_func.h"

cv::Mat CF::im2uint8(cv::Mat mat) {
	mat.convertTo(mat, CV_8UC3, 255);
	return mat;
}

cv::Mat CF::im2single(cv::Mat mat) {
	mat.convertTo(mat, CV_32FC3, 1.0 / 255);
	return mat;
}

cv::Mat CF::im2double(cv::Mat mat) {
	mat.convertTo(mat, CV_64FC3, 1.0 / 255);
	return mat;
}

cv::Mat CF::imremap(cv::Mat src, std::vector<cv::Mat> flow) {
	cv::Mat warp_result;
	return warp_result;
}


cv::Mat CF::arrange(int max) {

	cv::Mat sequence = cv::Mat::zeros(max, 1, CV_32FC1);

	for (int i = 0; i < max; i++) {
		sequence.at<float>(i,0) = i;
		std::cout << "x" << std::endl;
	}

	return sequence;
}

