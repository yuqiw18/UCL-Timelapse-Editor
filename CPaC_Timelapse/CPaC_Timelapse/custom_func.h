#pragma once



namespace CF {

	cv::Mat im2uint8(cv::Mat mat);
	cv::Mat im2single(cv::Mat mat);
	cv::Mat im2double(cv::Mat mat);
	cv::Mat imremap(cv::Mat src, std::vector<cv::Mat> flow);

	cv::Mat arrange(int max);
}