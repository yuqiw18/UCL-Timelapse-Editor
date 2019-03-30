#pragma once

namespace TLT {

	std::vector<cv::Mat> StabliseSequence(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int interval);
	std::vector<cv::Mat> Gamma(std::vector<cv::Mat> raw_sequence);






	
	cv::Mat im2uint8(cv::Mat mat);
	cv::Mat im2single(cv::Mat mat);
	cv::Mat im2double(cv::Mat mat);
}
