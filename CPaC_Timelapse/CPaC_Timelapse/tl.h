#pragma once

namespace TL {

	std::vector<cv::Mat> StabliseSequence(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int interval);
	std::vector<cv::Mat> Gamma(std::vector<cv::Mat> raw_sequence);

}
