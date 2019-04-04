#pragma once

namespace core {

	std::vector<cv::Mat> StabliseSequence(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence);
	std::vector<cv::Mat> RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int interpolation_frames);
	std::vector<cv::Mat> Gamma(std::vector<cv::Mat> raw_sequence);

	std::vector<cv::Mat> ConvertFlowXY(cv::Mat optical_flow);

	std::vector<cv::Mat> GetRemapMatrix(int w, int h);
	std::vector<cv::Mat> ConvertFlowXY2(cv::Mat optical_flow, std::vector<cv::Mat> remap_xy);

}



