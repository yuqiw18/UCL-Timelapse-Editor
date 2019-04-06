#pragma once

namespace core {
	std::vector<cv::Mat> GammaCorrection(std::vector<cv::Mat> raw_sequence, cv::Mat &gamma_lookup_table);

	std::vector<cv::Mat> ComputeOpticalFlow(std::vector<cv::Mat> &raw_sequence, bool &use_cuda);
	std::vector<cv::Mat> ConvertFlowXY(cv::Mat optical_flow);

	std::vector<cv::Mat> MaskOpticalFlow(std::vector<cv::Mat> optical_flow);
	std::vector<cv::Mat> GenerateMotionTrail(std::vector<cv::Mat>raw_sequence);
	std::vector<cv::Mat> ApplyMotionTrail(std::vector<cv::Mat>input_sequence, std::vector<cv::Mat>motion_trail);

	std::vector<cv::Mat> RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int interpolation_frames);

	std::vector<cv::Mat> EnhanceImage(std::vector<cv::Mat> input_sequence);

	std::vector<cv::Mat> VintageFilter(std::vector<cv::Mat> input_sequence, std::vector<cv::Mat> mask_sequence);

	std::vector<cv::Mat> GetRemapMatrix(int w, int h);
	std::vector<cv::Mat> ConvertFlowXY2(cv::Mat optical_flow, std::vector<cv::Mat> remap_xy);

	cv::Mat GenerateGammaLookupTable(double gamma);
};