#pragma once

namespace core {
	
	std::vector<cv::Mat> ComputeOpticalFlow(std::vector<cv::Mat>&input_sequence, bool &use_cuda);
	std::vector<cv::Mat> ConvertFlowXY(cv::Mat optical_flow);
	std::vector<cv::Mat> RetimeSequence(std::vector<cv::Mat>&input_sequence, std::vector<cv::Mat>&optical_flow, int &interpolation_frames);

	std::vector<cv::Mat> BrightnessSmoothing(std::vector<cv::Mat>&input_sequence);
	std::vector<cv::Mat> ContrastStretching(std::vector<cv::Mat>&input_sequence);
	cv::Mat ComputeCDF(cv::Mat input_channel);
	
	cv::Mat HistogramMatching(cv::Mat input_frame, cv::Mat cdf_source, cv::Mat cdf_target);
	cv::Mat AdjustBrightness(cv::Mat input_image, float value);
	cv::Mat AdjustContrast(cv::Mat input_image, float value);

	std::vector<cv::Mat> Vintage(std::vector<cv::Mat> &input_sequence, std::vector<cv::Mat> &mask_sequence);
	std::vector<cv::Mat> Miniature(std::vector<cv::Mat> &input_sequence, cv::Mat &mask_miniature);
	std::vector<cv::Mat> GenerateMotionTrail(std::vector<cv::Mat>&input_sequence);
	std::vector<cv::Mat> ApplyMotionTrail(std::vector<cv::Mat>&input_sequence, std::vector<cv::Mat>motion_trail);
	
	// UNUSED FUNCTIONS ...
	std::vector<cv::Mat> GammaCorrection(std::vector<cv::Mat>& input_sequence, cv::Mat &gamma_lookup_table);
	cv::Mat GenerateGammaLookupTable(double gamma);
};