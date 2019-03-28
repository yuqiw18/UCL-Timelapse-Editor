#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/video/tracking.hpp>
#include "tlt.h"

std::vector<cv::Mat> TLT::ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence) {
	std::cout << "@Computing Optical Flow" << std::endl;
	std::vector<cv::Mat> optical_flow;

	cv::cuda::GpuMat frame_previous, frame_next, frame_flow;
	cv::Mat flow;

	for (int i = 0; i < raw_sequence.size()-1; i++) {
	
		frame_previous.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		std::cout << "PASS" << std::endl;

		// REQUIRE CORNER DETECTOR

		cv::Ptr< cv::cuda::DensePyrLKOpticalFlow> d_pyrLK = cv::cuda::DensePyrLKOpticalFlow::create();

		d_pyrLK->calc(frame_previous, frame_next, frame_flow);

		std::cout << "PASS2" << std::endl;

		frame_flow.download(flow);

		optical_flow.push_back(flow);

		std::cout << "Cal" << std::endl;
	
	}

	return optical_flow;
}