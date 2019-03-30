#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "tlt.h"

std::vector<cv::Mat> TLT::ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence) {
	std::cout << "@Computing Optical Flow" << std::endl;
	
	// Tic
	clock_t start_time = std::clock();

	std::vector<cv::Mat> optical_flow;

	cv::cuda::GpuMat frame_previous, frame_next, frame_flow, corver_previous, corner_next;
	cv::Mat flow;

	for (int i = 0; i < raw_sequence.size()-1; i++) {
	
		frame_previous.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		cv::cuda::cvtColor(frame_previous, frame_previous, CV_BGR2GRAY);
		cv::cuda::cvtColor(frame_next, frame_next, CV_BGR2GRAY);
		

		std::cout << "PASS" << std::endl;

		// REQUIRE CORNER DETECTOR
		//cv::Ptr< cv::cuda::CornersDetector> detector = cv::cuda::createGoodFeaturesToTrackDetector(frame_previous.type());
		cv::Ptr< cv::cuda::DensePyrLKOpticalFlow> d_pyrLK = cv::cuda::DensePyrLKOpticalFlow::create();

		//detector->detect(frame_previous,corver_previous);
		//detector->detect(frame_next, corner_next);

		d_pyrLK->calc(frame_previous, frame_next, frame_flow);

		frame_flow.download(flow);

		optical_flow.push_back(flow);
	
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "CUDA Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size() - 1) + " frames" << std::endl;

	return optical_flow;
}

std::vector<cv::Mat> TLT::RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int frame_interval) {

	std::vector<cv::Mat> processed_sequence;

	cv::cuda::GpuMat previous_frame, next_frame, previous_interpolated_frame, next_interpolated_frame, weighted_flow, imwarp;
	cv::Mat affine_warp, interpolated_frame, f0, f1;

	cv::Mat empty_warp(optical_flow[0].size(), optical_flow[0].type());

	empty_warp = 0;

	for (int i = 0; i < raw_sequence.size() - 1; i++) {

		previous_frame.upload(raw_sequence[i]);
		next_frame.upload(raw_sequence[i + 1]);

		affine_warp = cv::getAffineTransform(empty_warp, optical_flow[i]);

		for (int f = 1; f < frame_interval + 1; f++) {
			float alpha = f / (frame_interval + 1);

			affine_warp = cv::getAffineTransform(empty_warp, alpha*-1.0*optical_flow[i]);
			imwarp.upload(affine_warp);
			cv::cuda::warpAffine(previous_frame, previous_interpolated_frame, imwarp, previous_interpolated_frame.size());
			
			affine_warp = cv::getAffineTransform(empty_warp, (1.0 - alpha)* optical_flow[i]);
			imwarp.upload(affine_warp);
			cv::cuda::warpAffine(next_frame, next_interpolated_frame, affine_warp, next_interpolated_frame.size());

			previous_interpolated_frame.download(f0);
			next_interpolated_frame.download(f1);
			cv::Mat intepolated_frame = (1 - alpha)* + alpha * f1;
			processed_sequence.push_back(intepolated_frame);
		}
	
	}

	return processed_sequence;

}
