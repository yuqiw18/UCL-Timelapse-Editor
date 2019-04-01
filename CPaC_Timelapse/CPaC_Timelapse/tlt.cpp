#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "tlt.h"

std::vector<cv::Mat> TLT::ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence) {
	std::cout << "@Computing Optical Flowv (Dense)" << std::endl;
	
	int cuda = true;

	// Tic
	clock_t start_time = std::clock();

	std::vector<cv::Mat> optical_flow;

	cv::cuda::GpuMat frame_previous, frame_next, frame_flow, corver_previous, corner_next;
	
	cv::Ptr<cv::FarnebackOpticalFlow> fof = cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3);
	cv::Ptr< cv::cuda::FarnebackOpticalFlow> fof_cuda = cv::cuda::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3);

	for (int i = 0; i < raw_sequence.size() - 1; i++) {

		cv::Mat f0, f1, flow;

		if (cuda){
			frame_previous.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		cv::cuda::cvtColor(frame_previous, frame_previous, CV_BGR2GRAY);
		cv::cuda::cvtColor(frame_next, frame_next, CV_BGR2GRAY);
		fof_cuda->calc(frame_previous, frame_next, frame_flow);

		frame_flow.download(flow);
		optical_flow.push_back(flow);

	}else {
			cv::Mat f0, f1, flow;
			cv::cvtColor(raw_sequence[i], f0, CV_BGR2GRAY);
			cv::cvtColor(raw_sequence[i+1], f1, CV_BGR2GRAY);
			fof->calc(f0, f1, flow);
			optical_flow.push_back(flow);
		}
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	
	if (cuda) {
		std::cout << "CUDA Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()) + " frames" << std::endl;
	}else{
		std::cout << "CPU Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()) + " frames" << std::endl;
	}
	

	return optical_flow;
}

std::vector<cv::Mat> TLT::RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int frame_interval) {

	std::vector<cv::Mat> processed_sequence;

	cv::Mat previous_frame, next_frame, previous_interpolated_frame, next_interpolated_frame, weighted_flow, imwarp;
	cv::Mat affine_warp, interpolated_frame, f0, f1;

	std::vector<cv::Mat> map;

	cv::Mat hsv = cv::Mat(raw_sequence[0].size(), raw_sequence[0].type());

	for (int i = 0; i < raw_sequence.size()-1; i++) {

		//std::cout << optical_flow[i].at<cv::Vec2f>(0,0)[0] << std::endl;

		cv::Mat backward_flow = optical_flow[i];

		cv::Mat map(backward_flow.size(), CV_32FC2);
		for (int y = 0; y < map.rows; ++y)
		{
			for (int x = 0; x < map.cols; ++x)
			{
				cv::Point2f f = backward_flow.at<cv::Point2f>(y, x);
				map.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
			}
		}

		cv::Mat none, warp;

		cv::remap(raw_sequence[i], warp, map, none, cv::INTER_LINEAR);

		//cv::remap(raw_sequence[i], warp, map[0], map[1], cv::INTER_LINEAR);
		
		processed_sequence.push_back(warp);

		//previous_frame = raw_sequence[i];
		//next_frame = raw_sequence[i + 1];

		//std::vector<cv::Mat> map;
		//cv::split(-1*optical_flow[i], map);

		//cv::remap(previous_frame, interpolated_frame, optical_flow[i],cv::_InputArray::NONE,cv::INTER_LINEAR);

		/*
		processed_sequence.push_back(interpolated_frame);

		for (int f = 1; f < frame_interval + 1; f++) {
			float alpha = f / (frame_interval + 1);

			

			cv::split(optical_flow[i]*(alpha*-1.0), map);

			cv::remap(previous_frame, f0,map[0],map[1], cv::INTER_LINEAR);


			cv::split(optical_flow[i]*(1.0 - alpha), map);

			cv::remap(next_frame, f1, map[0], map[1], cv::INTER_LINEAR);
		
			cv::Mat intepolated_frame = (1 - alpha)*f0 + alpha * f1;
			processed_sequence.push_back(intepolated_frame);
		}
		*/

	}


	//cv::cuda::GpuMat previous_frame, next_frame, previous_interpolated_frame, next_interpolated_frame, weighted_flow, imwarp;
	//cv::Mat affine_warp, interpolated_frame, f0, f1;

	//cv::Mat empty_warp(optical_flow[0].size(), optical_flow[0].type());

	//empty_warp = 0;

	//for (int i = 0; i < raw_sequence.size() - 1; i++) {

	//	previous_frame.upload(raw_sequence[i]);
	//	next_frame.upload(raw_sequence[i + 1]);

	//	for (int f = 1; f < frame_interval + 1; f++) {
	//		float alpha = f / (frame_interval + 1);

	//		imwarp.upload(alpha*-1.0*optical_flow[i]);
	//		cv::cuda::warpAffine(previous_frame, previous_interpolated_frame, imwarp, previous_interpolated_frame.size());
	//		
	//		//affine_warp = cv::getAffineTransform(empty_warp, (1.0 - alpha)* optical_flow[i]);
	//		imwarp.upload((1.0 - alpha)* optical_flow[i]);
	//		cv::cuda::warpAffine(next_frame, next_interpolated_frame, imwarp, next_interpolated_frame.size());

	//		previous_interpolated_frame.download(f0);
	//		next_interpolated_frame.download(f1);
	//		cv::Mat intepolated_frame = (1 - alpha)* + alpha * f1;
	//		processed_sequence.push_back(intepolated_frame);
	//	}
	//
	//}

	return processed_sequence;

}














cv::Mat TLT::im2uint8(cv::Mat mat) {
	mat.convertTo(mat, CV_8UC3, 255);
	return mat;
}

cv::Mat TLT::im2single(cv::Mat mat) {
	mat.convertTo(mat, CV_32FC3, 1.0 / 255);
	return mat;
}

cv::Mat TLT::im2double(cv::Mat mat) {
	mat.convertTo(mat, CV_64FC3, 1.0 / 255);
	return mat;
}