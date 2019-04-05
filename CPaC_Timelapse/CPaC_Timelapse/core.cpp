#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "core.h"
#include "utility.h"

std::vector<cv::Mat> core::ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence) {
	
	// Tic
	std::cout << "@Computing Optical Flow (Dense)" << std::endl;
	clock_t start_time = std::clock();

	bool cuda = true;
	std::vector<cv::Mat> optical_flow;

	// Create Farneback optical flow operator
	cv::Ptr<cv::FarnebackOpticalFlow> optical_flow_cv = cv::FarnebackOpticalFlow::create(3, 0.5, false, 17, 3);
	cv::Ptr< cv::cuda::FarnebackOpticalFlow> optical_flow_cuda = cv::cuda::FarnebackOpticalFlow::create(5,0.5,false,15,20);
	//cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> optical_flow_cuda = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(21, 21), 5, 20, true);

	for (int i = 0; i < raw_sequence.size() - 1; i++) {

		if (cuda){
		cv::cuda::GpuMat frame_previous, frame_next, frame_flow;
		cv::Mat flow;

		// Assign frames to GPU memory
		frame_previous.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		// Convert frames to grasacle using GPU
		cv::cuda::cvtColor(frame_previous, frame_previous, CV_BGR2GRAY);
		cv::cuda::cvtColor(frame_next, frame_next, CV_BGR2GRAY);

		// Compute optical flows between two frames using GPU
		optical_flow_cuda->calc(frame_previous, frame_next, frame_flow);

		// Get results from GPU memory and save them
		frame_flow.download(flow);
		optical_flow.push_back(flow);

		}else {
			cv::Mat f0, f1, flow;
			
			// Convert frames to grasacle using CPU
			cv::cvtColor(raw_sequence[i], f0, CV_BGR2GRAY);
			cv::cvtColor(raw_sequence[i+1], f1, CV_BGR2GRAY);

			// Compute optical flows between two frames using CPU
			optical_flow_cv->calc(f0, f1, flow);

			// Save results
			optical_flow.push_back(flow);
		}
		//std::cout << "... ";
	}
	//std::cout << "" << std::endl;
	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;	
	if (cuda) {
		std::cout << "CUDA Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()-1) + " optical flows" << std::endl;
	}else{
		std::cout << "CPU Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()-1) + " optical flows" << std::endl;
	}
	
	return optical_flow;
}

std::vector<cv::Mat> core::RetimeSequence(std::vector<cv::Mat> raw_sequence, std::vector<cv::Mat> optical_flow, int interpolation_frames) {
	
	// Tic
	std::cout << "@Retiming Timelapse" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> processed_sequence;

	for (int i = 0; i < raw_sequence.size()-1; i++) {
	
		cv::cuda::GpuMat frame_prev, frame_next;

		// Assign frames to GPU memory
		frame_prev.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		cv::Mat current_flow = optical_flow[i];

		// Add original frame to the output sequence
		processed_sequence.push_back(raw_sequence[i]);

		for (int f = 1; f < interpolation_frames + 1; f++) {

			float alpha = f / (float)(interpolation_frames + 1);

			cv::cuda::GpuMat frame_prev_interp, frame_next_interp;
			cv::cuda::GpuMat flow_x1, flow_y1, flow_x2, flow_y2;
			cv::Mat f0, f1, frame_interp, none;

			// Convert optical flow structure for image warping
			std::vector<cv::Mat> flow_xy1 = ConvertFlowXY(-current_flow * alpha);

			flow_x1.upload(flow_xy1[0]);
			flow_y1.upload(flow_xy1[1]);
			cv::cuda::remap(frame_prev, frame_prev_interp, flow_x1, flow_y1, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

			std::vector<cv::Mat> flow_xy2 = ConvertFlowXY(current_flow * (1.0f - alpha));
			flow_x2.upload(flow_xy2[0]);
			flow_y2.upload(flow_xy2[1]);
			cv::cuda::remap(frame_next, frame_next_interp, flow_x2, flow_y2, cv::INTER_LINEAR, cv::BORDER_REPLICATE);

			// Get results from the GPU
			frame_prev_interp.download(f0);
			frame_next_interp.download(f1);

			// Weight interpolated frame
			frame_interp = (1.0f - alpha) * f0 + alpha * f1;

			// Save the result
			processed_sequence.push_back(frame_interp);
		}
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Retiming Task takes " + std::to_string(time_taken) + "s to complete " + std::to_string((raw_sequence.size() - 1)* interpolation_frames) + " frames" << std::endl;

	return processed_sequence;

}

std::vector<cv::Mat> core::ConvertFlowXY(cv::Mat optical_flow) {

	cv::Mat flow(optical_flow.size(), CV_32FC2);
	for (int y = 0; y < flow.rows; y++)
	{
		for (int x = 0; x < flow.cols; x++)
		{
			cv::Point2f f = optical_flow.at<cv::Point2f>(y, x);
			flow.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
		}
	}

	std::vector<cv::Mat> flow_xy;
	cv::split(flow, flow_xy);

	return flow_xy;
}

std::vector<cv::Mat> core::GetRemapMatrix(int h, int w) {

	std::vector<cv::Mat> remap_xy;

	cv::Mat remap_x_row = cv::Mat::zeros(1, w, CV_32FC1);
	cv::Mat remap_y_col = cv::Mat::zeros(h, 1, CV_32FC1);

	cv::Mat remap_x, remap_y;

	for (int i = 0; i < w; i++) {
		remap_x_row.at<float>(0, i) = i;
	}
	cv::repeat(remap_x_row, h, 1, remap_x);


	for (int i = 0; i < h; i++) {
		remap_y_col.at<float>(i, 0) = i;
	}
	cv::repeat(remap_y_col, 1, w, remap_y);

	remap_xy.push_back(remap_y);
	remap_xy.push_back(remap_x);

	return remap_xy;

}

std::vector<cv::Mat> core::ConvertFlowXY2(cv::Mat optical_flow, std::vector<cv::Mat> remap_xy) {

	bool cuda = false;

	if (cuda) {
		std::vector<cv::cuda::GpuMat> flow_xy;
		cv::cuda::GpuMat remap_x, remap_y, result_x, result_y;
		cv::Mat flow_x, flow_y;

		cv::cuda::split(optical_flow, flow_xy);
	
		remap_y.upload(remap_xy[0]);
		remap_x.upload(remap_xy[1]);
	
		cv::cuda::add(flow_xy[0], remap_xy[0], result_y);
		cv::cuda::add(flow_xy[1], remap_xy[1], result_x);

		result_y.download(flow_y);
		result_x.download(flow_x);

		std::vector<cv::Mat> new_flow_xy;
		new_flow_xy.push_back(flow_x);
		new_flow_xy.push_back(flow_y);
		return new_flow_xy;
	}
	else {
		std::vector<cv::Mat> flow_xy;
		cv::split(optical_flow, flow_xy);

		std::vector<cv::Mat> new_flow_xy;

		new_flow_xy.push_back(flow_xy[1] + remap_xy[1]);
		new_flow_xy.push_back(flow_xy[0] + remap_xy[0]);

		return new_flow_xy;
	}
}