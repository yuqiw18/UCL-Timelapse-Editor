#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/video/tracking.hpp>
#include "custom_func.h"
#include "tlt.h"

std::vector<cv::Mat> TLT::ComputeOpticalFlow(std::vector<cv::Mat> raw_sequence) {
	std::cout << "@Computing Optical Flowv (Dense)" << std::endl;
	clock_t start_time = std::clock();

	bool cuda = true;

	std::vector<cv::Mat> optical_flow;

	cv::Ptr<cv::FarnebackOpticalFlow> fof = cv::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3);
	cv::Ptr< cv::cuda::FarnebackOpticalFlow> fof_cuda = cv::cuda::FarnebackOpticalFlow::create(3, 0.5, false, 15, 3);

	for (int i = 0; i < raw_sequence.size() - 1; i++) {

		if (cuda){

		cv::cuda::GpuMat frame_previous, frame_next, frame_flow;
		cv::Mat flow;

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

	for (int i = 0; i < raw_sequence.size()-1; i++) {
	
		cv::cuda::GpuMat frame_prev, frame_next;
		frame_prev.upload(raw_sequence[i]);
		frame_next.upload(raw_sequence[i + 1]);

		cv::Mat current_flow = -optical_flow[i];

		processed_sequence.push_back(raw_sequence[i]);

		for (int f = 1; f < frame_interval + 1; f++) {

			float alpha = f / (float)(frame_interval + 1);

			printf("float: %f", alpha);
		
			//cv::cuda::GpuMat frame_prev_interp, frame_next_interp;
			//cv::cuda::GpuMat flow_x1, flow_y1, flow_x2, flow_y2;
			//cv::Mat f0, f1, frame_interp;
		
			//std::vector<cv::Mat> flow_xy1 = ConvertFlowXY(-current_flow * (1.0 - alpha));

			//flow_x1.upload(flow_xy1[0]);
			//flow_y1.upload(flow_xy1[1]);

			//cv::cuda::remap(frame_prev, frame_prev_interp, flow_x1, flow_y1, cv::INTER_LINEAR);

			//std::vector<cv::Mat> flow_xy2 = ConvertFlowXY(current_flow * alpha);

			//flow_x2.upload(flow_xy2[0]);
			//flow_y2.upload(flow_xy2[1]);

			//cv::cuda::remap(frame_next, frame_next_interp, flow_x2, flow_y2, cv::INTER_LINEAR);

			//frame_prev_interp.download(f0);
			//frame_next_interp.download(f1);
			//frame_interp = alpha * f0 + (1-alpha) * f1;
			//processed_sequence.push_back(frame_interp);
		
			cv::cuda::GpuMat frame_prev_interp, frame_next_interp;
			cv::cuda::GpuMat flow_x1, flow_y1, flow_x2, flow_y2;
			cv::Mat f0, f1, frame_interp;

			std::vector<cv::Mat> flow_xy1 = ConvertFlowXY2(-current_flow * alpha);

			flow_x1.upload(flow_xy1[0]);
			flow_y1.upload(flow_xy1[1]);

			cv::cuda::remap(frame_prev, frame_prev_interp, flow_x1, flow_y1, cv::INTER_LINEAR);

			std::vector<cv::Mat> flow_xy2 = ConvertFlowXY2(current_flow * (1 - alpha));

			flow_x2.upload(flow_xy2[0]);
			flow_y2.upload(flow_xy2[1]);

			cv::cuda::remap(frame_next, frame_next_interp, flow_x2, flow_y2, cv::INTER_LINEAR);

			frame_prev_interp.download(f0);
			frame_next_interp.download(f1);
			frame_interp = (1 - alpha) * f0 + alpha * f1;
			processed_sequence.push_back(frame_interp);

		}

		/*processed_sequence.push_back(raw_sequence[i]);

		std::vector<cv::Mat> flow_xy = ConvertFlowXY(-optical_flow[i] * 0.5);
		cv::cuda::GpuMat frame_prev, frame_next, frame_interp, flow_x, flow_y;
		cv::Mat none, warp;

		frame_prev.upload(raw_sequence[i]);
		flow_x.upload(flow_xy[0]);
		flow_y.upload(flow_xy[1]);

		cv::cuda::remap(frame_prev, frame_interp, flow_x, flow_y, cv::INTER_LINEAR);
		
		frame_interp.download(warp);

		processed_sequence.push_back(warp);*/

	}

	return processed_sequence;

}


std::vector<cv::Mat> TLT::ConvertFlowXY(cv::Mat optical_flow) {

	cv::Mat flow(optical_flow.size(), CV_32FC2);
	for (int y = 0; y < flow.rows; ++y)
	{
		for (int x = 0; x < flow.cols; ++x)
		{
			cv::Point2f f = optical_flow.at<cv::Point2f>(y, x);
			flow.at<cv::Point2f>(y, x) = cv::Point2f(x + f.x, y + f.y);
		}
	}

	std::vector<cv::Mat> flow_xy;
	cv::split(flow, flow_xy);

	return flow_xy;
}


std::vector<cv::Mat> TLT::ConvertFlowXY2(cv::Mat optical_flow) {

	int w = optical_flow.cols;
	int h = optical_flow.rows;

	std::vector<cv::Mat> flow_xy;
	cv::split(optical_flow, flow_xy);

	cv::Mat flow_y = flow_xy[0];
	cv::Mat flow_x = flow_xy[1];

	flow_y += CF::arrange(w);
	flow_x += CF::arrange(h).t();


	std::vector<cv::Mat> new_flow_xy;
	new_flow_xy.push_back(flow_x);
	new_flow_xy.push_back(flow_y);
	return new_flow_xy;
}