#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaoptflow.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <math.h>
#include "core.h"
#include "utility.h"

std::vector<cv::Mat> core::ComputeOpticalFlow(std::vector<cv::Mat> &raw_sequence, bool &use_cuda) {
	
	// Tic
	std::cout << "@Computing Optical Flow (Dense)" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> optical_flow;

	// Create Farneback optical flow operator
	cv::Ptr<cv::FarnebackOpticalFlow> optical_flow_cv = cv::FarnebackOpticalFlow::create(5, 0.5, false, 15, 5);
	cv::Ptr< cv::cuda::FarnebackOpticalFlow> optical_flow_cuda = cv::cuda::FarnebackOpticalFlow::create(5,0.5,false,15,5);

	// Initially used for multi-view video sprite
	//cv::Ptr<cv::cuda::DensePyrLKOpticalFlow> optical_flow_cuda = cv::cuda::DensePyrLKOpticalFlow::create(cv::Size(21, 21), 5, 20, true);

	for (int i = 0; i < raw_sequence.size() - 1; i++) {

		if (use_cuda){
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
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;	
	if (use_cuda) {
		std::cout << "CUDA Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()-1) + " optical flows" << std::endl;
	}else{
		std::cout << "CPU Optical Flow takes " + std::to_string(time_taken) + "s to complete " + std::to_string(raw_sequence.size()-1) + " optical flows" << std::endl;
	}
	
	return optical_flow;
}

std::vector<cv::Mat> core::RetimeSequence(std::vector<cv::Mat> &raw_sequence, std::vector<cv::Mat> &optical_flow, int &interpolation_frames) {
	
	// Tic
	std::cout << "@Retiming Timelapse" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> processed_sequence;

	for (int i = 0; i < raw_sequence.size()-1; i++) {
	
		cv::Mat current_flow = optical_flow[i];

		// Add original frame to the output sequence
		processed_sequence.push_back(raw_sequence[i]);

		for (int f = 1; f < interpolation_frames + 1; f++) {

			float alpha = f / (float)(interpolation_frames + 1);

			cv::Mat f0, f1, frame_interp;

			// Convert optical flow structure for image warping
			std::vector<cv::Mat> flow_xy1 = ConvertFlowXY(-current_flow * alpha);
			cv::remap(raw_sequence[i], f0, flow_xy1[0], flow_xy1[1], cv::INTER_LINEAR, cv::BORDER_REPLICATE);

			std::vector<cv::Mat> flow_xy2 = ConvertFlowXY(current_flow * (1.0f - alpha));
			cv::remap(raw_sequence[i + 1], f1, flow_xy2[0], flow_xy2[1], cv::INTER_LINEAR, cv::BORDER_REPLICATE);

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

std::vector<cv::Mat> core::GenerateMotionTrail(std::vector<cv::Mat>input_sequence) {

	// Tic
	std::cout << "@Generating Motion Trail" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> motion_sequence;

	for (int i = 0; i < input_sequence.size()-1; i++) {
		cv::Mat frame_prev, frame_next;
		cv::cvtColor(input_sequence[i], frame_prev, CV_BGR2GRAY);
		cv::cvtColor(input_sequence[i+1], frame_next, CV_BGR2GRAY);

		cv::Mat motion = -(frame_prev- frame_next);

		cv::cvtColor(motion, motion, CV_GRAY2BGR);
		motion_sequence.push_back(motion);
	}

	cv::Mat motion_frame_last = cv::Mat::zeros(motion_sequence.front().size(), motion_sequence.front().type());
	motion_sequence.push_back(motion_frame_last);

	std::vector<cv::Mat> motion_sequence_merge;

	for (int i = 0; i < motion_sequence.size(); i++) {
	
		int motion_interval = 0.1 * input_sequence.size();
		cv::Mat motion_trail = cv::Mat::zeros(motion_sequence.front().size(), CV_8UC3);
		cv::cuda::GpuMat motion_trail_cuda;

		if (i < motion_interval) {
			for (int f = 0; f < i; f++) {

				motion_trail += ((float)i - (float)f) / (float)i * motion_sequence[i - f];
			}
		}
		else {
			for (int f = 0; f < motion_interval; f++) {
				motion_trail += ((float)motion_interval - (float)f) / (float)motion_interval * motion_sequence[i - f];
			}
		}

		cv::Mat motion_trail_adjusted;
		cv::normalize(motion_trail, motion_trail_adjusted, 0, 255, cv::NORM_MINMAX);
		cv::applyColorMap(motion_trail_adjusted, motion_trail_adjusted, cv::COLORMAP_HOT);
		motion_sequence_merge.push_back(motion_trail_adjusted);
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is " + std::to_string(time_taken) + "s " << std::endl;

	return motion_sequence_merge;
}


std::vector<cv::Mat> core::ApplyMotionTrail(std::vector<cv::Mat>input_sequence, std::vector<cv::Mat>motion_trail) {
	
	// Tic
	std::cout << "@Applying Motion Trail" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> blended_sequence;

	double alpha = 0.2;
	double beta = (1.0 - alpha);

	for (int i = 0; i < input_sequence.size(); i++) {

		/*cv::Mat frame_current;
		cv::cvtColor(input_sequence[i], frame_current, CV_BGR2GRAY);
		cv::cvtColor(frame_current, frame_current, CV_GRAY2BGR);*/
		cv::Mat frame_blended;
		cv::addWeighted(motion_trail[i], alpha, input_sequence[i], beta, 0.0, frame_blended);
		blended_sequence.push_back(frame_blended);
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is " + std::to_string(time_taken) + "s " << std::endl;

	return blended_sequence;
}

std::vector<cv::Mat> core::EnhanceImage(std::vector<cv::Mat> input_sequence) {

	// Tic
	std::cout << "@Enhancing Images" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> enhanced_senquance;

	// Histogram Matching
	float alpha, beta;
	int pixel_count = input_sequence.front().rows * input_sequence.front().cols;
	
	enhanced_senquance.push_back(input_sequence.front());

	for (int i = 1; i < input_sequence.size(); i++) {

		cv::Mat frame_matched = cv::Mat::zeros(input_sequence.front().size(), input_sequence.front().type());

		// Use intensity channel for colored image histogram matching e.g. YCrCb and HSV
		cv::Mat target_frame, source_frame;
		cv::cvtColor(input_sequence[i - 1], target_frame, CV_BGR2YCrCb);
		cv::cvtColor(input_sequence[i], source_frame, CV_BGR2YCrCb);

		// Split the frame into Hue, Saturation and Value(intensity)
		std::vector<cv::Mat> target_frame_channels, source_frame_channels;
		split(target_frame, target_frame_channels);
		split(source_frame, source_frame_channels);

		// Normalise CDF
		cv::Mat target_frame_intensity, source_frame_intensity;
		target_frame_channels[0].convertTo(target_frame_intensity, CV_32FC1);
		source_frame_channels[0].convertTo(source_frame_intensity, CV_32FC1);
		
		cv::Mat cdf_target_norm = ComputeCDF(target_frame_intensity)/(float)pixel_count;
		cv::Mat cdf_source_norm = ComputeCDF(source_frame_intensity)/(float)pixel_count;

		source_frame_channels[0] = HistogramMatching(source_frame_channels[0], cdf_source_norm, cdf_target_norm);

		merge(source_frame_channels, frame_matched);

		cv::cvtColor(frame_matched, frame_matched, CV_YCrCb2BGR);

		enhanced_senquance.push_back(frame_matched);

	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is " + std::to_string(time_taken) + "s " << std::endl;

	return enhanced_senquance;

}

cv::Mat core::ComputeCDF(cv::Mat &input_channel) {

	cv::Mat cdf = cv::Mat::zeros(256, 1, CV_32FC1);

	// Compute histogram
	cv::Mat histogram;
	int size = 256;
	float range[] = { 0, size }; // Range = [min, max)
	const float* histRange = { range };
	cv::calcHist(&input_channel, 1, 0, cv::Mat(), histogram, 1, &size, &histRange, true, false);

	// Compute CDF
	cdf.at<float>(0, 0) = histogram.at<float>(0, 0);
	for (int i = 1; i < 256; i++) {
		cdf.at<float>(i, 0) = cdf.at<float>(i - 1, 0) + histogram.at<float>(i, 0);
	}

	return cdf;
}

cv::Mat core::HistogramMatching(cv::Mat intensity_source, cv::Mat cdf_source, cv::Mat cdf_target) {

	cv::Mat matched_intensity;
	cv::Mat cdf_matched = cv::Mat::zeros(256, 1, CV_32FC1);

	for (int i = 0; i < 256; i++) {
		cv::Mat cdf_difference;
		int min_idx[2] = {255,255};
		cdf_difference = cv::abs(cdf_source.at<float>(i, 0) - cdf_target);
		cv::minMaxIdx(cdf_difference, NULL, NULL, min_idx, NULL);
		cdf_matched.at<float>(i, 0) = (float)min_idx[0];
	}

	cdf_matched.convertTo(cdf_matched, CV_8UC1);

	// Map the values back to the source channel
	cv::LUT(intensity_source, cdf_matched, matched_intensity);

	return matched_intensity;

}

std::vector<cv::Mat> core::Vintage(std::vector<cv::Mat> &input_sequence, std::vector<cv::Mat> &mask_sequence) {

	// Tic
	std::cout << "@Applying Vintage Filter" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> filtered_sequence;

	int mask_size = mask_sequence.size();
	double alpha = 0.2;
	double beta = (1.0 - alpha);

	double mask_duration_count = 0;
	int m = rand() % mask_size;

	for (int i = 0; i < input_sequence.size(); i++) {

		// Randomly pick an artefact mask
		if (mask_duration_count < 6) {
			mask_duration_count++;
		}
		else {
			m = rand() % mask_size;
			mask_duration_count = 0;
		}
		
		// Apply the sepia filter
		cv::Mat filtered_frame = cv::Mat::zeros(input_sequence.front().size(), CV_8UC3);
		cv::Mat mask, blended_frame;
		for (int x = 0; x < input_sequence[i].rows; x++) {
			for (int y = 0; y < input_sequence[i].cols; y++) {

				// Get B, G, R value from each channel
				int b = input_sequence[i].at<cv::Vec3b>(x, y)[0];
				int g = input_sequence[i].at<cv::Vec3b>(x, y)[1];
				int r = input_sequence[i].at<cv::Vec3b>(x, y)[2];

				// Create the new B, G, R value using the formula
				int b_new = 0.272*(float)r + 0.534*(float)g + 0.131*(float)b;
				int g_new = 0.349*(float)r + 0.686*(float)g + 0.168*(float)b;
				int r_new = 0.393*(float)r + 0.769*(float)g + 0.189*(float)b;

				// Make sure the obtained values are between 0 and 255
				if (b_new > 255) b_new = 255;
				if (g_new > 255) g_new = 255;
				if (r_new > 255) r_new = 255;
				if (b_new < 0) b_new = 0;
				if (g_new < 0) b_new = 0;
				if (r_new < 0) b_new = 0;

				// Assign the new B, G, R value
				filtered_frame.at<cv::Vec3b>(x, y)[0] = b_new;
				filtered_frame.at<cv::Vec3b>(x, y)[1] = g_new;
				filtered_frame.at<cv::Vec3b>(x, y)[2] = r_new;
			}
		}

		// Apply the mask to the new frame
		cv::resize(mask_sequence[m], mask, filtered_frame.size());
		cv::bitwise_not(filtered_frame, filtered_frame);
		cv::addWeighted(mask, alpha, filtered_frame, beta, 0.0, blended_frame);
		cv::bitwise_not(blended_frame, blended_frame);
		filtered_sequence.push_back(blended_frame);
	}

	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is " + std::to_string(time_taken) + "s " << std::endl;

	return filtered_sequence;
}

std::vector<cv::Mat> core::Miniature(std::vector<cv::Mat> &input_sequence, cv::Mat &mask_miniature) {

	// Tic
	std::cout << "@Applying Miniature Effect" << std::endl;
	clock_t start_time = std::clock();

	std::vector<cv::Mat> filtered_sequence;

	cv::Mat mask;
	cv::resize(mask_miniature, mask, input_sequence.front().size());
	cv::cvtColor(mask, mask, CV_BGR2GRAY);

	for (int i = 0; i < input_sequence.size(); i++) {
	
		cv::Mat blurred_mask, blurred_frame;
		cv::Mat filtered_frame = cv::Mat::zeros(input_sequence.front().size(), input_sequence.front().type());

		cv::GaussianBlur(mask, blurred_mask, cv::Size(7, 7), 0, 0, cv::BORDER_REPLICATE);
		cv::GaussianBlur(input_sequence[i], blurred_frame, cv::Size(7, 7), 0, 0, cv::BORDER_REPLICATE);

		for (int x = 0; x < filtered_frame.rows; x++){
			for (int y = 0; y < filtered_frame.cols; y++){

				cv::Vec3b input_pixel = input_sequence[i].at<cv::Vec3b>(x, y);
				cv::Vec3b blurred_pixel = blurred_frame.at<cv::Vec3b>(x, y);
				uchar mask_pixel = blurred_mask.at<uchar>(x, y);

				float alpha = mask_pixel / 255.0;
				float beta = 1.0 - alpha;

				filtered_frame.at<cv::Vec3b>(x, y) = beta * input_pixel + alpha * blurred_pixel;
			}
		}
		//cv::cvtColor(blurred_mask, blurred_mask, CV_GRAY2BGR);
		filtered_sequence.push_back(filtered_frame);
	}
	
	// Toc
	double time_taken = (clock() - start_time) / (double)CLOCKS_PER_SEC;
	std::cout << "Elapsed time is " + std::to_string(time_taken) + "s " << std::endl;

	return filtered_sequence;
}

std::vector<cv::Mat> core::ContrastStretching(std::vector<cv::Mat> input_sequence) {

	std::vector<cv::Mat> stretched_senquance;
	for (int i = 0; i < input_sequence.size(); i++) {

		// Convert to YCrCb to get intensity channel
		cv::Mat source_frame;
		std::vector<cv::Mat> color_channels;
		cv::cvtColor(input_sequence[i], source_frame, CV_BGR2YCrCb);
		cv::split(source_frame, color_channels);

		// Get minimum and maximum intensity
		double min, max;
		cv::minMaxIdx(color_channels[0], &min, &max);
		
		// Stretch using linear mapping formula so that values are distributed across [0, 255]
		color_channels[0] = 255 * (color_channels[0] - (int)min) / ((int)max - (int)min);

		// Convert back to B, G, R
		cv::Mat stretched_frame;
		cv::merge(color_channels, stretched_frame);
		cv::cvtColor(stretched_frame, stretched_frame, CV_YCrCb2BGR);

		stretched_senquance.push_back(stretched_frame);
	}
	return stretched_senquance;
}

// ...... UNUSED FUNCTIONS 
cv::Mat core::GenerateGammaLookupTable(double gamma) {
	// Gamma correction for sRGB
	cv::Mat gamma_lookup_table = cv::Mat::zeros(1, 256, CV_8UC1);
	for (int i = 0; i < 256; i++) {
		gamma_lookup_table.at<uint8_t>(0, i) = pow(i / 255.0, gamma) * 255.0;

		//std::cout << std::to_string(gamma_lookup_table.at<uint8_t>(0, i)) << std::endl;
	}
	return gamma_lookup_table;
}

std::vector<cv::Mat> core::GammaCorrection(std::vector<cv::Mat> raw_sequence, cv::Mat &gamma_lookup_table) {

	for (int i = 0; i < raw_sequence.size(); i++) {
		cv::Mat frame_corrected;
		cv::LUT(raw_sequence[i], gamma_lookup_table, frame_corrected);
		raw_sequence[i] = frame_corrected;
	}

	return raw_sequence;
}

