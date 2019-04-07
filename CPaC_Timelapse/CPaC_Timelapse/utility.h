#pragma once

namespace utility {

	// Global variables
	const std::vector<std::string> VIDEO_FORMAT{};
	const std::vector<std::string> IMAGE_FORMAT{ "bmp","pbm","pgm","ppm","sr","ras","jpeg","jpg","jpe","jp2","tiff","tif","png" };

	std::string FilePathParser(std::string file_path);
	std::string ConvertFPStoTime(int total_frames, int fps);

	cv::Mat im2uint8(cv::Mat mat);
	cv::Mat im2single(cv::Mat mat);
	cv::Mat im2double(cv::Mat mat);
}