#pragma once

namespace utility {

	// Global variables
	const std::vector<std::string> VIDEO_FORMAT{};
	const std::vector<std::string> IMAGE_FORMAT{ "bmp","pbm","pgm","ppm","sr","ras","jpeg","jpg","jpe","jp2","tiff","tif","png" };

	std::string FilePathParser(std::string file_path);
	std::string ConvertFPStoTime(int total_frames, int fps);
}