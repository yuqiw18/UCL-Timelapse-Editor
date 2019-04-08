#include <iostream>
#include <algorithm>
#include <string>
#include <vector>
#include <opencv2/core.hpp>
#include "utility.h"

std::string utility::FilePathParser(std::string file_path) {

	// Convert file path to lower case
	std::transform(file_path.begin(), file_path.end(), file_path.begin(), ::tolower); 

	// Check the file format
	int pos_format = file_path.find_last_of('.');
	std::string file_format = file_path.substr(pos_format + 1);

	// For loading image sequence
	if (std::find(IMAGE_FORMAT.begin(), IMAGE_FORMAT.end(), file_format) != IMAGE_FORMAT.end()) {
		
		// Remove the format
		std::string new_file_path = file_path;
		std::size_t found = new_file_path.find(file_format);
		new_file_path.erase(found-1, file_format.size()+1);

		// Find the file name
		int pos_name = new_file_path.find_last_of('\\');
		std::string file_name = new_file_path.substr(pos_name + 1);

		// Find the file number
		int pos_number = file_name.find_last_of('_');
		std::string file_number = file_name.substr(pos_number + 1);

		// Remove the number
		found = new_file_path.find(file_number);
		new_file_path.erase(found, file_number.size());

		// Rewrite the number to "Filename_%0#d" so that the sequence can be loaded
		new_file_path = new_file_path + "%0" + std::to_string(file_number.size()) + "d" + "." + file_format;

		return new_file_path;
	}
	else {
		// For loading video footage
		return file_path;
	}
}

std::string utility::ConvertFPStoTime(int total_frames, int fps){
	
	// Estimate the video length using total frame and export fps
	std::string h, m, s;
	int second = total_frames / fps;
	int remain_second = second % 60;
	int minute = (second-remain_second) / 60;
	int remain_minute = minute % 60;
	int hour = (minute-remain_minute) / 60;

	// Format 
	if (hour / 10 >= 1) {
		h = std::to_string(hour);
	}
	else {
		h = "0" + std::to_string(hour);
	}

	if (remain_minute / 10 >= 1) {
		m = std::to_string(remain_minute);
	}
	else {
		m = "0" + std::to_string(remain_minute);
	}

	if (remain_second / 10 >= 1) {
		s = std::to_string(remain_second);
	}
	else {
		s = "0" + std::to_string(remain_second);
	}

	std::string time;
	time = h + ":" + m + ":" + s;
	return time;
}