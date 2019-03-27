/*
University College London
Department of Computer Science, CGVI
COMP0028: Computational Photography and Capture
Class Project Part B: Timelapse
Coded By Yuqi Wang (18043263)
*/
/*
Runtime: WINDOWS X64 + OpenCV 3.4 + CUDA 9.1 
*/


#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include "tl.h"
#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

#define WINDOW_NAME "Timelapse Toolbox"
#define PADDING_HORIZONTAL 6
#define PADDING_VERTICAL 6

int main(void){
	
	if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
		std::cout << "No Cuda" << std::endl;
	}

	// File browser
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name
	HWND hwnd = NULL;              // owner window
	HANDLE hf;              // file handle

	// Initialize OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;
	//
	// Set lpstrFile[0] to '\0' so that GetOpenFileName does not 
	// use the contents of szFile to initialize itself.
	//
	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Init cvui and tell it to create a OpenCV window, i.e. cv::namedWindow(WINDOW_NAME).
	int count = 0;
	cv::Mat gui = cv::Mat(600, 800, CV_8UC3);
	gui = cv::Scalar(49, 52, 49);
	bool use_canny = false;
	int low_threshold = 50, high_threshold = 150;

	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);


	// 
	cv::VideoCapture footage;
	std::vector<cv::Mat> raw_sequence;
	std::vector<cv::Mat> processed_sequence;
	std::vector<cv::Mat> preview_sequence;

	std::vector<cv::Mat> optical_flow;

	bool video_input_ready = false;
	bool sequence_loaded = false;

	bool input_check_pass = false;

	bool is_video_play = false;
	int current_clip = 0;
	int current_frame = 0;
	int sequence_length = 1;

	bool idle = false;

	std::string play_button_name = "Play";

	while (true) {
		
		// GUI: Read & Save Files
		cvui::window(gui, 6, 6, 140, 100, "File");
		if (cvui::button(gui, 12, 32, 128, 32,"Import(V/I)")) {
			if (GetOpenFileName(&ofn) == TRUE) {
				video_input_ready = false;
				current_frame = 0;
				sequence_length = 1;
				raw_sequence.clear();
				preview_sequence.clear();
				preview_sequence.clear();

				cv::VideoCapture input_video(ofn.lpstrFile);
				if (!input_video.isOpened()) {
					
				}
				else {
					footage = input_video;
					video_input_ready = true;
				}
			}
		}

		if (cvui::button(gui, 12, 68, 128, 32, "Export Video")) {
			count++;
		}

		// GUI: Editor
		cvui::window(gui, 6, 112, 140, 180, "Editor");
		if (cvui::button(gui, 12, 138, 128, 32, "Load Image")) {
			if (GetOpenFileName(&ofn) == TRUE) {


			}
		}

		if (cvui::button(gui, 12, 174, 128, 32, "Export Video")) {
			count++;
		}

		// GUI: Previwer
		cvui::window(gui, 150, 6, 644, 504, "Preview");
		if (raw_sequence.empty()) {
			cvui::text(gui, 400, 256, "No Video/Images Loaded");
		}
		
		// GUI: Previewer Control
		cvui::window(gui, 150, 514, 644, 82, "Control");
		cvui::trackbar(gui, 158, 540, 512, &current_frame, (int)0, (int)sequence_length, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);
		cvui::counter(gui, 690, 539, &current_frame);
		if (cvui::button(gui, 690, 564, 92, 28, play_button_name)) {
			if (!raw_sequence.empty()) {

				if (is_video_play) {
					is_video_play = false;
					play_button_name = "Play";
				}
				else {
					// If reach the end frame
					if (current_frame == sequence_length) {
						current_frame = 0;
					}
					is_video_play = true;
					play_button_name = "Pause";
				}
			}
		}

		// Frame check
		if (current_frame < 0) {
			current_frame = 0;
		}
		if (current_frame > sequence_length) {
			current_frame = sequence_length;
		}

		// Start reading frames from selected video and store them into a vector
		while (video_input_ready) {
			cv::Mat frame;
			bool is_reading_video = footage.read(frame);

			// If reach the end of video
			if (!is_reading_video) {
				video_input_ready = false;
				sequence_length = preview_sequence.size()-1;
				processed_sequence = raw_sequence;
			}
			else {
				// Pass each frame to the image sequence vector
				raw_sequence.push_back(frame);
				cv::resize(frame, frame, cv::Size(640, 480));
				preview_sequence.push_back(frame);
			}
		}

		// Show the frame in the preview
		if (!raw_sequence.empty()) {
			if (current_frame < raw_sequence.size()) {
				cvui::image(gui, 152, 28, preview_sequence[current_frame]);
			}	

			// Play the sequence
			if (is_video_play) {
				cvui::image(gui, 152, 28, preview_sequence[current_frame]);
				if (current_frame + 1 == raw_sequence.size()) {
					is_video_play = false;
					play_button_name = "Play";
				}
				else {
					current_frame++;
				}

			}
			else {
			}

		}

		// Update the GUI
		cvui::update();
		cvui::imshow(WINDOW_NAME, gui);

		if (cv::waitKey(20) == 27) {
			break;
		}
	}

	return 0;

}
