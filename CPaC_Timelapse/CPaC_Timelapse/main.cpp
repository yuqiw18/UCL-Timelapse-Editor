/*
University College London
Department of Computer Science, CGVI
COMP0028: Computational Photography and Capture
Class Project Part B: Timelapse
Coded By Yuqi Wang (18043263)
*/
/*
Runtime: WINDOWS 10 X64 + OpenCV 3.4 + CUDA 9.1 + VS2017(v15)
OpenCV CUDA Binaries: https://yuqi.dev/tools/opencv340cuda91.zip
*/

#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include "tlt.h"
#include "utility.h"
#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

#define WINDOW_NAME "Time-Lapse Toolbox"
#define PADDING_HORIZONTAL 6
#define PADDING_VERTICAL 6

// State Machine (Non-OO)
const enum STATE { IDLE, LOAD, PROCESS, PLAY, SAVE };

int main(void){
	
	STATE CURRENT_STATE = STATE::IDLE;

	if (cv::cuda::getCudaEnabledDeviceCount() == 0) {
		std::cout << "No Cuda" << std::endl;	
	}

	// File browser
	OPENFILENAME ofn;       // common dialog box structure
	char szFile[260];       // buffer for file name
	HWND hwnd = NULL;              // owner window
	HANDLE hf;              // file handle

	// Initialise OPENFILENAME
	ZeroMemory(&ofn, sizeof(ofn));
	ofn.lStructSize = sizeof(ofn);
	ofn.hwndOwner = hwnd;
	ofn.lpstrFile = szFile;

	ofn.lpstrFile[0] = '\0';
	ofn.nMaxFile = sizeof(szFile);
	ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	ofn.nFilterIndex = 1;
	ofn.lpstrFileTitle = NULL;
	ofn.nMaxFileTitle = 0;
	ofn.lpstrInitialDir = NULL;
	ofn.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Initialise GUI
	cv::Mat gui = cv::Mat(650, 800, CV_8UC3);
	gui = cv::Scalar(49, 52, 49);
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);

	// Initialise variables
	cv::VideoCapture footage;
	std::vector<cv::Mat> raw_sequence;
	std::vector<cv::Mat> processed_sequence;
	std::vector<cv::Mat> preview_sequence;
	std::vector<cv::Mat> optical_flow;

	int current_clip = 0;
	int current_frame = 0;
	int sequence_length = 1;

	std::string PREVIEWER_BUTTON = "Play";
	std::string EDITOR_MODE = "Create TL";

	while (true) {
		
		// GUI: Read & Save Files
		cvui::window(gui, 6, 6, 140, 100, "File");
		if (cvui::button(gui, 12, 32, 128, 32,"Import(V/I)")) {
			if (GetOpenFileName(&ofn) == TRUE) {

				// Reset variables
				//READY_TO_LOAD = false;
				CURRENT_STATE = STATE::IDLE;
				current_frame = 0;
				sequence_length = 1;
				raw_sequence.clear();
				preview_sequence.clear();
				preview_sequence.clear();

				// Determine file type
				std::string file_path = UTIL::FilePathParser(ofn.lpstrFile);
				cv::VideoCapture input_video(file_path);
				if (!input_video.isOpened()) {
					std::cerr << "Invalid File" << std::endl;
				}
				else {
					footage = input_video;
					CURRENT_STATE = STATE::LOAD;
					//READY_TO_LOAD = true;
				}
			}
		}

		if (cvui::button(gui, 12, 68, 128, 32, "Export(V)")) {
			if (GetSaveFileName(&ofn) == TRUE) {
				
			}
		}

		// GUI: Editor
		cvui::window(gui, 6, 112, 140, 484, "Editor: "+EDITOR_MODE);
		if (cvui::button(gui, 12, 138, 128, 32, "Change Mode")) {
			if (EDITOR_MODE == "Create TL") {
				EDITOR_MODE = "Modify TL";
			}
			else {
				EDITOR_MODE = "Create TL";
			}
		}

		if (cvui::button(gui, 12, 174, 128, 32, "Retime")) {
			if (optical_flow.empty() || optical_flow.size() + 1 != raw_sequence.size()) {
				optical_flow = TLT::ComputeOpticalFlow(raw_sequence);
			}

			processed_sequence = TLT::RetimeSequence(raw_sequence, optical_flow, 3);

			preview_sequence.clear();

			for (int i = 0; i < processed_sequence.size(); i++) {
				
				cv::Mat current_frame = processed_sequence[i];
				cv::resize(current_frame, current_frame, cv::Size(640, 480));
				preview_sequence.push_back(current_frame);
			}

			sequence_length = preview_sequence.size() - 1;

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
		if (cvui::button(gui, 690, 564, 92, 28, PREVIEWER_BUTTON)) {
			if (!raw_sequence.empty()) {

				if (CURRENT_STATE == STATE::PLAY) {
					CURRENT_STATE = STATE::IDLE;
					PREVIEWER_BUTTON = "Play";
				}
				else {
					// If reach the end frame
					if (current_frame == sequence_length) {
						current_frame = 0;
					}
					CURRENT_STATE = STATE::PLAY;
					PREVIEWER_BUTTON = "Pause";
				}
			}
		}

		cvui::window(gui, 6, 600, 788, 44, "Output");
		cvui::text(gui, 12, 626, ">");

		// Frame check
		if (current_frame < 0) {
			current_frame = 0;
		}
		if (current_frame > sequence_length) {
			current_frame = sequence_length;
		}

		// Start reading frames from selected video and store them into a vector
		while (CURRENT_STATE == STATE::LOAD) {
			cv::Mat frame;
			bool is_reading_video = footage.read(frame);

			// If reach the end of video
			if (!is_reading_video) {
				CURRENT_STATE = STATE::IDLE;
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
			if (CURRENT_STATE == STATE::PLAY) {
				cvui::image(gui, 152, 28, preview_sequence[current_frame]);
				if (current_frame + 1 == raw_sequence.size()) {
					CURRENT_STATE = STATE::IDLE;
					PREVIEWER_BUTTON = "Play";
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
