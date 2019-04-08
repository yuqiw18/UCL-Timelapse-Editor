/*
University College London
Department of Computer Science, CGVI
COMP0028: Computational Photography and Capture
Class Project Part B: Timelapse
Coded By Yuqi Wang (18043263)
*/

/*
*Runtime: WINDOWS 10 X64 + OpenCV 3.4.4 + VS2017(v15)
OpenCV Binaries: https://opencv.org/opencv-4-0-0.html

*External Framework/Library/Plugin:
cvui(MIT License) https://github.com/Dovyski/cvui
*/

#include <opencv2/opencv.hpp>
#include <windows.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "core.h"
#include "utility.h"
#define CVUI_IMPLEMENTATION
#include "cvui/cvui.h"

#define WINDOW_NAME "Time-Lapse Toolbox"
#define WINDOW_WIDTH 850
#define WINDOW_HEIGHT 600
// State Machine (Non-OO)
const enum STATE { IDLE, LOAD, PROCESS, PLAY, SAVE };

int main(void){
	
	STATE CURRENT_STATE = STATE::IDLE;
	srand(time(NULL));

	/*
	if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
		HAS_CUDA = true;
		USE_CUDA = true;
	}
	*/

	// File browser
	// Reference: https://docs.microsoft.com/en-us/windows/desktop/api/commdlg/nf-commdlg-getopenfilenamea
	OPENFILENAME open_file_name;
	char szFile[256];
	HWND hwnd = NULL;
	HANDLE file_handler;

	// Initialise OPENFILENAME
	ZeroMemory(&open_file_name, sizeof(open_file_name));
	open_file_name.lStructSize = sizeof(open_file_name);
	open_file_name.hwndOwner = hwnd;
	open_file_name.lpstrFile = szFile;

	open_file_name.lpstrFile[0] = '\0';
	open_file_name.nMaxFile = sizeof(szFile);
	open_file_name.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
	open_file_name.nFilterIndex = 1;
	open_file_name.lpstrFileTitle = NULL;
	open_file_name.nMaxFileTitle = 0;
	open_file_name.lpstrInitialDir = NULL;
	open_file_name.Flags = OFN_PATHMUSTEXIST | OFN_FILEMUSTEXIST;

	// Initialise GUI
	cv::Mat gui = cv::Mat(WINDOW_HEIGHT, WINDOW_WIDTH, CV_8UC3);
	gui = cv::Scalar(49, 52, 49);
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);

	// Initialise variables
	cv::VideoCapture video_cache;
	std::vector<cv::Mat> raw_sequence;
	std::vector<cv::Mat> processed_sequence;
	std::vector<cv::Mat> optical_flow;
	std::vector<cv::Mat> mask_vintage;
	std::vector<cv::Mat> mask_miniature;
	cv::Mat gamma_lookup_table = core::GenerateGammaLookupTable(2.2);

	std::string EXPORT_PATH = "";
	std::string IMPORT_PATH = "";
	std::string PREVIEWER_BUTTON = "Play";

	bool INIT_VINTAGE_MASK = false;
	bool INIT_MINIATURE_MASK = false;
	bool HAS_CUDA = false;
	bool USE_CUDA = false;

	bool chk_enhance = false;
	bool chk_vintage = false;
	bool chk_miniature = false;
	bool chk_motion_trail = false;
	int val_interp_frame = 0;
	int val_import_fps = 1;
	int val_export_fps = 60;
	int current_frame = 0;
	int sequence_length = 1;

	// Load Vintage Filters
	cv::VideoCapture input_mask("appdata/mask_v/mask_v-01.png");
	if (!input_mask.isOpened()) {
		std::cout << "Cannot load vintage filters" << std::endl;
	}
	else {
		video_cache = input_mask;
	}
	while (!INIT_VINTAGE_MASK) {
		cv::Mat frame;
		bool is_reading_video = video_cache.read(frame);
		// If reach the end of video
		if (!is_reading_video) {
			INIT_VINTAGE_MASK = true;
			video_cache.release();
		}
		else {
			// Pass each frame to the image sequence vector
			mask_vintage.push_back(frame);
		}
	}

	// Load Miniature Filters
	cv::VideoCapture input_mask_m("appdata/mask_m/mask_m-01.png");
	if (!input_mask_m.isOpened()) {
		std::cout << "Cannot load miniature filters" << std::endl;
	}
	else {
		video_cache = input_mask_m;
	}
	while (!INIT_MINIATURE_MASK) {
		cv::Mat frame;
		bool is_reading_video = video_cache.read(frame);
		// If reach the end of video
		if (!is_reading_video) {
			INIT_MINIATURE_MASK = true;
			video_cache.release();
		}
		else {
			// Pass each frame to the image sequence vector
			mask_miniature.push_back(frame);
		}
	}

	// Main Program
	while (true) {
		
		// GUI: Read & Save Files
		cvui::window(gui, 6, 6, 190, 196, "File");
		if (cvui::button(gui, 12, 32, 178, 32,"Import (Video/Image)")) {
			open_file_name.lpstrFilter = "All\0*.*\0";
			if (GetOpenFileName(&open_file_name) == TRUE) {

				// Reset variables
				CURRENT_STATE = STATE::IDLE;
				current_frame = 0;
				sequence_length = 1;
				raw_sequence.clear();
				processed_sequence.clear();
				optical_flow.clear();

				// Determine file type
				IMPORT_PATH = utility::FilePathParser(open_file_name.lpstrFile);
				cv::VideoCapture input_video(IMPORT_PATH);
				if (!input_video.isOpened()) {
					std::cout << "Invalid File" << std::endl;
				}
				else {
					video_cache = input_video;
					CURRENT_STATE = STATE::LOAD;
				}
			}
		}
		cvui::text(gui, 12, 82, "FPS(I)");
		cvui::trackbar(gui, 50, 65, 148, &val_import_fps, (int)1, (int)60, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);

		if (cvui::button(gui, 12, 116, 178, 32, "Export (Video)")) {
			open_file_name.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
			if (GetSaveFileName(&open_file_name) == TRUE) {
				EXPORT_PATH = open_file_name.lpstrFile;
				CURRENT_STATE = STATE::SAVE;
			}
		}
		cvui::text(gui, 12, 166, "FPS(O)");
		cvui::trackbar(gui, 50, 149, 148, &val_export_fps, (int)1, (int)60, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);

		// GUI: Retiming
		cvui::window(gui, 6, 206, 190, 98, "Retiming (Interpolation)");
		cvui::text(gui, 12, 247, "Frame");
		cvui::trackbar(gui, 50, 230, 148, &val_interp_frame, (int)0, (int)60, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);
		cvui::checkbox(gui, 12, 282, "Image Enhancement", &chk_enhance);

		// GUI: Visual Effect
		cvui::window(gui, 6, 308, 190, 106, "Visual Effect");
		cvui::checkbox(gui, 12, 336, "Vintage (Scenery)", &chk_vintage);
		cvui::checkbox(gui, 12, 364, "Miniature (City)", &chk_miniature);
		cvui::checkbox(gui, 12, 392, "Motion Trail (People)", &chk_motion_trail);

		cvui::window(gui, 6, 418, 190, 100, "Operation");
		if (cvui::button(gui, 12, 444, 178, 32, "Proccess")) {
			if (!processed_sequence.empty()) {
				if (val_interp_frame > 0) {
					if (optical_flow.empty() || optical_flow.size() + 1 != raw_sequence.size()) {
						optical_flow = core::ComputeOpticalFlow(raw_sequence, USE_CUDA);
					}
					else {
						std::cout << "Opitcal Flow Already Computed" << std::endl;
					}
				}

				processed_sequence = raw_sequence;

				if (val_interp_frame > 0) {
					processed_sequence = core::RetimeSequence(processed_sequence, optical_flow, val_interp_frame);
				}

				// Preprocessing
				if (chk_enhance) {
					
					processed_sequence = core::HistogramAnalysis(processed_sequence);
					processed_sequence = core::ContrastStretching(processed_sequence);
				}

				// Postprocessing
				if (chk_motion_trail) {
					processed_sequence = core::ApplyMotionTrail(processed_sequence, core::GenerateMotionTrail(processed_sequence));
				}

				if (chk_miniature) {
					processed_sequence = core::Miniature(processed_sequence, mask_miniature[4]);
				}

				if (chk_vintage) {
					processed_sequence = core::Vintage(processed_sequence, mask_vintage);

				}
				processed_sequence = core::ContrastStretching(processed_sequence);

				sequence_length = processed_sequence.size() - 1;
			}
		}

		if (cvui::button(gui, 12, 480, 178, 32, "Reset")) {
			CURRENT_STATE = STATE::IDLE;
			current_frame = 0;
			sequence_length = 1;
			raw_sequence.clear();
			processed_sequence.clear();
			cv::VideoCapture input_video(IMPORT_PATH);
			if (!input_video.isOpened()) {
				std::cerr << "Invalid File" << std::endl;
			}
			else {
				video_cache = input_video;
				CURRENT_STATE = STATE::LOAD;
			}

		}

		cvui::window(gui, 6, 522, 190, 52, "Option");
		if (HAS_CUDA) {
			cvui::checkbox(gui, 12, 550, "Use CUDA", &USE_CUDA);
		}
		else {
			cvui::text(gui, 12, 552, "CUDA Disabled");
		}

		// GUI: Previwer
		if (!raw_sequence.empty()) {
			std::string preiviewer_title = std::string("Preview [640x480@30FPS] ") + ":: Raw [" + std::to_string(processed_sequence.front().cols) + "x" + std::to_string(processed_sequence.front().rows) + "] Length(" + std::to_string(val_export_fps) + "FPS): " + utility::ConvertFPStoTime(processed_sequence.size(), val_export_fps);
			cvui::window(gui, 200, 6, 644, 504, preiviewer_title);
		}
		else {
			cvui::window(gui, 200, 6, 644, 504, "Preview [640x480@30FPS]");
		}

		if (raw_sequence.empty()) {
			cvui::text(gui, 450, 256, "No Video/Images Loaded");
		}
		
		// GUI: Previewer Control
		cvui::window(gui, 200, 514, 644, 82, "Control");
		cvui::trackbar(gui, 208, 540, 512, &current_frame, (int)0, (int)sequence_length, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);
		cvui::counter(gui, 740, 539, &current_frame);
		if (cvui::button(gui, 740, 564, 92, 28, PREVIEWER_BUTTON)) {
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

		// Frame check
		if (current_frame < 0) {
			current_frame = 0;
		}
		if (current_frame > sequence_length) {
			current_frame = sequence_length;
		}

		int frame_count = 0;

		// Start reading frames from selected video and store them into a vector
		while (CURRENT_STATE == STATE::LOAD) {
			cv::Mat frame;
			bool is_reading_video = video_cache.read(frame);
			

			// If reach the end of video
			if (!is_reading_video) {
				CURRENT_STATE = STATE::IDLE;

				// Add an extra frame to avoid trackbar error
				if (raw_sequence.size() == 1) {
					raw_sequence.push_back(raw_sequence.front());
				}

				processed_sequence = raw_sequence;		
				sequence_length = processed_sequence.size() - 1;
				video_cache.release();
			}
			else {
				// Pass each frame to the image sequence vector
				if (frame_count % val_import_fps == 0) {
					raw_sequence.push_back(frame);
				}
				frame_count++;
			}
		}

		// Saving
		while (CURRENT_STATE == STATE::SAVE) {
			if (!processed_sequence.empty()) {
				cv::VideoWriter video_writer(EXPORT_PATH + "_output.avi", CV_FOURCC('M', 'J', 'P', 'G'), val_export_fps, processed_sequence.front().size());
				for (int f = 0; f < processed_sequence.size(); f++) {
					video_writer.write(processed_sequence[f]);
				}
				std::cout << "Saved" << std::endl;
				video_writer.release();
				CURRENT_STATE = STATE::IDLE;
			}	
		}

		// Show the frame in the preview
		if (!raw_sequence.empty()) {
			if (current_frame < processed_sequence.size()) {
				cv::Mat preview_frame = processed_sequence[current_frame];
				cv::resize(preview_frame, preview_frame, cv::Size(640, 480));
				cvui::image(gui, 202, 28, preview_frame);
			}	

			// Play the sequence
			if (CURRENT_STATE == STATE::PLAY) {
				cv::Mat preview_frame = processed_sequence[current_frame];
				cv::resize(preview_frame, preview_frame, cv::Size(640, 480));
				cvui::image(gui, 202, 28, preview_frame);
				if (current_frame + 1 == processed_sequence.size()) {
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