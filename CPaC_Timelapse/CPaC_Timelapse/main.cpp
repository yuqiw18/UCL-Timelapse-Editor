/*
University College London
Department of Computer Science, CGVI
COMP0028: Computational Photography and Capture
Class Project Part B: Timelapse
Coded By Yuqi Wang (18043263)
*/

/*
*Runtime: WINDOWS 10 X64 + OpenCV 3.4 + CUDA 9.1 + VS2017(v15)
OpenCV CUDA Binaries: https://yuqi.dev/tools/opencv340cuda91.zip

*External Framework/Library/Plugin:
cvui(MIT License) https://github.com/Dovyski/cvui

*Software/Applications Used for Data Collection:
Timer Camera (Android) https://play.google.com/store/apps/details?id=com.cae.timercamera&hl=en_GB
	For taking photos at given time interval
Image Resizer https://www.bricelam.net/ImageResizer/
	For bulk resizing(downsizing) images
Bulk Rename Utility https://www.bulkrenameutility.co.uk/Main_Intro.php
	For bulk renaming image names to a specific format.
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

#define WINDOW_NAME "Time-Lapse/Slow-Mo Toolbox"
#define PADDING_HORIZONTAL 6
#define PADDING_VERTICAL 6

// State Machine (Non-OO)
const enum STATE { IDLE, LOAD, PROCESS, PLAY, SAVE };
// Prompt
std::string EXPORT_PATH ="";
std::string IMPORT_PATH = "";

bool chk_enhance = false;
bool chk_vintage = false;
bool chk_miniature = false;
bool chk_motion_trail = false;
bool chk_lomo = false;
int val_interp_frame = 0;
int val_import_fps = 1;
int val_export_fps = 60;

bool INIT_VINTAGE_MASK = false;
bool HAS_CUDA = false;
bool USE_CUDA = false;

int main(void){
	
	STATE CURRENT_STATE = STATE::IDLE;
	
srand(time(NULL));

	if (cv::cuda::getCudaEnabledDeviceCount() != 0) {
		HAS_CUDA = true;
		USE_CUDA = true;
	}

	// File browser
	// Reference: https://docs.microsoft.com/en-us/windows/desktop/api/commdlg/nf-commdlg-getopenfilenamea
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
	cv::Mat gui = cv::Mat(600, 850, CV_8UC3);
	gui = cv::Scalar(49, 52, 49);
	cv::namedWindow(WINDOW_NAME);
	cvui::init(WINDOW_NAME);

	// Initialise variables
	cv::VideoCapture video_cache;
	std::vector<cv::Mat> raw_sequence;
	std::vector<cv::Mat> processed_sequence;
	std::vector<cv::Mat> optical_flow;
	std::vector<cv::Mat> mask_vintage;
	cv::Mat gamma_lookup_table = core::GenerateGammaLookupTable(2.2);

	int current_frame = 0;
	int sequence_length = 1;

	std::string PREVIEWER_BUTTON = "Play";

	//core *toolbox = new core();

	cv::VideoCapture input_mask("data/mask_v/mask_v-01.png");
	if (!input_mask.isOpened()) {
		HAS_CUDA = false;
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


	while (true) {
		
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// GUI: Read & Save Files
		cvui::window(gui, 6, 6, 190, 196, "File");
		if (cvui::button(gui, 12, 32, 178, 32,"Import (Video/Image)")) {
			ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
			if (GetOpenFileName(&ofn) == TRUE) {

				// Reset variables
				CURRENT_STATE = STATE::IDLE;
				current_frame = 0;
				sequence_length = 1;
				raw_sequence.clear();
				processed_sequence.clear();
				optical_flow.clear();

				// Determine file type
				IMPORT_PATH = utility::FilePathParser(ofn.lpstrFile);
				cv::VideoCapture input_video(IMPORT_PATH);
				if (!input_video.isOpened()) {
					std::cerr << "Invalid File" << std::endl;
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
			ofn.lpstrFilter = "All\0*.*\0Text\0*.TXT\0";
			if (GetSaveFileName(&ofn) == TRUE) {
				EXPORT_PATH = ofn.lpstrFile;
				CURRENT_STATE = STATE::SAVE;
			}
		}
		cvui::text(gui, 12, 166, "FPS(O)");
		cvui::trackbar(gui, 50, 149, 148, &val_export_fps, (int)1, (int)60, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);
		

		// GUI: Editor
		cvui::window(gui, 6, 206, 190, 98, "Retiming (Interpolation)");
		cvui::text(gui, 12, 247, "Frame");
		cvui::trackbar(gui, 50, 230, 148, &val_interp_frame, (int)0, (int)60, 1, "%.0Lf", cvui::TRACKBAR_DISCRETE, (int)1);
		cvui::checkbox(gui, 12, 282, "Image Enhancement", &chk_enhance);

		cvui::window(gui, 6, 308, 190, 136, "Filter");
		cvui::checkbox(gui, 12, 336, "Vintage (Scenery)", &chk_vintage);
		cvui::checkbox(gui, 12, 364, "Miniature (City)", &chk_miniature);
		cvui::checkbox(gui, 12, 392, "Motion Trail (People)", &chk_motion_trail);
		cvui::checkbox(gui, 12, 420, "Lomo (General)", &chk_lomo);

		cvui::window(gui, 6, 448, 190, 100, "Operation");
		if (cvui::button(gui, 12, 474, 178, 32, "Proccess")) {
			if (!processed_sequence.empty()) {
				if (val_interp_frame > 0) {
					if (optical_flow.empty() || optical_flow.size() + 1 != raw_sequence.size()) {
						optical_flow = core::ComputeOpticalFlow(raw_sequence, USE_CUDA);
					}
					else {
						std::cout << "Opitcal Flow Already Computed" << std::endl;
					}
				
				}

				if (val_interp_frame > 0) {
					processed_sequence = core::RetimeSequence(processed_sequence, optical_flow, val_interp_frame);
				}

				if (chk_enhance) {
					processed_sequence = core::EnhanceImage(processed_sequence);
				}

				if (chk_vintage) {
					processed_sequence = core::VintageFilter(processed_sequence, mask_vintage);
				}
				else if (chk_miniature) {
				
				
				}
				else if (chk_motion_trail) {
					processed_sequence = core::ApplyMotionTrail(processed_sequence, core::GenerateMotionTrail(processed_sequence));
				}
				else if (chk_lomo) {
				
				
				}

				sequence_length = processed_sequence.size() - 1;
			}
		}

		if (cvui::button(gui, 12, 510, 178, 32, "Reset")) {
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

		cvui::window(gui, 6, 552, 190, 44, "Option");
		if (HAS_CUDA) {
			cvui::checkbox(gui, 12, 576, "Use CUDA", &USE_CUDA);
		}
		else {
			cvui::text(gui, 12, 576, "No CUDA Device");
		}



		// GUI: Previwer
		cvui::window(gui, 200, 6, 644, 504, "Preview");
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
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////



		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
		// Filter check

		
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
		/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

		// Update the GUI
		cvui::update();
		cvui::imshow(WINDOW_NAME, gui);

		if (cv::waitKey(20) == 27) {
			break;
		}
	}

	return 0;

}