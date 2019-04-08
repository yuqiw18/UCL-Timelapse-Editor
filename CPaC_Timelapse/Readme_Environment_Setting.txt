1. Requirement: OpenCV 3.4 + CUDA 9.1
2. Edit Environment Variables - System Variables - Double Click [Path] - [New] - C:\Program Files\OpenCV\x64\vc15\bin
3. Visual Studio Project Setting:

C/C++ - General - Additional Include Directories
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\include;
C:\Program Files\OpenCV\include;
%(AdditionalIncludeDirectories)

Linker - General - Additional Library Directories 
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.1\lib\x64;
C:\Program Files\OpenCV\x64\vc15\lib;

Linker - Input - Additional Dependencies 
For Release:
opencv_world340.lib;
%(AdditionalDependencies)

For Debug:
opencv_world340d.lib;
%(AdditionalDependencies)

Replace C:\Program Files\ if the dependencies are installed at other places