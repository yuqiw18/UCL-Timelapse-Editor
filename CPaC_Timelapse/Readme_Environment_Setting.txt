1. Requirement: OpenCV 3.4.4
2. Edit Environment Variables - System Variables - Double Click [Path] - [New] - C:\Program Files\OpenCV\x64\vc15\bin
3. Visual Studio Project Setting:

C/C++ - General - Additional Include Directories
C:\Program Files\OpenCV\include;
%(AdditionalIncludeDirectories);
*** C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\include; (Optional if you have a OpenCV 3.4.4 with CUDA binary)

Linker - General - Additional Library Directories 
C:\Program Files\OpenCV\x64\vc15\lib;
*** C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v9.2\lib\x64; (Optional if you have a OpenCV 3.4.4 with CUDA binary)

Linker - Input - Additional Dependencies 
For Release:
opencv_world344.lib;
%(AdditionalDependencies)

For Debug:
opencv_world344d.lib;
%(AdditionalDependencies)

Replace C:\Program Files\ if the dependencies are installed at other places