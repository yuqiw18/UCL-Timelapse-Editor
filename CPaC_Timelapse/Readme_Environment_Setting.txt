1. Requirement: OpenCV 3.4.4
2. Edit Environment Variables - System Variables - Double Click [Path] - [New] - C:\Program Files\OpenCV\x64\vc15\bin
3. Visual Studio Project Setting:

***C/C++ - General - Additional Include Directories
C:\Program Files\OpenCV\include;
%(AdditionalIncludeDirectories);

***Linker - General - Additional Library Directories 
C:\Program Files\OpenCV\x64\vc15\lib;

***Linker - Input - Additional Dependencies 
For Release:
opencv_world344.lib;
%(AdditionalDependencies)

For Debug:
opencv_world344d.lib;
%(AdditionalDependencies)

Replace C:\Program Files\ if the dependencies are installed at other places