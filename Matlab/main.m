path = 'ofdata';
prefix = 'gjbLookAtTarget_';
first = 0;
last = 10;
digits = 4;
suffix = 'jpg';

rawImageSequence = load_sequence_color(path, prefix, first, last, digits, suffix);
[w,h,c,n] = size(rawImageSequence);
flow = [];

opticalFlow = opticalFlowFarneback

tic
for i = 1: 11
    
    frameGrayscale = rgb2gray(rawImageSequence(:,:,:,i));
    currentFlow = estimateFlow(opticalFlow,frameGrayscale); 
    flow = [flow, currentFlow];
    
end
toc

% vidReader = VideoReader('test_clip.mp4');
% opticFlow = opticalFlowFarneback
% while hasFrame(vidReader)
%     frameRGB = readFrame(vidReader);
%     frameSequence = [frameSequence, frameRGB];
%     frameGray = rgb2gray(frameRGB);  
%     flow = estimateFlow(opticFlow,frameGray); 
%     opticalFlowx = [opticalFlow, flowx];
%     
%     disp(count)
%     count = count + 1;
%     
% end