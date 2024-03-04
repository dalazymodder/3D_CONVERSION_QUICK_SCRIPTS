# 3D_CONVERSION_QUICK_SCRIPTS

This Repo contains some quick scripts I use for 3d Conversions.

They require depth maps to be generated for generating depth maps I recommend depth anything by tiktok.

https://github.com/LiheYoung/Depth-Anything

[INSTALLATION]
Install python (I used version 3.10), and then 
open command prompt and run commands "pip install opencv-python" and "pip install numpy" without quotes.

From there to generate a depth map you can download depth anything from here.
https://github.com/LiheYoung/Depth-Anything
Then drop the Simple_Depth_Anything.py into the same directory.
From there open the Simple_Depth_Anything.py and edit the lines

cap = cv2.VideoCapture('input.avi')
frame_rate = 30.000

input.avi should be your video file path
and frame_rate should be its frame rate like 29.976 for a lot of movies, you can check it with vlc by open the file going to tools and then codec information.

From there it will generate a 2D depth map video for you.

Now to run 3D_SBS_CONVERT_LUME.py you will need to scroll down in the file and edit the lines 

video_2d_path = 'video_2d.avi'
video_depth_path = 'video_depth.avi'
output_video_path = 'output_video.avi'
disparity_scale = 3.0
frame_rate_var = 29.976


The lines should be obvious except for maybe disparity_scale that is the strength of the 3d effect too high and it will look like double vision.
Recommend you test with quick clips and see what you like before doing a full blown long convert.

This leaves the last script. It is for making lightfield _2x2 videos for android devices such as the lume pad. It requires a depth map and a 2d video source to run. To use the script is saame as lumefield_lightfield.py but you can specify the number of views.


video_2d_path = 'video_2d.avi'
video_depth_path = 'video_depth.avi'
output_video_path = 'output_video.avi'
disparity_scale = 3.0
frame_rate_var = 29.976

NOTES converted videos processed by opencv as XVID the format I have all the scripts defaulted to use may need reencoded by something like ffmpeg to play on all devices like lume pad.

Final Notes:
The Simple_Depth_Anything_Depth.py can use cuda this will significantly increase the speed up the script but running Simple_Depth_Anything_Depth.py and 3D_SBS_CONVERT.py on something like a 2 hour movie will take at least a few days to process.

You can improve the Simple_Depth_Anything_Depth.py speed by using something like https://github.com/spacewalk01/depth-anything-tensorrt to leverage tensor cores on rtx gpus to speed up making frames I'm still working on a tweak version of fork of that which is promising but it requires a lot of depenencies for it work right and requires nvidia gpu.



