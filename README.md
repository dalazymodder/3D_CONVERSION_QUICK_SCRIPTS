# 3D_CONVERSION_QUICK_SCRIPTS

This repository contains quick scripts used for 3D conversions. These scripts require depth maps to be generated. For generating depth maps, Depth-Anything by TikTok is recommended.

## Installation

1. Install Python (Version 3.10 was used in this case).
2. Open the command prompt and run the following commands:
    ```bash
    pip install opencv-python
    pip install numpy
    ```
   On the first run, it will download the model for Depth-Anything.

## Usage

To generate a depth map, follow these steps:

1. Download Depth-Anything from here.
2. Drop the `Simple_Depth_Anything.py` into the same directory.
3. Open the `Simple_Depth_Anything.py` and edit the lines:
    ```python
    cap = cv2.VideoCapture('input.avi')
    frame_rate = 30.000
    ```
   Here, 'input.avi' should be your video file path and `frame_rate` should be its frame rate (like 29.976 for a lot of movies). You can check it with VLC by opening the file, going to tools, and then codec information. On the first run, it will download the model for Depth-Anything.

This will generate a 2D depth map video for you.

To run `3D_SBS_CONVERT_LUME.py`, you will need to scroll down in the file and edit the lines:

```python
video_2d_path = 'video_2d.avi'
video_depth_path = 'video_depth.avi'
output_video_path = 'output_video.avi'
disparity_scale = 3.0
frame_rate_var = 29.976

The disparity_scale is the strength of the 3D effect. If it’s too high, it will look like double vision. It’s recommended to test with quick clips and see what you like before doing a full-blown long convert.

The last script is for making lightfield _2x2 videos for Android devices such as the Lume Pad. It requires a depth map and a 2D video source to run. To use the script, it’s the same as lumefield_lightfield.py, but you can specify the number of views.

video_2d_path = 'video_2d.avi'
video_depth_path = 'video_depth.avi'
output_video_path = 'output_video.avi'
disparity_scale = 3.0
frame_rate_var = 29.976


Notes

Converted videos processed by OpenCV as XVID (the format all the scripts are defaulted to use) may need to be re-encoded by something like FFmpeg to play on all devices like Lume Pad.
Final Notes

The Simple_Depth_Anything_Depth.py can use CUDA. This will significantly speed up the script, but running Simple_Depth_Anything_Depth.py and 3D_SBS_CONVERT.py on something like a 2-hour movie will take at least a few days to process.

You can improve the Simple_Depth_Anything_Depth.py speed by using something like Depth-Anything-TensorRT to leverage tensor cores on RTX GPUs to speed up making frames. A tweaked version of this fork is currently being worked on, which is promising, but it requires a lot of dependencies to work right and requires an NVIDIA GPU.
