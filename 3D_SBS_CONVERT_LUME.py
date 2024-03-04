import cv2
import numpy as np
import time

def generate_lightfield_video(video_path_2d, video_path_depth, disparity_scale, num_views, output_video_path, frame_rate):
    # Open video streams
    cap_2d = cv2.VideoCapture(video_path_2d)
    cap_depth = cv2.VideoCapture(video_path_depth)

    # Check if video streams are opened successfully
    if not cap_2d.isOpened() or not cap_depth.isOpened():
        print("Error: Unable to open video streams.")
        return

    # Get video properties
    width = int(cap_2d.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_2d.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap_2d.get(cv2.CAP_PROP_FPS))

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width*2, height*2))

    total_frames = min(int(cap_2d.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT)))
    current_frame = 0
    start_time = time.time()

    while True:
        ret_2d, frame_2d = cap_2d.read()
        ret_depth, frame_depth = cap_depth.read()

        if not ret_2d or not ret_depth:
            break

        # Convert frames to grayscale
        frame_2d_gray = cv2.cvtColor(frame_2d, cv2.COLOR_BGR2GRAY)
        frame_depth_gray = cv2.cvtColor(frame_depth, cv2.COLOR_BGR2GRAY)

        # Compute disparity map
        stereo = cv2.StereoBM_create(numDisparities=16, blockSize=15)
        disparity_map = stereo.compute(frame_2d_gray, frame_depth_gray)

        # Normalize disparity map
        disparity_map_normalized = cv2.normalize(disparity_map, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

        # Displace pixels based on the disparity map
        disparity_scale_value = disparity_scale / 10.0
        disparity_shift = disparity_map_normalized.astype(np.float32) * disparity_scale_value

        # Calculate maximum shift
        max_shift = np.max(disparity_shift)

        # Generate multiple shifted images from different viewpoints
        lightfield_images = []
        for i in range(num_views):
            shift = max_shift * (i / (num_views - 1) - 0.5)
            shift_matrix = np.float32([[1, 0, shift], [0, 1, 0]])
            shifted_image = cv2.warpAffine(frame_2d, shift_matrix, (frame_2d.shape[1], frame_2d.shape[0]))
            lightfield_images.append(shifted_image)

        # Combine images into one
        top = cv2.hconcat([lightfield_images[0], lightfield_images[1]])
        bottom = cv2.hconcat([lightfield_images[2], lightfield_images[3]])
        combined_image = cv2.vconcat([top, bottom])

        # Write frame to output video
        out.write(combined_image)

        current_frame += 1
        progress_percent = (current_frame / total_frames) * 100
        elapsed_time = time.time() - start_time
        estimated_total_time = elapsed_time / (progress_percent / 100)
        remaining_time = estimated_total_time - elapsed_time

        hours = int(remaining_time // 3600)
        minutes = int((remaining_time % 3600) // 60)
        seconds = int(remaining_time % 60)

        print(f"Progress: {progress_percent:.2f}%, ETA: {hours}h {minutes}m {seconds}s")

    # Release video streams and writer
    cap_2d.release()
    cap_depth.release()
    out.release()
    cv2.destroyAllWindows()

video_path_2d = 'Rose.mp4'
video_path_depth = 'myvid.mp4'
disparity_scale = 2.0
output_video_path = 'result.mp4'
num_views = 4
frame_rate = 29.00

generate_lightfield_video(video_path_2d, video_path_depth, disparity_scale, num_views, output_video_path, frame_rate)

import subprocess

def reencode_video(input_file, output_file):
    command = ['ffmpeg', '-i', input_file, '-c:v', 'libx264', '-crf', '23', '-preset', 'fast', '-c:a', 'aac', '-b:a', '128k', output_file]
    subprocess.run(command)

input_file = 'result.mp4'
output_file = 'reencoded_result.mp4'

reencode_video(input_file, output_file)