import cv2
import numpy as np

# Function to generate stereo pair from 2D and depth frames
def generate_stereo_pair(frame_2d, frame_depth, disparity_scale):
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

    # Shift original color frame based on disparity
    shift_matrix_left = np.float32([[1, 0, -max_shift], [0, 1, 0]])
    shift_matrix_right = np.float32([[1, 0, max_shift], [0, 1, 0]])

    shifted_left = cv2.warpAffine(frame_2d, shift_matrix_left, (frame_2d.shape[1], frame_2d.shape[0]))
    shifted_right = cv2.warpAffine(frame_2d, shift_matrix_right, (frame_2d.shape[1], frame_2d.shape[0]))

    # Combine shifted frames into a side-by-side format
    stereo_pair = np.hstack((shifted_left, shifted_right))

    return stereo_pair

# Example usage
def process_video(video_2d_path, video_depth_path, output_video_path, disparity_scale, frame_rate_var):
    # Open video streams
    video_2d = cv2.VideoCapture(video_2d_path)
    video_depth = cv2.VideoCapture(video_depth_path)

    # Get frame dimensions and frame rate
    frame_width = int(video_2d.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_2d.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = frame_rate_var

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (2 * frame_width, frame_height))

    # Process video frames
    while True:
        ret_2d, frame_2d = video_2d.read()
        ret_depth, frame_depth = video_depth.read()

        if not ret_2d or not ret_depth:
            break

        stereo_pair = generate_stereo_pair(frame_2d, frame_depth, disparity_scale)
        out.write(stereo_pair)

    # Release video streams and close output video
    video_2d.release()
    video_depth.release()
    out.release()
    cv2.destroyAllWindows()

# Example usage
video_2d_path = 'video_2d.avi'
video_depth_path = 'video_depth.avi'
output_video_path = 'output_video.avi'
disparity_scale = 3.0  # Example value, adjust as needed
frame_rate_var = 29.976


process_video(video_2d_path, video_depth_path, output_video_path, disparity_scale, frame_rate_var)
