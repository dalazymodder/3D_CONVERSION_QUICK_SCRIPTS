import cv2
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
from PIL import Image
import time

# Check if CUDA is available and set the device accordingly
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize the model
image_processor = AutoImageProcessor.from_pretrained("LiheYoung/depth-anything-large-hf")
model = AutoModelForDepthEstimation.from_pretrained("LiheYoung/depth-anything-large-hf").to(device)

# Open the video file
cap = cv2.VideoCapture('input.avi')  # replace 'input.mp4' with your video file path
frame_rate = 30.000
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi', fourcc, frame_rate, (int(cap.get(3)), int(cap.get(4))))

start_time = time.time()
processed_frames = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break

    # Convert the frame to PIL Image
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Prepare image for the model
    inputs = image_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    # Interpolate to original size
    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],
        mode="bicubic",
        align_corners=False,
    )

    # Visualize the prediction
    output = prediction.squeeze().cpu().numpy()
    formatted = (output * 255 / np.max(output)).astype("uint8")
    depth = Image.fromarray(formatted)

    # Convert the depth image to BGR for visualization with OpenCV
    depth_bgr = cv2.cvtColor(np.array(depth), cv2.COLOR_RGB2BGR)

    # Write the frame into the file 'output.avi'
    out.write(depth_bgr)

    processed_frames += 1
    elapsed_time = time.time() - start_time
    remaining_time = (elapsed_time / processed_frames) * (frame_count - processed_frames)
    eta_days, rem = divmod(remaining_time, 86400)
    eta_hours, rem = divmod(rem, 3600)
    eta_minutes, eta_seconds = divmod(rem, 60)
    print(f"ETA: {int(eta_days)} days, {int(eta_hours)} hours, {int(eta_minutes)} minutes, {int(eta_seconds)} seconds")
    print(f"Frames processed: {processed_frames}/{frame_count}")

# Release the video capture and close windows
cap.release()
out.release()
cv2.destroyAllWindows()
