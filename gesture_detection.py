import cv2
import numpy as np
from ultralytics import YOLO

# Load the pre-trained YOLO gesture detection model
model = YOLO("/home/exouser/voxel51-hack/ElderWatch/Models/yolo-gesture/YOLOv10x_gestures.pt")

# Open the video capture (replace 0 with the path to the local video file)
video_path = "./Data/real/gestures.mp4"  # Replace with your local video file path
cap = cv2.VideoCapture(video_path)

# Get video properties to set up the output video file
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Define the output path
output_path = r"fall_call_gesture_output_video.mp4"

# Create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can use other codecs like 'XVID'
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_count = 0
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break
        
    # Run YOLOv8 inference on the frame
    results = model(frame)
        
    # Visualize the results on the frame
    annotated_frame = results[0].plot()
    
    # Write the frame to the output video file
    out.write(annotated_frame)
    
    frame_count += 1
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames")

# Release the video capture and writer objects
cap.release()
out.release()

print(f"Output video saved to {output_path}")