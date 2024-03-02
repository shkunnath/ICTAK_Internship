import tkinter as tk
from tkinter import filedialog
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import os

# Load the model
model = load_model('model.h5')


# Create a function that captures the video and convert it into frames
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    frame_count = 0
    goal_moments = []
    last_goal_time = None
    cool_down_period = frame_rate * 45  # 45 seconds cool-down to group goal detections

    while success:
        # Preprocess the frame
        frame_resized = cv2.resize(frame, (224, 224)) / 255.0
        frame_reshaped = np.expand_dims(frame_resized, axis=0)

        # Predict
        prediction = model.predict(frame_reshaped)
        current_time = frame_count / frame_rate

        if prediction[0][0] < 0.5:
            if last_goal_time is None or (current_time - last_goal_time) > cool_down_period / frame_rate:
                # New goal detected or outside of cool-down period
                goal_moments.append(current_time)
                last_goal_time = current_time
            # If a goal is detected within the cool-down period, it's considered part of the previous goal event

        success, frame = cap.read()
        frame_count += 1

    cap.release()

    return goal_moments


# Create a small video highlight.
def extract_sub_videos(video_path, goal_moments):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    for i, moment in enumerate(goal_moments):
        start_time = max(0, moment - 30)
        output_name = f"{base_name}_goal_{i + 1}.mp4"

        # FFmpeg command to extract sub-video
        command = f"ffmpeg -ss {start_time} -i \"{video_path}\" -t 60 -c copy \"{output_name}\""
        subprocess.call(command, shell=True)


# Ask user to upload the file
def browse_file():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    return file_path


# The main function
if __name__ == "__main__":
    video_file = browse_file()
    if video_file:
        goal_moments = process_video(video_file)
        if goal_moments:
            extract_sub_videos(video_file, goal_moments)
            print(f"Sub-videos created for each detected goal.")
        else:
            print("No goals detected in the video.")
