# Import necessary libraries
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import os


app = Flask(__name__)


UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Load the necessary models
model = load_model('model.h5')
emotion_detection_model = load_model('emotions.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Function to detect faces
def detect_faces(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return [gray[y:y+h, x:x+w] for x, y, w, h in faces]


# Function to analyze emotions
def analyze_emotions(faces):
    emotions = []
    for face in faces:
        face_resized = cv2.resize(face, (96, 96))
        face_normalized = face_resized / 255.0
        face_expanded = np.expand_dims(face_normalized, axis=0)
        face_expanded = np.expand_dims(face_expanded, axis=-1)
        prediction = np.argmax(emotion_detection_model.predict(face_expanded))
        emotions.append(prediction)  # 0: Happy, 1: Loss, 2: Ordinary
    return max(set(emotions), key=emotions.count) if emotions else 2


# Process the video for goals
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    frame_count = 0
    goal_moments = []
    last_goal_time = None
    cool_down_period = (frame_rate / 2) * 45  # Considering every other frame

    while success:
        # Process every other frame
        if frame_count % 2 == 0:
            frame_resized = cv2.resize(frame, (224, 224)) / 255.0
            frame_reshaped = np.expand_dims(frame_resized, axis=0)
            prediction = model.predict(frame_reshaped)
            current_time = frame_count / frame_rate

            if prediction[0][0] < 0.5:
                if last_goal_time is None or (current_time - last_goal_time) > cool_down_period / frame_rate:
                    goal_moments.append(current_time)
                    last_goal_time = current_time

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return goal_moments


# Process the video for emotions
def process_video_for_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    frame_count = 0
    emotion_moments = []

    while success:
        # Process the first frame of each second
        if frame_count % frame_rate == 0:
            faces = detect_faces(frame)
            dominant_emotion = analyze_emotions(faces)

            if dominant_emotion in [0, 1]:  # If Happy or Loss is detected
                emotion_moments.append((frame_count / frame_rate, dominant_emotion))
                frame_count += int(frame_rate * 15)  # Skip 15 seconds
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                continue

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return emotion_moments


# Extract the highlight videos for goal moments
def extract_sub_videos(video_path, goal_moments):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    processed_videos = []
    for i, moment in enumerate(goal_moments):
        start_time = max(0, moment - 15)
        output_name = f"{base_name}_goal_{i + 1}.mp4"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        command = f"ffmpeg -ss {start_time} -i \"{video_path}\" -t 30 -c copy \"{output_path}\""
        subprocess.call(command, shell=True)
        processed_videos.append(output_name)
    return processed_videos


# Extract highlight videos for emotion moments
def extract_sub_videos_for_emotions(video_path, emotion_moments):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    for moment_time, emotion in emotion_moments:
        start_time = max(0, moment_time - 15)
        emotion_label = 'happy' if emotion == 0 else 'loss'
        output_name = f"{base_name}_{emotion_label}_{moment_time}.mp4"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        command = f"ffmpeg -ss {start_time} -i \"{video_path}\" -t 30 -c copy \"{output_path}\""
        subprocess.call(command, shell=True)


# Function the calls processing functions for goal and emotions moments.
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            goal_moments = process_video(file_path)
            emotion_moments = process_video_for_emotions(file_path)

            processed_videos_goal = extract_sub_videos(file_path, goal_moments) if goal_moments else []
            processed_videos_emotion = extract_sub_videos_for_emotions(file_path, emotion_moments) if emotion_moments else []

            return render_template('results.html', videos_goal=processed_videos_goal, videos_emotion=processed_videos_emotion)
        else:
            return "No significant moments detected in the video."
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
