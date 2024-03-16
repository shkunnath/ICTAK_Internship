# Import necessary libraries
from flask import Flask, render_template, request, redirect, send_from_directory
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import subprocess
import os

# Initialize Flask app
app = Flask(__name__)

# Define the folder to store uploaded files
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Route to display uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    # Send the file from the uploads directory
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Load trained models and Haar cascade for face detection
model = load_model('model.h5')
emotion_detection_model = load_model('emotions.h5')
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Function to detect faces in an image
def detect_faces(frame):
    # Convert image to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Return the grayscale face regions
    return [gray[y:y+h, x:x+w] for x, y, w, h in faces]

# Function to analyze emotions in detected faces
def analyze_emotions(faces):
    emotions = []  # List to store emotions of all faces
    for face in faces:
        # Resize face for the emotion model
        face_resized = cv2.resize(face, (96, 96))
        # Normalize pixel values
        face_normalized = face_resized / 255.0
        # Expand dimensions to match model input
        face_expanded = np.expand_dims(face_normalized, axis=0)
        face_expanded = np.expand_dims(face_expanded, axis=-1)
        # Predict emotion and append to list
        prediction = np.argmax(emotion_detection_model.predict(face_expanded))
        emotions.append(prediction)  # 0: Happy, 1: Loss, 2: Ordinary
    # Return the most frequent emotion, default to 'Ordinary'
    return max(set(emotions), key=emotions.count) if emotions else 2

# Function to process video and detect goals
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # Get frame rate
    success, frame = cap.read()  # Read first frame
    frame_count = 0
    goal_moments = []  # List to store times of detected goals
    last_goal_time = None  # Track time of last detected goal
    cool_down_period = (frame_rate / 2) * 45  # Cooldown period to avoid repeated detections

    while success:
        # Process every other frame to save computation
        if frame_count % 2 == 0:
            frame_resized = cv2.resize(frame, (224, 224)) / 255.0  # Resize and normalize frame
            frame_reshaped = np.expand_dims(frame_resized, axis=0)
            prediction = model.predict(frame_reshaped)  # Predict goal
            current_time = frame_count / frame_rate  # Calculate current time in video

            # If a goal is detected and cooldown has passed
            if prediction[0][0] < 0.5:
                if last_goal_time is None or (current_time - last_goal_time) > cool_down_period / frame_rate:
                    goal_moments.append(current_time)  # Add goal moment
                    last_goal_time = current_time

        success, frame = cap.read()  # Read next frame
        frame_count += 1

    cap.release()
    return goal_moments  # Return list of goal moments

# Function to process video and analyze emotions
def process_video_for_emotions(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    success, frame = cap.read()
    frame_count = 0
    emotion_moments = []  # List to store moments with detected emotions

    while success:
        # Process the first frame of each second
        if frame_count % frame_rate == 0:
            faces = detect_faces(frame)  # Detect faces
            dominant_emotion = analyze_emotions(faces)  # Analyze emotions

            # If Happy or Loss is detected
            if dominant_emotion in [0, 1]:
                emotion_moments.append((frame_count / frame_rate, dominant_emotion))
                frame_count += int(frame_rate * 15)  # Skip 15 seconds to next analysis
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                continue

        success, frame = cap.read()
        frame_count += 1

    cap.release()
    return emotion_moments  # Return moments of detected emotions

# Function to extract sub-videos around detected goal moments
def extract_sub_videos(video_path, goal_moments):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    processed_videos = []  # List to store names of processed videos
    for i, moment in enumerate(goal_moments):
        start_time = max(0, moment - 15)  # Start 15 seconds before the goal
        output_name = f"{base_name}_goal_{i + 1}.mp4"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        # Use ffmpeg to cut the video
        command = f"ffmpeg -ss {start_time} -i \"{video_path}\" -t 30 -c copy \"{output_path}\""
        subprocess.call(command, shell=True)  # Execute the command
        processed_videos.append(output_name)
    return processed_videos  # Return list of processed video names

# Function to extract highlight videos for emotion moments
def extract_sub_videos_for_emotions(video_path, emotion_moments):
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    for moment_time, emotion in emotion_moments:
        start_time = max(0, moment_time - 15)  # Start 15 seconds before the emotion moment
        emotion_label = 'happy' if emotion == 0 else 'loss'
        output_name = f"{base_name}_{emotion_label}_{moment_time}.mp4"
        output_path = os.path.join(app.config['UPLOAD_FOLDER'], output_name)
        # Use ffmpeg to cut the video
        command = f"ffmpeg -ss {start_time} -i \"{video_path}\" -t 30 -c copy \"{output_path}\""
        subprocess.call(command, shell=True)  # Execute the command


# This route handles both GET and POST requests on the homepage ('/').
# For GET requests, it simply renders the video upload form.
# For POST requests, it processes an uploaded video file to identify goal and emotion moments,
# extracts highlights based on these moments, and then renders a page to display the processed videos.
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    # Check if the request method is POST, indicating that the form has been submitted.
    if request.method == 'POST':
        # Retrieve the file from the submitted form data.
        file = request.files['file']
        # Ensure a file was submitted and it has a filename.
        if file and file.filename:
            # Secure the filename to prevent directory traversal attacks.
            filename = secure_filename(file.filename)
            # Construct the full path where the file will be saved.
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            # Save the uploaded file to the specified path.
            file.save(file_path)

            # Process the video to identify goal moments.
            goal_moments = process_video(file_path)
            # Process the video to identify emotion moments.
            emotion_moments = process_video_for_emotions(file_path)

            # Extract sub-videos for the identified goal moments.
            # If no goal moments were identified, an empty list is used.
            processed_videos_goal = extract_sub_videos(file_path, goal_moments) if goal_moments else []
            # Extract sub-videos for the identified emotion moments.
            # If no emotion moments were identified, an empty list is used.
            processed_videos_emotion = extract_sub_videos_for_emotions(file_path, emotion_moments) if emotion_moments else []

            # Render the results page, passing the lists of processed videos for goals and emotions.
            return render_template('results.html', videos_goal=processed_videos_goal, videos_emotion=processed_videos_emotion)
        else:
            # If no file was submitted, return a message indicating that no significant moments were detected.
            return "No significant moments detected in the video."
    # If the request method is not POST (i.e., GET), render the upload form.
    return render_template('upload.html')


if __name__ == '__main__':
    app.run(debug=True)
