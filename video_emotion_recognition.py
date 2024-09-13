#Recognises emotion using VIDEO 
#Prints emotion on the terminal and the OpenCV 

import cv2
from fer import FER
import time

# Open the default camera once
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize the emotion detector once
detector = FER()

# Function to calculate FPS
def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return current_time, fps

# Resize frame for better performance
def resize_frame(frame, scale=0.75):
    return cv2.resize(frame, None, fx=scale, fy=scale)

# Color mapping for emotions
emotion_colors = {
    'angry': (0, 0, 255),
    'disgust': (0, 128, 0),
    'fear': (128, 0, 128),
    'happy': (0, 255, 255),
    'sad': (255, 0, 0),
    'surprise': (255, 255, 0),
    'neutral': (128, 128, 128)
}

# Initialize time for FPS calculation
prev_time = time.time()

# Process every nth frame
frame_skip = 2
frame_count = 0

# Main loop
while True:
    # Capture frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Resize the frame for better performance
    frame = resize_frame(frame, scale=0.75)

    # Process every nth frame for performance boost
    frame_count += 1
    if frame_count % frame_skip != 0:
        continue
    
    # Detect emotions on the current frame
    emotions = detector.detect_emotions(frame)
    
    if emotions:
        print(f"Emotions detected: {emotions}")  # Debug: print the detected emotions
        
        # Loop over each face detected and its emotions
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            
            # Find the emotion with the highest probability
            max_emotion = max(emotion["emotions"], key=emotion["emotions"].get)
            emotion_label = f'{max_emotion}: {emotion["emotions"][max_emotion]:.2f}'
            
            # Select bounding box color based on dominant emotion
            color = emotion_colors.get(max_emotion, (0, 255, 0))

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, emotion_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        # Only print this once every 30 frames to reduce noise in the terminal
        if frame_count % 30 == 0:
            print("No emotions detected")  # Debug: when no emotions are found

    # Calculate and display FPS
    prev_time, fps = calculate_fps(prev_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the camera feed with bounding boxes and emotions
    cv2.imshow('Emotion Detection', frame)
    
    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
