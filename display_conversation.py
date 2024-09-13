import cv2
from fer import FER
import time
from collections import Counter
import random

# Define conversation starters
conversation_starters = {
    'happy': [
        "You seem in a good mood, tell me about it.",
        "You look like you're enjoying yourself. What's the best part of your day?",
        "I can see you're happy! What's something exciting that's happened recently?",
        "You seem really cheerful! What's something that made you smile today?"
    ],
    'sad': [
        "Something seems to be bothering you. Want to talk about it?",
        "You look a bit down. Is there anything on your mind?",
        "I can see you're feeling sad. Is there anything I can do to help?",
        "You appear to be feeling blue. Do you want to share what's troubling you?"
    ],
    'disgust': [
        "It seems like something I said didn’t sit well with you. Should we change the topic?",
        "You look a bit unsettled. Is there something you’d prefer to talk about instead?",
        "You seem to be reacting strongly to what was just said. Let’s switch gears.",
        "It looks like you're not enjoying this topic. What would you like to discuss?"
    ],
    'neutral': [
        "You seem calm. What’s something exciting that happened to you recently?",
        "No specific emotion detected. What’s your favorite song?",
        "It looks like you’re feeling neutral. What classes did you have today?",
        "You appear to be in a neutral state. What’s something you enjoy doing in your free time?"
    ],
    'surprise': [
        "It looks like you were surprised. What was it that caught you off guard?",
        "You seem taken aback. Did something unexpected happen?",
        "I can see you were surprised. What’s something new or unexpected you’ve encountered recently?",
        "You look amazed. What’s a recent surprise that you enjoyed?"
    ],
    'fear': [
        "You seem a bit scared. Is there something that’s worrying you?",
        "You look frightened. Is there something specific causing you fear?",
        "I can see you’re feeling anxious. What’s on your mind?",
        "You appear to be scared. What’s making you feel this way?"
    ],
    'angry': [
        "You seem quite angry. What’s been bothering you?",
        "It looks like you’re upset. Is there something you’d like to talk about?",
        "You appear frustrated. What’s making you feel this way?",
        "I can see you’re angry. What’s the source of your frustration?"
    ],
    'no_emotion': [
        "I don’t see a clear emotion. What’s something you’re passionate about?",
        "No specific emotion detected. What’s your favorite song?",
        "It looks like you’re feeling neutral. What’s your favorite way to relax?",
        "You seem calm. What did you find interesting about your day today?"
    ]
}

# Open a connection to the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open video capture.")
    exit()

# Initialize FER detector
detector = FER()

# Function to calculate FPS
def calculate_fps(prev_time):
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    return current_time, fps

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
# Initialize time for emotion tracking
start_time = time.time()
# Emotion counter
emotion_counter = Counter()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if not ret:
        break
    
    # Detect emotions in the frame
    emotions = detector.detect_emotions(frame)
    
    # Process detected emotions
    if emotions:
        for emotion in emotions:
            (x, y, w, h) = emotion['box']
            max_emotion = max(emotion["emotions"], key=emotion["emotions"].get)
            # Update emotion counter
            emotion_counter[max_emotion] += 1

            # Select bounding box color based on detected emotion
            color = emotion_colors.get(max_emotion, (0, 255, 0))
            
            # Draw rectangle around the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, max_emotion.capitalize(), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    else:
        max_emotion = 'no_emotion'
    
    # Calculate and display FPS
    prev_time, fps = calculate_fps(prev_time)
    cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display the camera feed
    cv2.imshow('Emotion Detection', frame)
    
    # Check if 7 seconds have passed
    current_time = time.time()
    if current_time - start_time >= 7:
        # Determine the most frequent emotion
        if emotion_counter:
            most_common_emotion, count = emotion_counter.most_common(1)[0]
        else:
            most_common_emotion = 'no_emotion'
        
        # Get random conversation starter for the most frequent emotion
        starters = conversation_starters.get(most_common_emotion, ["I see you have an emotion I don't recognize."])
        conversation_prompt = random.choice(starters)
        
        # Print the most frequent emotion and conversation starter
        print(f"Most frequent emotion in the last 7 seconds: {most_common_emotion}")
        print(f"Conversation starter: {conversation_prompt}")

        # Reset counters and timer
        emotion_counter.clear()
        start_time = time.time()

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
