import cv2
from fer import FER
import time
from collections import Counter

# Define conversation starters
conversation_starters = {
    'happy': [
        "The person looks like he/she is happy, ask 'You seem in a good mood, tell me about it'",
        "The person looks like he/she is enjoying the conversation, continue to talk to him/her"
    ],
    'sad': [
        "The person looks like he/she is sad, ask 'Something is bothering you, what happened? Tell me.'",
        "The person looks like he/she is down in the dumps, try asking what's bothering them."
    ],
    'disgust': [
        "The person looks like he/she didn't like what you just said, try to change the conversation or come out of it.",
        "The person looks like he/she is offended by you, try to get out from there."
    ],
    'neutral': [
        "The person looks like he/she is not expressing his emotions, try to start the conversation",
        "The person looks like he/she is bored, try to spice up the conversation"
    ],
    'surprise': [
        "The person looks like he/she was taken aback by your speech, try to review your words and try again",
        "The person looks like he/she was taken by surprise, try to ask why he made that expression"
    ],
    'fear': [
        "The person looks like he/she is scared of you, talk in a calm way",
        "The person looks like he/she does not like you out of fear, show them you mean no harm"
    ],
    'angry': [
        "The person looks like he/she is not liking how you are talking, don't antagonize them further",
        "The person looks like he/she is very angry, fight back"
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
            most_common_emotion = 'neutral'
        
        # Get conversation starter for the most frequent emotion
        starters = conversation_starters.get(most_common_emotion, ["I see you have an emotion I don't recognize."])
        
        # Print the most frequent emotion and conversation starter
        print(f"Most frequent emotion in the last 7 seconds: {most_common_emotion}")
        print(f"Conversation starter: {starters[0]}")

        # Reset counters and timer
        emotion_counter.clear()
        start_time = time.time()

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
