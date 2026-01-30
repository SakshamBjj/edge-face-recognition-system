"""
Real-Time Face Recognition Inference Pipeline
Loads trained KNN model and performs real-time identification.

Usage:
    python test.py
    
    Press 'o' to log attendance (saves to CSV)
    Press 'q' to quit
"""

import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
from sklearn.neighbors import KNeighborsClassifier

# Configuration
DATA_DIR = 'data/'
ATTENDANCE_DIR = 'attendance/'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
FACE_SIZE = (50, 50)
KNN_K = 5

def load_training_data():
    """Load face data and train KNN classifier."""
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    
    if not os.path.exists(faces_path) or not os.path.exists(names_path):
        raise FileNotFoundError(
            f"Training data not found in {DATA_DIR}. Run add_faces.py first."
        )
    
    with open(faces_path, 'rb') as f:
        faces_data = pickle.load(f)
    with open(names_path, 'rb') as f:
        names = pickle.load(f)
    
    print(f"[INFO] Loaded {len(faces_data)} face samples for {len(set(names))} individuals")
    print(f"[INFO] Feature dimensions: {faces_data.shape}")
    
    return faces_data, names

def train_knn(faces_data, names):
    """Train KNN classifier."""
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights='distance', algorithm='auto')
    knn.fit(faces_data, names)
    print(f"[INFO] Trained KNN classifier (k={KNN_K})")
    return knn

def initialize_video():
    """Initialize video capture and face detector."""
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise IOError("Cannot access webcam")
    
    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_detector.empty():
        raise IOError(f"Failed to load Haar cascade from {FACE_CASCADE_PATH}")
    
    return video, face_detector

def save_attendance(name):
    """Log attendance to CSV file."""
    if not os.path.exists(ATTENDANCE_DIR):
        os.makedirs(ATTENDANCE_DIR)
    
    timestamp = datetime.now()
    date_str = timestamp.strftime("%Y-%m-%d")
    time_str = timestamp.strftime("%H:%M:%S")
    
    csv_path = os.path.join(ATTENDANCE_DIR, f"{date_str}.csv")
    
    # Check if file exists to determine if header is needed
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['NAME', 'TIME'])
        writer.writerow([name, time_str])
    
    print(f"[ATTENDANCE] Logged: {name} at {time_str}")

def recognize_faces(video, face_detector, knn):
    """Main recognition loop with real-time inference."""
    print("\n[INFO] Starting face recognition...")
    print("[CONTROLS] Press 'o' to log attendance | Press 'q' to quit\n")
    
    frame_count = 0
    last_recognized = {}  # Track last recognition time per person
    
    while True:
        ret, frame = video.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Face detection
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Preprocess face for KNN
            face_crop = frame[y:y+h, x:x+w, :]
            face_resized = cv2.resize(face_crop, FACE_SIZE)
            face_flattened = face_resized.reshape(1, -1)
            
            # Predict identity
            predicted_name = knn.predict(face_flattened)[0]
            
            # Get confidence (inverse of distance)
            distances, indices = knn.kneighbors(face_flattened)
            avg_distance = distances[0].mean()
            confidence = max(0, 100 - avg_distance)  # Heuristic confidence score
            
            # Visualization
            color = (0, 255, 0) if confidence > 50 else (0, 165, 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
            # Display name and confidence
            label = f"{predicted_name} ({confidence:.1f}%)"
            cv2.putText(frame, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Track recognition for attendance logging
            current_time = datetime.now()
            if predicted_name not in last_recognized:
                last_recognized[predicted_name] = current_time
        
        # Display frame
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow('Face Recognition - Press Q to quit', frame)
        
        frame_count += 1
        
        # Key controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] Exiting...")
            break
        elif key == ord('o') and last_recognized:
            print("\n[ATTENDANCE] Logging detected individuals...")
            for name in last_recognized.keys():
                save_attendance(name)
            print()
    
    video.release()
    cv2.destroyAllWindows()

def main():
    """Main execution flow."""
    try:
        # Load training data and train classifier
        faces_data, names = load_training_data()
        knn = train_knn(faces_data, names)
        
        # Initialize video capture
        video, face_detector = initialize_video()
        
        # Run recognition pipeline
        recognize_faces(video, face_detector, knn)
    
    except FileNotFoundError as e:
        print(f"[ERROR] {str(e)}")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {str(e)}")
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
