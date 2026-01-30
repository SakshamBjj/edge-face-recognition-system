"""
Face Data Collection Module
Captures face samples for training KNN classifier.

Usage:
    python add_faces.py
    
    Enter person's name when prompted, position face centrally.
    System automatically collects 100 samples with visual feedback.
"""

import cv2
import pickle
import numpy as np
import os

# Configuration
DATA_DIR = 'data/'
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
SAMPLES_PER_PERSON = 100
FACE_SIZE = (50, 50)

def initialize_capture():
    """Initialize video capture and face detector."""
    video = cv2.VideoCapture(0)
    if not video.isOpened():
        raise IOError("Cannot access webcam")
    
    face_detector = cv2.CascadeClassifier(FACE_CASCADE_PATH)
    if face_detector.empty():
        raise IOError(f"Failed to load Haar cascade from {FACE_CASCADE_PATH}")
    
    return video, face_detector

def collect_faces(video, face_detector, person_name):
    """Collect face samples with real-time feedback."""
    faces_data = []
    frame_count = 0
    
    print(f"\n[INFO] Collecting faces for: {person_name}")
    print("[INFO] Position your face centrally. Collection starts automatically.")
    
    while len(faces_data) < SAMPLES_PER_PERSON:
        ret, frame = video.read()
        if not ret:
            print("[ERROR] Failed to capture frame")
            break
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            # Crop and preprocess face
            face_crop = frame[y:y+h, x:x+w, :]
            face_resized = cv2.resize(face_crop, FACE_SIZE)
            
            # Collect every 10th frame to ensure diversity
            if len(faces_data) < SAMPLES_PER_PERSON and frame_count % 10 == 0:
                faces_data.append(face_resized)
            
            # Visual feedback
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f"Collected: {len(faces_data)}/{SAMPLES_PER_PERSON}", 
                       (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        frame_count += 1
        cv2.imshow('Collecting Faces - Press Q to cancel', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[WARNING] Collection cancelled by user")
            break
    
    video.release()
    cv2.destroyAllWindows()
    
    if len(faces_data) < SAMPLES_PER_PERSON:
        print(f"[WARNING] Only collected {len(faces_data)}/{SAMPLES_PER_PERSON} samples")
    else:
        print(f"[SUCCESS] Collected {len(faces_data)} samples")
    
    return np.array(faces_data)

def save_data(faces_data, person_name):
    """Save face data and labels to disk."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    # Reshape: (n_samples, height, width, channels) -> (n_samples, features)
    faces_flattened = faces_data.reshape(len(faces_data), -1)
    
    # Create labels (repeat name for each sample)
    new_labels = [person_name] * len(faces_data)
    
    faces_path = os.path.join(DATA_DIR, 'faces_data.pkl')
    names_path = os.path.join(DATA_DIR, 'names.pkl')
    
    # Load existing data or create new
    if os.path.exists(faces_path) and os.path.exists(names_path):
        with open(faces_path, 'rb') as f:
            existing_faces = pickle.load(f)
        with open(names_path, 'rb') as f:
            existing_names = pickle.load(f)
        
        faces_flattened = np.vstack([existing_faces, faces_flattened])
        new_labels = existing_names + new_labels
        
        print(f"[INFO] Appended to existing data. Total samples: {len(faces_flattened)}")
    else:
        print(f"[INFO] Created new dataset. Total samples: {len(faces_flattened)}")
    
    # Save updated data
    with open(faces_path, 'wb') as f:
        pickle.dump(faces_flattened, f)
    with open(names_path, 'wb') as f:
        pickle.dump(new_labels, f)
    
    print(f"[SUCCESS] Data saved to {DATA_DIR}")

def main():
    """Main execution flow."""
    try:
        # Get person name
        person_name = input("Enter person's name: ").strip()
        if not person_name:
            print("[ERROR] Name cannot be empty")
            return
        
        # Initialize hardware
        video, face_detector = initialize_capture()
        
        # Collect face samples
        faces_data = collect_faces(video, face_detector, person_name)
        
        # Save to disk
        if len(faces_data) > 0:
            save_data(faces_data, person_name)
        else:
            print("[ERROR] No faces collected. Exiting.")
    
    except Exception as e:
        print(f"[ERROR] {str(e)}")
    
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
