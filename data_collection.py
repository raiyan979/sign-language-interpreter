import os
import time
import sys

def log_to_file(msg):
    try:
        with open("startup_log.txt", "a") as f:
            f.write(f"{time.ctime()}: {msg}\n")
    except:
        pass
    print(msg, flush=True)

log_to_file("--- Script Starting ---")
import cv2
log_to_file("Importing numpy...")
import numpy as np
log_to_file("Importing os...")
import os
log_to_file("Importing mediapipe...")
import mediapipe as mp
log_to_file("All imports successful.")

# 1. Setup MediaPipe
mp_holistic = mp.solutions.holistic 
mp_drawing = mp.solutions.drawing_utils 

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
    image.flags.writeable = False                  
    results = model.process(image)                 
    image.flags.writeable = True                    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image, results

def draw_styled_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)) 
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)) 
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

# 2. Setup Folders for Collection
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data') 

# Actions that we try to detect
actions_all = np.array(['hello', 'how', 'you', 'A', 'B', 'C', 'D', 'E'])

# Ten videos worth of data
no_sequences = 10

# Videos are going to be 30 frames in length
sequence_length = 30

for action in actions_all: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# 3. Collect Data
def get_camera():
    print("Searching for cameras (minimal mode)...")
    for index in [0, 1]:
        print(f"Trying index {index} (DSHOW)...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("SUCCESS!")
                return cap
            cap.release()
        
        print(f"Trying index {index} (Default)...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print("SUCCESS!")
                return cap
            cap.release()
    return None

cap = get_camera()

if cap is None:
    print("Error: Could not open any camera.")
    exit()

# Set mediapipe model 
print("Initializing MediaPipe...")
stop_collection = False
try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("MediaPipe initialized. Starting data collection loops...")
        # Loop through actions
        for action in actions_all:
            if stop_collection: break
            
            # Loop through sequences aka videos
            for sequence in range(no_sequences):
                if stop_collection: break
                
                # Loop through video length aka sequence length
                for frame_num in range(sequence_length):
                    # Read feed
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        continue
                        
                    # Mirror the image (Horizontal flip)
                    frame = cv2.flip(frame, 1)

                    # Make detections
                    image, results = mediapipe_detection(frame, holistic)

                    # Draw landmarks
                    draw_styled_landmarks(image, results)
                    
                    # Apply wait logic
                    if frame_num == 0: 
                        cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                        # Wait for 2 seconds before each sequence starts
                        if cv2.waitKey(2000) & 0xFF == ord('q'):
                            stop_collection = True
                            break
                    else: 
                        cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                        cv2.imshow('OpenCV Feed', image)
                    
                    # Export keypoints
                    keypoints = extract_keypoints(results)
                    npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                    np.save(npy_path, keypoints)

                    # Break gracefully
                    key = cv2.waitKey(10) & 0xFF
                    if key == ord('q') or cv2.getWindowProperty('OpenCV Feed', cv2.WND_PROP_VISIBLE) < 1:
                        print("Collection interrupted by user.")
                        stop_collection = True
                        break
finally:
    print("Releasing resources...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Done.")
