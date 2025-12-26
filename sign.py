print("1. Starting sign.py...", flush=True)
import sys
print(f"2. Python version: {sys.version}", flush=True)
import os
print("3. os imported", flush=True)
import time
print("4. time imported", flush=True)
import cv2
print(f"5. cv2 version: {cv2.__version__}", flush=True)
import numpy as np
print("6. numpy imported", flush=True)
import mediapipe as mp
print("7. mediapipe imported", flush=True)

def log_to_file(msg):
    try:
        with open("startup_log.txt", "a") as f:
            f.write(f"{time.ctime()}: {msg}\n")
    except:
        pass
    print(msg, flush=True)

log_to_file("--- Initialized ---")

log_to_file("Setting environment variables...")
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

log_to_file("Importing OpenCV...")
import cv2
log_to_file("Importing NumPy...")
import numpy as np
log_to_file("Importing MediaPipe...")
import mediapipe as mp
log_to_file("All imports successful.")

# 1. Setup MediaPipe
mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Actions that we try to detect
try:
    actions = np.load('actions.npy')
    log_to_file(f"Loaded actions from actions.npy: {actions}")
except:
    actions = np.array(['hello', 'thanks', 'iloveyou'])
    log_to_file("Actions.npy not found, using default actions.")

# Load the model
try:
    try:
        from tensorflow.keras.models import load_model
    except ImportError:
        from keras.models import load_model
    
    model = load_model('action.h5')
    model_loaded = True
    log_to_file("Model 'action.h5' loaded successfully.")
    # Verify model output shape matches actions
    if model.output_shape[-1] != len(actions):
        log_to_file(f"CRITICAL: Model output shape {model.output_shape[-1]} does not match number of actions {len(actions)}!")
        model_loaded = False
except Exception as e:
    log_to_file(f"Model 'action.h5' NOT loaded: {e}")
    model_loaded = False

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results

def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             ) 

def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])

def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        cv2.rectangle(output_frame, (0, 60+num*40), (int(prob*100), 90+num*40), colors[num % len(colors)], -1)
        cv2.putText(output_frame, actions[num], (0, 85+num*40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

# Probability vizualization colors
colors = [(245,117,16), (117,245,16), (16,117,245), (255,0,0), (0,255,0), (0,0,255)]

# 2. Open Camera and Run Loop
def get_camera():
    log_to_file("Attempting camera search...")
    # Try CAP_DSHOW first as it's most reliable for webcams on Windows
    for index in [0, 1]:
        log_to_file(f"Trying index {index} with CAP_DSHOW...")
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                log_to_file(f"SUCCESS on index {index} (DSHOW)!")
                return cap
            cap.release()
            
        log_to_file(f"Trying index {index} with Default backend...")
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                log_to_file(f"SUCCESS on index {index} (Default)!")
                return cap
            cap.release()
    return None

# 2. Open Camera with Retry Loop
cap = None
while cap is None:
    cap = get_camera()
    if cap is None:
        print("\n[!] CAMERA NOT FOUND or busy.")
        print("Please check connection and ensure no other app is using it.")
        print("Retrying in 3 seconds... (Press Ctrl+C to stop)")
        time.sleep(3)
    else:
        print("\n[+] Camera connected successfully!")

# 3. Prediction Logic Variables
sequence = []
sentence = []
predictions = []
threshold = 0.5

# Set mediapipe model 
print("Initializing MediaPipe Holistic model...")
try:
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        print("MediaPipe initialized. Press 'q' to quit.")
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            
            frame_count += 1
            if frame_count % 30 == 0:
                print(f"Processed {frame_count} frames...", flush=True)

            frame = cv2.flip(frame, 1)
            image, results = mediapipe_detection(frame, holistic)
            draw_styled_landmarks(image, results)
            
            # 2. Prediction logic
            keypoints = extract_keypoints(results)
            sequence.append(keypoints)
            sequence = sequence[-30:]
            
            if len(sequence) == 30 and model_loaded:
                res = model.predict(np.expand_dims(sequence, axis=0), verbose=0)[0]
                predictions.append(np.argmax(res))
                
                # 3. Viz logic
                if np.unique(predictions[-10:])[0] == np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                        else:
                            sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5: 
                    sentence = sentence[-5:]

                # Viz probabilities
                if len(res) > 0:
                    image = prob_viz(res, actions, image, colors)
            
            # Display info
            cv2.rectangle(image, (0,0), (640, 40), (245, 117, 16), -1)
            cv2.putText(image, ' '.join(sentence), (3,30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            if not model_loaded:
                cv2.putText(image, "MODEL NOT FOUND! (action.h5)", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow('OpenCV Feed', image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            try:
                if cv2.getWindowProperty('OpenCV Feed', cv2.WND_PROP_VISIBLE) < 1:
                    break
            except:
                pass
finally:
    print("Releasing camera...")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("Cleanup complete.")
