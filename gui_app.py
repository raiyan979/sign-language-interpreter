import tkinter as tk
from tkinter import ttk
import cv2
import numpy as np
import os
import time
import mediapipe as mp
from PIL import Image, ImageTk

# Set MediaPipe and Environment
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

class SignLanguageApp:
    def __init__(self, window):
        self.window = window
        self.window.title("Antigravity Sign Language Interpreter")
        self.window.geometry("1000x800")
        self.window.configure(bg="#1a1a1a")

        # Load Model and Actions
        self.load_resources()

        # Variables for Prediction
        self.sequence = []
        self.sentence = []
        self.predictions = []
        self.threshold = 0.5
        self.is_running = False
        self.cap = None
        self.holistic = None

        # Create Screens
        self.setup_styles()
        self.show_home_screen()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TButton", 
                        font=("Helvetica", 14, "bold"), 
                        padding=10, 
                        background="#2ecc71", 
                        foreground="white")
        style.map("TButton", background=[('active', '#27ae60')])

    def load_resources(self):
        # Actions
        try:
            self.actions = np.load('actions.npy')
        except:
            self.actions = np.array(['hello', 'how', 'you', 'A', 'B', 'C', 'D', 'E'])
        
        # Colors for viz
        self.colors = [(245,117,16), (117,245,16), (16,117,245), (255,0,0), (0,255,0), (0,0,255)]

        # Model Loading
        self.model = None
        try:
            from tensorflow.keras.models import load_model
            if os.path.exists('action.h5'):
                self.model = load_model('action.h5')
                print("Model loaded successfully.")
            else:
                print("Warning: action.h5 not found.")
        except Exception as e:
            print(f"Model load error: {e}")

    def show_home_screen(self):
        self.clear_window()
        
        # Main Container
        self.container = tk.Frame(self.window, bg="#1a1a1a")
        self.container.pack(expand=True, fill="both")

        # Title
        title_label = tk.Label(self.container, 
                              text="Sign Language Interpreter", 
                              font=("Helvetica", 32, "bold"), 
                              bg="#1a1a1a", 
                              fg="#ffffff")
        title_label.pack(pady=(150, 20))

        subtitle_label = tk.Label(self.container, 
                                 text="AI-Powered Gesture Recognition", 
                                 font=("Helvetica", 14), 
                                 bg="#1a1a1a", 
                                 fg="#aaaaaa")
        subtitle_label.pack(pady=(0, 50))

        # Start Button
        start_button = ttk.Button(self.container, text="START INTERPRETING", command=self.start_app, style="TButton")
        start_button.pack(pady=20, ipadx=40)

        # Instructions
        footer = tk.Label(self.container, 
                         text="Ensure your camera is connected and you are in a well-lit area.", 
                         font=("Helvetica", 10), 
                         bg="#1a1a1a", 
                         fg="#666666")
        footer.pack(side="bottom", pady=20)

    def start_app(self):
        self.is_running = True
        self.show_main_screen()
        
        # Initialize Camera
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        # Start the processing loop
        self.process_loop()

    def show_main_screen(self):
        self.clear_window()

        # Header
        header = tk.Frame(self.window, bg="#2d2d2d", height=60)
        header.pack(fill="x")
        
        header_title = tk.Label(header, text="Interpreter Active", font=("Helvetica", 16, "bold"), bg="#2d2d2d", fg="#2ecc71")
        header_title.pack(side="left", padx=20, pady=15)

        back_button = tk.Button(header, text="Back to Menu", command=self.stop_app, bg="#e74c3c", fg="white", font=("Helvetica", 10, "bold"))
        back_button.pack(side="right", padx=20, pady=15)

        # Video Frame
        self.video_frame = tk.Label(self.window, bg="black")
        self.video_frame.pack(pady=20, padx=20, expand=True)

        # Interpretation Panel
        self.interp_panel = tk.Frame(self.window, bg="#1a1a1a")
        self.interp_panel.pack(fill="x", side="bottom", pady=20)

        self.interp_text = tk.Label(self.interp_panel, 
                                   text="Waiting for signs...", 
                                   font=("Helvetica", 28, "bold"), 
                                   bg="#1a1a1a", 
                                   fg="#ffffff")
        self.interp_text.pack(pady=10)

        self.sub_text = tk.Label(self.interp_panel, 
                                text="Signs detected: ", 
                                font=("Helvetica", 12), 
                                bg="#1a1a1a", 
                                fg="#aaaaaa")
        self.sub_text.pack()

    def clear_window(self):
        for widget in self.window.winfo_children():
            widget.destroy()

    def stop_app(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.show_home_screen()

    def mediapipe_detection(self, image, model):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = model.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return image, results

    def extract_keypoints(self, results):
        pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
        face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
        lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
        rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
        return np.concatenate([pose, face, lh, rh])

    def process_loop(self):
        if not self.is_running:
            return

        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.is_running and self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    break

                # 1. Processing
                frame = cv2.flip(frame, 1)
                image, results = self.mediapipe_detection(frame, holistic)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

                # 2. Prediction
                keypoints = self.extract_keypoints(results)
                self.sequence.append(keypoints)
                self.sequence = self.sequence[-30:]

                if len(self.sequence) == 30 and self.model is not None:
                    res = self.model.predict(np.expand_dims(self.sequence, axis=0), verbose=0)[0]
                    self.predictions.append(np.argmax(res))

                    if np.unique(self.predictions[-10:])[0] == np.argmax(res):
                        if res[np.argmax(res)] > self.threshold:
                            action = self.actions[np.argmax(res)]
                            if len(self.sentence) > 0:
                                if action != self.sentence[-1]:
                                    self.sentence.append(action)
                            else:
                                self.sentence.append(action)

                            # Update UI text
                            self.interp_text.config(text=action.upper(), fg="#2ecc71")
                            self.sub_text.config(text="Sentence: " + " ".join(self.sentence[-5:]))

                    if len(self.sentence) > 5:
                        self.sentence = self.sentence[-5:]

                # 3. Update Graphics
                img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                imgtk = ImageTk.PhotoImage(image=img)
                self.video_frame.imgtk = imgtk
                self.video_frame.configure(image=imgtk)
                
                self.window.update()

# Main Entry
if __name__ == "__main__":
    root = tk.Tk()
    app = SignLanguageApp(root)
    root.mainloop()
