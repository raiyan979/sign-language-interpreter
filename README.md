# Sign Language Interpreter AI

An AI-powered application that interprets sign language gestures (phrases and alphabets) in real-time using a webcam. Built with Python, OpenCV, MediaPipe, and TensorFlow.

## 🚀 Key Features

- **Real-time Detection**: Uses MediaPipe Holistic to track hand, pose, and face landmarks.
- **Deep Learning**: Employs an LSTM (Long Short-Term Memory) neural network to recognize sequences of movements.
- **Modern GUI**: A custom-built Tkinter interface for a professional user experience.
- **Personalized Training**: Tools included to record and train the model on your own signing style.

## 🛠 Tech Stack

- **Python 3.12**
- **MediaPipe**: For hand and body landmark extraction.
- **TensorFlow / Keras**: For the LSTM prediction model.
- **OpenCV**: For camera feed and image preprocessing.
- **Tkinter**: For the graphical user interface.

## 📁 Project Structure

- `gui_app.py`: The main graphical user interface.
- `sign.py`: Core logic for real-time interpretation (console version).
- `data_collection.py`: Script to record new sign samples for training.
- `train_model.py`: Script to build and train the LSTM model.
- `action.h5`: The trained neural network model.
- `actions.npy`: The list of signs the model is trained to recognize.

## ⚙️ How to Run

1. **Setup the Environment**:
   Ensure you have the virtual environment activated and dependencies installed (TensorFlow, MediaPipe, OpenCV).
   
2. **Launch the App**:
   Run `run_gui.bat` to open the professional interface.

3. **Record Data (Optional)**:
   Run `run_collection.bat` to record samples for 'A'-'E' and common phrases.

4. **Train the Model (Optional)**:
   Run `run_training.bat` after collecting data to update the neural network.

## 📝 Training Settings
- **Samples per sign**: 10
- **Recording duration**: 30 frames (~1 second)
- **Vocabulary**: A, B, C, D, E, Hello, How, You

---
Developed with ❤️ by raiyan979
