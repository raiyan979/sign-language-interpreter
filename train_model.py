import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'
from sklearn.model_selection import train_test_split
try:
    from tensorflow.keras.utils import to_categorical
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.callbacks import TensorBoard
except ImportError:
    from keras.utils import to_categorical
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from keras.callbacks import TensorBoard
import numpy as np
import os

# 1. Load Data
DATA_PATH = os.path.join('MP_Data') 
actions_all = np.array(['hello', 'thanks', 'iloveyou'])
no_sequences = 30
sequence_length = 30

sequences, labels = [], []
actions_loaded = []
label_map = {}

for action in actions_all:
    print(f"Checking data for action: {action}")
    action_sequences = []
    count = 0
    for sequence in range(no_sequences):
        window = []
        skip_sequence = False
        for frame_num in range(sequence_length):
            path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            if not os.path.exists(path):
                skip_sequence = True
                break
            res = np.load(path)
            window.append(res)
        
        if not skip_sequence:
            action_sequences.append(window)
            count += 1
    
    if count > 0:
        print(f"Total sequences loaded for {action}: {count}")
        label_map[action] = len(actions_loaded)
        actions_loaded.append(action)
        for seq in action_sequences:
            sequences.append(seq)
            labels.append(label_map[action])
    else:
        print(f"No complete data for {action}, skipping.")

actions = np.array(actions_loaded)
print(f"Actions found with data: {actions}")

if len(actions) == 0:
    print("Error: No complete data found for ANY action. Please run data_collection.py first.")
    exit()

X = np.array(sequences)
y = to_categorical(labels, num_classes=len(actions)).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

# 2. Build and Train LSTM Neural Network
log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)

model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,1662)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

# Train with 200 epochs for a quick testable model
model.fit(X_train, y_train, epochs=200, callbacks=[tb_callback])

# 3. Save Model
model.save('action.h5')
# Also save the actions names so the prediction script can use them
np.save('actions.npy', actions)
print(f"Model saved as 'action.h5' for actions: {actions}")
