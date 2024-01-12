from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib as plt
from keras.callbacks import TensorBoard
from sklearn.metrics import classification_report, confusion_matrix
label_map = {label:num for num, label in enumerate(actions)}
print(label_map)
sequences, labels = [], []

# Assuming sequence_length is the desired length
sequence_length = 30

for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)

            # Skip empty arrays
            if res.size == 0:
                continue

            window.append(res)

        # Check if all arrays in the window have the same shape
        if not window:
            print(f"Empty window for action {action}, sequence {sequence}. Skipping.")
            continue

        if not all(res.shape == window[0].shape for res in window):
            print(f"Inconsistent shapes in the window list for action {action}, sequence {sequence}.")
            print([res.shape for res in window])
            raise ValueError("Inconsistent shapes in the window list.")

        # Pad or truncate the sequence to the desired length
        padded_window = np.pad(window, ((0, sequence_length - len(window)), (0, 0)), 'constant', constant_values=0)

        sequences.append(padded_window)
        labels.append(label_map[action])




X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30,63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=10, callbacks=[tb_callback])
model.summary()

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")






















'''
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')
'''