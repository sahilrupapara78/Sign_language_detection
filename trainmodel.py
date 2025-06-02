import matplotlib.pyplot as plt
from function import *
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard

label_map = {label: num for num, label in enumerate(actions)}
# print(label_map)
sequences, labels = [], []
for action in actions:
    for sequence in range(no_sequences):
        window = []
        for frame_num in range(sequence_length):
            file_path = os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num))
            # print(f"Loading file from: {file_path}")
            res = np.load(file_path, allow_pickle=True)
            # res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)))
            window.append(res)
        sequences.append(window)
        labels.append(label_map[action])


X = np.array(sequences)
y = to_categorical(labels).astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(30, 63)))
model.add(LSTM(128, return_sequences=True, activation='relu'))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))
res = [.7, 0.2, 0.1]

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
model.fit(X_train, y_train, epochs=15, callbacks=[tb_callback])
model.summary()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save('model.h5')


# Training the model
history = model.fit(x_train, y_train, epochs=15, callbacks=[tb_callback])

# Plotting the accuracy
plt.plot(history.history['categorical_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train'], loc='upper left')
plt.show()

# Plotting the loss
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train'], loc='upper left')
plt.show()





























#
# from tensorboard.program import TensorBoard
# from keras.layers import LSTM, Dense
# from keras.models import Sequential
# from keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# import os
# from function import *
#
# # Define the full path to the data directory
# DATA_PATH = os.path.join(os.getcwd(), 'MP_Data')
#
# # Define label mapping
# label_map = {label: num for num, label in enumerate(actions)}
#
# # Initialize lists to store sequences and labels
# sequences, labels = [], []
#
# # Print the shape of each sequence
# for i, seq in enumerate(sequences):
#     print(f"Sequence {i+1} shape:", np.array(seq).shape)
# # Iterate over actions and sequences to load data
# for action in actions:
#     for sequence in range(no_sequences):
#         window = []
#         for frame_num in range(sequence_length):
#             # Load numpy arrays with allow_pickle=True
#             res = np.load(os.path.join(DATA_PATH, action, str(sequence), "{}.npy".format(frame_num)), allow_pickle=True)
#             window.append(res)
#         sequences.append(window)
#         labels.append(label_map[action])
#
#
# # Convert lists to numpy arrays
# X = np.array(sequences)
# y = to_categorical(labels).astype(int)
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)
#
# # Define the directory for TensorBoard logs
# log_dir = os.path.join('Logs')
# tb_callback = TensorBoard(log_dir=log_dir)
#
# # Define and compile the LSTM model
# model = Sequential()
# model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(sequence_length, X.shape[2])))
# model.add(LSTM(128, return_sequences=True, activation='relu'))
# model.add(LSTM(64, return_sequences=False, activation='relu'))
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(actions.shape[0], activation='softmax'))
#
# model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
# # Train the model
# model.fit(X_train, y_train, epochs=20 , callbacks=[tb_callback])
#
# # Print model summary
# model.summary()
#
# # Save the model architecture as JSON and weights as HDF5
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# model.save('model.h5')
