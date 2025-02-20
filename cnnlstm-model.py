import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dropout, Dense, TimeDistributed, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
import tikzplotlib
import keras_tuner as kt
from tensorflow.keras.layers import GlobalAveragePooling1D


def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)

    # Extract labels and sequences
    labels = data['Gesture Label'].values
    sequence_indices = data['Sequence Index'].values

    # Drop non-feature columns to get the feature data
    feature_data = data.drop(['Gesture Label', 'Sequence Index', 'Frame Index'], axis=1).values

    # Create sequences
    sequences = []
    for seq_index in np.unique(sequence_indices):
        seq_data = feature_data[sequence_indices == seq_index]
        sequences.append(seq_data)
    print('len',len(sequences))
    #print(sequences)

    # Calculate the number of frames per sequence
    frames_per_sequence = [len(seq) for seq in sequences]

    # Print the number of frames per sequence
    print("Anzahl der Frames pro Sequenz:", frames_per_sequence)
    print("Maximale Anzahl der Frames in einer Sequenz:", max(frames_per_sequence))

    # Pad sequences
    max_sequence_length = max(len(seq) for seq in sequences)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, padding='post', dtype='float32')
    print('padded_sequences',padded_sequences.shape)

    # Extract labels
    reshaped_labels = []
    for seq_index in np.unique(sequence_indices):
        seq_labels = labels[sequence_indices == seq_index]
        reshaped_labels.append(seq_labels[0])
    reshaped_labels = np.array(reshaped_labels)
    print('reshaped_labels',reshaped_labels)

    return padded_sequences, reshaped_labels

# Load dataset
file_path = 'filtered_labeled_sequences.csv'
X, y = load_data_from_csv(file_path)



# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
print('X_train.shape',X_train.shape)
print('X_test.shape',X_test.shape)
print('y_train.shape',y_train.shape)
print('y_test.shape',y_test.shape)
print(X_train)
print(y_train)
print('X_val.shape',X_val.shape)
print('y_val.shape',y_val.shape)
max_length = max(len(seq) for seq in X_train)  # Für Trainingsdaten
print(f"Maximale Anzahl an Frames pro Sequenz: {max_length}")




# One-hot encoding for labels
num_classes = len(np.unique(y))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
y_val = np.eye(num_classes)[y_val]
print(y_train.shape)



target_sequence_length = 16 # Set this based on your model's output



y_train = np.repeat(y_train[:, np.newaxis, :], target_sequence_length, axis=1)  # Shape: (samples, time_steps, num_classes)

y_val = np.repeat(y_val[:, np.newaxis, :], target_sequence_length, axis=1)

y_test = np.repeat(y_test[:, np.newaxis, :], target_sequence_length, axis=1)


print("Shape of X_train after reshaping:", X_train.shape)
print("Shape of y_train after reshaping:", y_train.shape)  # Should be (samples, 24, num_classes)

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)




# Define the CNN-LSTM model
model = Sequential()
model.add(Conv1D(64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(MaxPooling1D(pool_size=3))
model.add(Dropout(0.3))

model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.3))

print("Shape after LSTM:", model.output_shape)
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Train the model
history = model.fit(X_train, y_train, epochs=80, batch_size=32, validation_data=(X_val, y_val), callbacks=[reduce_lr, early_stopping])

model.save('cnn-lstm_model.h5')  # Beispiel für TensorFlow/Keras


# Plotting the results
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

plt.figure(figsize=(12, 5))
epochs = range(1, len(train_loss) + 1)

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
file_name = 'cnnlstmloss11.png'
ordner_path = 'C:/Users/saman/Downloads/bama_latex_vorlage/graphics'
full_path = os.path.join(ordner_path, file_name)
plt.tight_layout()
plt.savefig(full_path)
# Save to TikZ format for LaTeX
tikzplotlib.save(os.path.join(ordner_path, 'cnnlstmloss11.tex'))
plt.show()

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Ensure y_test is in the original format with one-hot encoding
y_test_flat = y_test[:, 0, :]  # Flatten to shape (samples, num_classes)
y_pred_prob = model.predict(X_test)[:, 0, :]  # Predict probabilities and flatten the output to (samples, num_classes)

# Output for the first 5 examples
for i in range(5):
    print(f"Beispiel {i+1}:")
    for j, prob in enumerate(y_pred_prob[i]):
        print(f"   Klasse {j}: {prob:.4f}")
    print("Gesamtwahrscheinlichkeitssumme:", np.sum(y_pred_prob[i]))  # Sollte ~1.0 sein
    print()
# Initialize plot
plt.figure(figsize=(10, 8))

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
thresholds = {}  # thresholds for each class
for i in range(num_classes):
    fpr[i], tpr[i], thresholds[i] = roc_curve(y_test_flat[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    # Add thresholds as text labels
    for idx in range(0, len(thresholds[i]), max(len(thresholds[i]) // 10, 1)):  # Alle paar Punkte
        plt.text(fpr[i][idx], tpr[i][idx], f'{thresholds[i][idx]:.2f}', fontsize=8)





# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc="lower right")

plt.tight_layout()
ordner_path = 'graphics'
full_path = os.path.join(ordner_path, file_name)

plt.savefig(full_path)
# Save to TikZ format for LaTeX
tikzplotlib.save(os.path.join(ordner_path, 'roccnnlstmfulldata-threshold.tex'))

plt.show()
