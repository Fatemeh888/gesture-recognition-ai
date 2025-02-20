import os
import numpy as np
import pandas as pd
import sns
import seaborn as sns
import tikzplotlib
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.src.layers import Conv1D, MaxPooling1D
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, TimeDistributed, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Bidirectional
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.regularizers import l2
# from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve
from tensorflow.keras.layers import SimpleRNN, Dense, Dropout, TimeDistributed, Input
from cnnlstmmodel import y_test_flat
from sklearn.metrics import roc_curve, auc


# import tsaug

# Load and preprocess data
def load_data_from_csv(file_path):
    data = pd.read_csv(file_path)

    # Extract labels and sequences
    labels = data['Gesture Label'].values
    sequence_indices = data['Sequence Index'].values
    frame_indices = data['Frame Index'].values

    # Drop non-feature columns to get the feature data
    feature_data = data.drop(['Gesture Label', 'Sequence Index', 'Frame Index'], axis=1).values

    # Determine number of unique sequences and number of features
    num_sequences = len(np.unique(sequence_indices))
    num_features = feature_data.shape[1]

    # Create an empty list to store reshaped sequences
    sequences = []
    sequence_lengths = []

    for seq_index in np.unique(sequence_indices):
        seq_data = feature_data[sequence_indices == seq_index]
        sequence_lengths.append(len(seq_data))
        sequences.append(seq_data)

    # Find the maximum sequence length for padding
    max_sequence_length = max(sequence_lengths)

    # Pad sequences to ensure consistent input dimensions
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length, dtype='float32', padding='post')

    # Convert list to numpy array
    reshaped_sequences = np.array(padded_sequences)

    # Extract labels
    reshaped_labels = []
    for seq_index in np.unique(sequence_indices):
        seq_labels = labels[sequence_indices == seq_index]
        reshaped_labels.append(seq_labels[0])
    reshaped_labels = np.array(reshaped_labels)

    return reshaped_sequences, reshaped_labels


# Load your dataset
file_path = 'filtered_labeled_sequences.csv'
X, y = load_data_from_csv(file_path)


# Training- and Test Dataset (80% Training, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training- and Validation Dataset (80% Training, 20% Validation)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


print("Trainingsdaten:")
print(X_train, y_train)
print("Validierungsdaten:")
print(X_val, y_val)
print("Testdaten:")
print(X_test, y_test)

num_classes = len(np.unique(y))
y_train = np.eye(num_classes)[y_train]
y_test = np.eye(num_classes)[y_test]
y_val = np.eye(num_classes)[y_val]

# Normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)
X_val = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)

# Expand y_train and y_test to 3D
y_train = np.repeat(y_train[:, np.newaxis, :], X_train.shape[1], axis=1)
y_test = np.repeat(y_test[:, np.newaxis, :], X_test.shape[1], axis=1)
y_val = np.repeat(y_val[:, np.newaxis, :], X_val.shape[1], axis=1)
# Define the LSTM model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(150, return_sequences=True))
model.add(Dropout(0.5))
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))  # TimeDistributed layer
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001), metrics=['accuracy'])
model.summary()


early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping])

# Retrieve loss and accuracy from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot for Loss
plt.figure(figsize=(12, 5))
epochs = range(1, len(train_loss) + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# callbacks=[early_stopping]
# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

# Ensure y_test is in the original format with one-hot encoding
y_test_flat = y_test[:, 0, :]  # Flatten to shape (samples, num_classes)
y_pred_prob = model.predict(X_test)[:, 0, :]  # Predict probabilities and flatten the output to (samples, num_classes)

# Initialize plot
plt.figure(figsize=(10, 8))

# Compute ROC curve and AUC for each class
fpr = {}
tpr = {}
roc_auc = {}
for i in range(num_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_flat[:, i], y_pred_prob[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

# Plot the ROC curve
plt.plot([0, 1], [0, 1], 'k--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for Each Class')
plt.legend(loc="lower right")

plt.tight_layout()
file_name = 'roclstmfulldata.png'
ordner_path = 'graphics'
full_path = os.path.join(ordner_path, file_name)

plt.savefig(full_path)
# Save to TikZ format for LaTeX
tikzplotlib.save(os.path.join(ordner_path, 'roclstmfulldata.tex'))
plt.show()

# Make predictions
y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=-1)  # Convert probabilities to class labels

# Convert y_test from one-hot encoding to class labels
y_test_labels = np.argmax(y_test, axis=-1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels.flatten(), y_pred.flatten())

# Display the confusion matrix
display_labels = np.arange(num_classes)  # Ensure this matches the number of classes



# Define the RNN model
model = Sequential()
model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))  # Input-Schicht, gleiche Form
model.add(SimpleRNN(150, return_sequences=True))  # SimpleRNN-Schicht
model.add(Dropout(0.5))  # Dropout zur Regularisierung
model.add(TimeDistributed(Dense(num_classes, activation='softmax')))  # TimeDistributed-Schicht
model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.0001),
              metrics=['accuracy'])  # Model kompilieren
model.summary()

# Early stopping and reduce learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Model training
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_val, y_val),
                    callbacks=[reduce_lr, early_stopping])

# Retrieve loss and accuracy from the history object
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']

# Plot for Loss
plt.figure(figsize=(12, 5))
epochs = range(1, len(train_loss) + 1)
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'r', label='Validation Loss')
plt.title('Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
plt.title('Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss}')
print(f'Test accuracy: {accuracy}')

y_pred_prob = model.predict(X_test)
y_pred = np.argmax(y_pred_prob, axis=-1)  # Convert probabilities to class labels

# Convert y_test from one-hot encoding to class labels
y_test_labels = np.argmax(y_test, axis=-1)

# Compute the confusion matrix
cm = confusion_matrix(y_test_labels.flatten(), y_pred.flatten())


display_labels = np.arange(num_classes)  # Ensure this matches the number of classes

#plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=display_labels, yticklabels=display_labels)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Confusion Matrix")
plt.show()

