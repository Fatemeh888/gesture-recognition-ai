import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# ---------------- Load MoveNet Model ---------------- #
def get_model(name='thunder'):

    if name == 'lightning':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        pixels = 192
    elif name == 'thunder':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        pixels = 256
    else:
        raise Exception("Model unknown")  # Raise an error for invalid model name
    return model, pixels

# Initialize the model with 'thunder' version
model, pixels = get_model('thunder')
movenet = model.signatures['serving_default']

# ---------------- Keypoint Detection Function ---------------- #
def detect_keypoints(frame):

    image = cv2.resize(frame, (pixels, pixels))  # Resize frame to match model input size
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert image to tensor
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.cast(image, dtype=tf.int32)  # Convert data type to int32
    keypoints_with_scores = movenet(image)  # Run the model on the image
    return keypoints_with_scores['output_0'][0, 0, :, :].numpy()  # Extract keypoints data

# ---------------- Process Gestures from Annotations ---------------- #

# Dictionary to store sequences of keypoints for each gesture
gesture_sequences = {}

# Read gesture annotations from a text file
with open('annotations.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':')  # Split each line at ':'
        gesture_name = parts[0].split("'")[1]  # Extract gesture name
        frames = parts[1].strip().split('-')  # Get frame range
        start_frame = int(frames[0].split()[1])  # Extract start frame
        end_frame = int(frames[1]) if frames[1] != 'None' else None  # Extract end frame (handle missing values)

        # Skip gestures with missing end frame
        if end_frame is None:
            print(f"Skipping gesture '{gesture_name}' due to missing end frame.")
            continue

        keypoints_sequence = []  # Initialize list to store keypoints for this sequence

        # Open the video file
        cap = cv2.VideoCapture('output.avi')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Set video position to start frame

        frame_number = start_frame
        while frame_number <= end_frame:
            ret, frame = cap.read()  # Read the frame
            if not ret:
                break  # Stop if no more frames are available

            keypoints = detect_keypoints(frame)
            keypoints_sequence.append(keypoints)

            frame_number += 1

        cap.release()

        # Store the sequence in the dictionary under the corresponding gesture
        if gesture_name in gesture_sequences:
            gesture_sequences[gesture_name].append(keypoints_sequence)  # Append new sequence
        else:
            gesture_sequences[gesture_name] = [keypoints_sequence]  # Create a new list with the first sequence

# ---------------- Print Extracted Gesture Sequences ---------------- #
for gesture_name, sequences in gesture_sequences.items():
    print(f"Gesture: {gesture_name}")
    print(f"Number of sequences: {len(sequences)}")
    for idx, sequence in enumerate(sequences):
        print(f"Sequence {idx + 1}: {len(sequence)} frames")
        # Uncomment to print sequence details
        # print(sequence)
