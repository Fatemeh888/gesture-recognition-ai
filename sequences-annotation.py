import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import csv


# ---------------- Load MoveNet Model ---------------- #
def get_model(name='thunder'):

    if name == 'lightning':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        pixels = 192  # Input size for lightning model
    elif name == 'thunder':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        pixels = 256  # Input size for thunder model
    else:
        raise Exception("Model unknown")  # Raise an error for invalid model name
    return model, pixels

# Initialize the MoveNet model with 'thunder' version
model, pixels = get_model('thunder')
movenet = model.signatures['serving_default']


# ---------------- Function to Extract Keypoints ---------------- #
def detect_keypoints(frame):

    image = cv2.resize(frame, (pixels, pixels))  # Resize frame to match model input size
    image = tf.convert_to_tensor(image, dtype=tf.float32)  # Convert image to tensor
    image = tf.expand_dims(image, axis=0)  # Add batch dimension
    image = tf.cast(image, dtype=tf.int32)  # Convert data type to int32
    keypoints_with_scores = movenet(image)  # Run the model on the image
    return keypoints_with_scores['output_0'][0, 0, :, :].numpy()  # Extract keypoints data


# ---------------- Read Gesture Annotations from File ---------------- #
gesture_sequences = {}  # Dictionary to store sequences of keypoints

# Read gesture annotations from a text file
with open('annotations.txt', 'r') as f:
    for line in f:
        parts = line.strip().split(':')  # Split each line at ':'
        gesture_name = parts[0].split("'")[1]  # Extract gesture name
        frames = parts[1].strip().split('-')  # Get frame range
        start_frame = int(frames[0].split()[1])  # Extract start frame
        end_frame = int(frames[1]) if frames[1] != 'None' else None  # Extract end frame

        # Skip gestures with missing end frame
        if end_frame is None:
            print(f"Skipping gesture '{gesture_name}' due to missing end frame.")
            continue

        keypoints_sequence = []  # Initialize list to store keypoints for this sequence

        # Open the video file and set the starting frame
        cap = cv2.VideoCapture('output.avi')
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        frame_number = start_frame
        while frame_number <= end_frame:
            ret, frame = cap.read()  # Read the frame
            if not ret:
                break  # Stop if no more frames are available

            keypoints = detect_keypoints(frame)
            keypoints_sequence.append(keypoints)

            frame_number += 1

        cap.release()  # Release video file

        # Store the sequence in the dictionary under the corresponding gesture
        if gesture_name in gesture_sequences:
            gesture_sequences[gesture_name].append(keypoints_sequence)
        else:
            gesture_sequences[gesture_name] = [keypoints_sequence]


# ---------------- Print Extracted Gesture Sequences ---------------- #
for gesture_name, sequences in gesture_sequences.items():
    print(f"Gesture: {gesture_name}")
    print(f"Number of sequences: {len(sequences)}")
    for idx, sequence in enumerate(sequences):
        print(f"Sequence {idx + 1}: {len(sequence)} frames")



# ---------------- Define Gesture Label Mapping ---------------- #
gesture_label_mapping = {
    'person detected,no movement': 0,
    'waving hand right': 1,
    'waving hand left': 2,
    'waving hand front': 3,
    'pointing up': 4,
    'pointing down': 5,
    'hand circle': 6,
    'hand clapping': 7,
}

# Assign labels to each recorded gesture sequence
labeled_sequences = []
for gesture_name, sequences in gesture_sequences.items():
    label = gesture_label_mapping.get(gesture_name, -1)  # Get label or -1 if not found
    for sequence in sequences:
        labeled_sequences.append((sequence, label))


# ---------------- Flatten Sequence and Convert to CSV Format ---------------- #
def flatten_and_convert_to_csv_rows(sequence):

    flattened_sequence = []
    for frame in sequence:
        flattened_frame = frame.flatten()  # Flatten each frame into a 1D array
        flattened_sequence.append(flattened_frame)
    return flattened_sequence


# ---------------- Save Gesture Data to CSV ---------------- #
output_csv_file = 'labeled_sequences.csv'

with open(output_csv_file, 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)

    # Write header row
    header = ['Gesture Label', 'Sequence Index', 'Frame Index'] + [f'Value{i}' for i in range(
        len(flatten_and_convert_to_csv_rows(labeled_sequences[0][0])[0]))]
    csvwriter.writerow(header)

    # Write keypoint data
    global_frame_idx = 0
    for seq_idx, (sequence, label) in enumerate(labeled_sequences):
        flattened_sequence = flatten_and_convert_to_csv_rows(sequence)
        for frame_idx, flattened_frame in enumerate(flattened_sequence):
            row = [label, seq_idx, frame_idx] + list(flattened_frame)
            csvwriter.writerow(row)
            global_frame_idx += 1

print(f'Saved gesture sequences to {output_csv_file}')


# ---------------- Load CSV for Inspection ---------------- #
with open(output_csv_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    print(f"CSV Header: {header}")
    for i, row in enumerate(csvreader):
        if i < 378:  # Adjust number of rows printed
            print(f"Row {i}: {row}")
        else:
            break


# ---------------- Filter Invalid Gesture Labels (-1) ---------------- #
input_csv_file = 'labeled_sequences.csv'
filtered_output_csv_file = 'filtered_labeled_sequences.csv'

with open(input_csv_file, 'r') as infile, open(filtered_output_csv_file, 'w', newline='') as outfile:
    csvreader = csv.reader(infile)
    csvwriter = csv.writer(outfile)

    # Read header and write it to the output file
    header = next(csvreader)
    csvwriter.writerow(header)

    # Remove invalid labels (-1) and save filtered data
    for row in csvreader:
        label = int(row[0])  # First column is the gesture label
        if label != -1:
            csvwriter.writerow(row)

print(f'Filtered data saved to {filtered_output_csv_file}')


# ---------------- Verify Filtered CSV ---------------- #
with open(filtered_output_csv_file, 'r') as csvfile:
    csvreader = csv.reader(csvfile)
    header = next(csvreader)
    print(f"CSV Header: {header}")
    for i, row in enumerate(csvreader):
        if i < 378:  # Adjust range to see more or fewer samples
            print(f"Row {i}: {row}")
        else:
            break
