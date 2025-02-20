import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import os
import time

name = 'thunder'


# Load the MoveNet model from TensorFlow Hub based on the selected version
def get_model(name):
    if name == 'lightning':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        pixels = 192  # Input size for the lightning model
    elif name == 'thunder':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        pixels = 256  # Input size for the thunder model
    else:
        raise Exception("Model unknown")
    return model, pixels


# Initialize the model and get input size
model, pixels = get_model(name)
movenet = model.signatures['serving_default']

# ---------------- Keypoint Definitions ----------------#

# Mapping keypoints to body parts
KEYPOINT_DICT = {
    'nose': 0, 'left_eye': 1, 'right_eye': 2, 'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6, 'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10, 'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14, 'left_ankle': 15, 'right_ankle': 16
}

# Defines connections between keypoints for visualization
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm', (0, 2): 'c', (1, 3): 'm', (2, 4): 'c',
    (0, 5): 'm', (0, 6): 'c', (5, 7): 'm', (7, 9): 'm',
    (6, 8): 'c', (8, 10): 'c', (5, 6): 'y', (5, 11): 'm',
    (6, 12): 'c', (11, 12): 'y', (11, 13): 'm', (13, 15): 'm',
    (12, 14): 'c', (14, 16): 'c'
}


# ---------------- Draw Keypoints and Edges ----------------#
def draw_prediction_on_image(image, keypoints_with_scores, keypoint_threshold=0.2, pixels=256):


    height, width, channel = image.shape

    # Extract x and y coordinates of keypoints
    x_coordinates = keypoints_with_scores['output_0'][0, 0, :, 1].numpy() * width
    x_coordinates = x_coordinates.astype(int)
    y_coordinates = keypoints_with_scores['output_0'][0, 0, :, 0].numpy() * height
    y_coordinates = y_coordinates.astype(int)
    scores = keypoints_with_scores['output_0'][0, 0, :, 2].numpy()

    # Draw keypoints on the image if confidence score is above the threshold
    for i in range(len(scores)):
        if scores[i] > keypoint_threshold:
            if y_coordinates[i] < height and x_coordinates[i] < width:
                image[y_coordinates[i], x_coordinates[i], :] = [0, 0, 255]  # Red dot
                if x_coordinates[i] + 1 < width:
                    image[y_coordinates[i], x_coordinates[i] + 1, :] = [0, 0, 255]
                if x_coordinates[i] - 1 >= 0:
                    image[y_coordinates[i], x_coordinates[i] - 1, :] = [0, 0, 255]
                if y_coordinates[i] + 1 < height:
                    image[y_coordinates[i] + 1, x_coordinates[i], :] = [0, 0, 255]
                if y_coordinates[i] - 1 >= 0:
                    image[y_coordinates[i] - 1, x_coordinates[i], :] = [0, 0, 255]

    # Draw edges between detected keypoints
    kpts_absolute_xy = np.stack([x_coordinates, y_coordinates], axis=-1)
    for edge_pair, color in KEYPOINT_EDGE_INDS_TO_COLOR.items():
        if scores[edge_pair[0]] > keypoint_threshold and scores[edge_pair[1]] > keypoint_threshold:
            cv2.line(image, tuple(kpts_absolute_xy[edge_pair[0]]), tuple(kpts_absolute_xy[edge_pair[1]]), (0, 0, 255),
                     2)

    # Resize image if it is the original model size
    if height == pixels and width == pixels:
        return cv2.resize(image, dsize=(600, 600)).astype('uint8')
    else:
        return image


# ---------------- Video Capture and Processing ----------------#

# Open webcam preview window
cv2.namedWindow("preview camera")
vc = cv2.VideoCapture(0)  # Capture from default webcam

# Initialize video writer for saving output
save_path = 'output.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(save_path, fourcc, 20.0, (600, 600))

# Check if video writer is initialized correctly
if not out.isOpened():
    print("Error: VideoWriter could not be opened.")
else:
    print(f"VideoWriter initialized, saving to {save_path}")

# Check if webcam is available
if vc.isOpened():
    rval, frame = vc.read()
else:
    rval = False

# Set recording duration in seconds
recording_duration = 1200  # 20 minutes
start_time = time.time()

# ---------------- Main Loop ----------------#
try:
    previous_label = None
    while rval and (time.time() - start_time) < recording_duration:
        rval, picture = vc.read()
        image = cv2.resize(picture, (pixels, pixels))
        image = tf.convert_to_tensor(image, dtype=tf.float32)
        image = tf.expand_dims(image, axis=0)
        image = tf.cast(image, dtype=tf.int32)

        # Run the MoveNet model to detect keypoints
        keypoints_with_scores = movenet(image)

        # Convert image back to NumPy and overlay detected keypoints
        image = image.numpy().squeeze(axis=0).astype('uint8')
        output_overlay = draw_prediction_on_image(image, keypoints_with_scores, keypoint_threshold=0.2, pixels=pixels)

        # Resize frame for saving
        output_overlay = cv2.resize(output_overlay, (600, 600))

        # Display the output with keypoints
        cv2.imshow("preview camera", output_overlay)
        out.write(output_overlay)  # Save the frame to the video file

        # Count keypoints with confidence > 0.5
        count = sum(keypoints_with_scores['output_0'][0, 0, :, 2].numpy() > 0.5)

        # Simple label based on detected keypoints
        label = "Person Detected" if count > 3 else "No Person / Gesture recognized"

        # Print label only if it changes
        if label != previous_label:
            print(label)
            previous_label = label

        # Exit if ESC key is pressed
        key = cv2.waitKey(20)
        if key == 27:
            break

finally:
    # Release resources
    vc.release()
    out.release()
    cv2.destroyWindow("preview camera")
