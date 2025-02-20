import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from collections import deque


# ---------------- Load MoveNet Model ---------------- #
def get_model(name):

    if name == 'lightning':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        pixels = 192  # Input size for lightning model
    elif name == 'thunder':
        model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        pixels = 256  # Input size for thunder model
    else:
        raise Exception("Model unknown")
    return model, pixels


# Initialize the model
name = 'thunder'
model, pixels = get_model(name)
movenet = model.signatures['serving_default']


# ---------------- Draw Keypoints on Image ---------------- #
def draw_prediction_on_image(image, keypoints_with_scores, keypoint_threshold=0.2, pixels=256):


    height, width, channel = image.shape

    # Extract x, y coordinates and confidence scores
    x_coordinates = keypoints_with_scores['output_0'][0, 0, :, 1].numpy() * width
    x_coordinates = x_coordinates.astype(int)
    y_coordinates = keypoints_with_scores['output_0'][0, 0, :, 0].numpy() * height
    y_coordinates = y_coordinates.astype(int)
    scores = keypoints_with_scores['output_0'][0, 0, :, 2].numpy()

    # Draw keypoints if confidence is above threshold
    for i in range(len(scores)):
        if scores[i] > keypoint_threshold:
            image[y_coordinates[i], x_coordinates[i], :] = [0, 0, 255]  # Red color

    # Define keypoint connections and draw them
    kpts_absolute_xy = np.stack([x_coordinates, y_coordinates], axis=-1)
    for edge_pair in [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10)]:
        if scores[edge_pair[0]] > keypoint_threshold and scores[edge_pair[1]] > keypoint_threshold:
            cv2.line(image, tuple(kpts_absolute_xy[edge_pair[0]]), tuple(kpts_absolute_xy[edge_pair[1]]), (0, 0, 255),
                     2)

    return image if height != pixels else cv2.resize(image, dsize=(600, 600)).astype('uint8')


# ---------------- Video Processing ---------------- #
video_path = 'output.avi'  # Set video file path
cap = cv2.VideoCapture(video_path)  # Load video

frame_number = 0  # Track frame index
gestures = {}  # Store labeled gestures
frame_buffer = deque(maxlen=50)  # Buffer for rewinding frames

# Display controls
print("Press 's' to start a gesture, 'e' to end a gesture, 'b' to rewind, 'q' to quit.")

current_gesture = None  # Track current gesture being labeled

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_buffer.append((frame_number, frame.copy()))

    # Process the frame with MoveNet
    image = cv2.resize(frame, (pixels, pixels))
    image = tf.convert_to_tensor(image, dtype=tf.float32)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(image, dtype=tf.int32)

    keypoints_with_scores = movenet(image)
    image = image.numpy().squeeze(axis=0).astype('uint8')
    output_overlay = draw_prediction_on_image(image, keypoints_with_scores, keypoint_threshold=0.2, pixels=pixels)

    # Display the frame
    cv2.imshow('Video', output_overlay)
    key = cv2.waitKey(0) & 0xFF  # Wait for key press

    # ---------------- Labeling Gestures ---------------- #
    if key == ord('s'):  # Start gesture
        gesture_name = input("Enter gesture name: ")
        current_gesture = gesture_name
        gestures.setdefault(current_gesture, []).append([frame_number, None])
        print(f"Started '{gesture_name}' at frame {frame_number}")

    elif key == ord('e'):  # End gesture
        if current_gesture and gestures[current_gesture]:
            gestures[current_gesture][-1][1] = frame_number
            print(f"Ended '{current_gesture}' at frame {frame_number}")
            current_gesture = None

    elif key == ord('b'):  # Rewind a few frames
        if len(frame_buffer) > 1:
            frame_number, frame = frame_buffer.pop()
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            print(f"Rewound to frame {frame_number}")

    elif key == ord('d'):  # Delete last gesture label
        if gestures:
            last_gesture = list(gestures.keys())[-1]
            if gestures[last_gesture]:
                gestures[last_gesture].pop()
                if not gestures[last_gesture]:
                    del gestures[last_gesture]
                print(f"Deleted last label for '{last_gesture}'")
            else:
                print("No gesture to delete.")

    elif key == ord('q'):  # Quit
        break

    frame_number += 1  # Increment frame counter

cap.release()
cv2.destroyAllWindows()

# ---------------- Save Annotations ---------------- #
for gesture, occurrences in gestures.items():
    for start, end in occurrences:
        print(f"Gesture '{gesture}': frames {start}-{end}")

# Save annotations to file
with open('annotations.txt', 'w') as f:
    for gesture, occurrences in gestures.items():
        for start, end in occurrences:
            f.write(f"Gesture '{gesture}': frames {start}-{end}\n")
