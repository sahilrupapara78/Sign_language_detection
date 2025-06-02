from function import *
from time import sleep

for action in actions:
    for sequence in range(no_sequences):
        try:
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except:
            pass

# cap = cv2.VideoCapture(0)
# Set mediapipe model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # NEW LOOP
    # Loop through actions
    for action in actions:
        # Loop through sequences aka videos
        for sequence in range(no_sequences):
            # Loop through video length aka sequence length
            for frame_num in range(sequence_length):

                # Read feed
                # ret, frame = cap.read()
                frame=cv2.imread('Image/{}/{}.png'.format(action,sequence))
                # frame=cv2.imread('{}{}.png'.format(action,sequence))
                # frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

                # Make detections
                image, results = mediapipe_detection(frame, hands)
#                 print(results)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Apply wait logic
                if frame_num == 0:
                    cv2.putText(image, 'STARTING COLLECTION', (120,200),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(200)
                else:
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Show to screen
                    cv2.imshow('OpenCV Feed', image)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num))
                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

    # cap.release()
    cv2.destroyAllWindows()







#
#
# import os
# import cv2
# import mediapipe as mp
# import numpy as np
#
# # Function to perform Mediapipe detection
# def mediapipe_detection(image, hands):
#     # Convert BGR image to RGB
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
#     # Perform hands detection
#     results = hands.process(image)
#     return image, results
#
# # Define paths and variables
# DATA_PATH = 'Image'
# actions = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']  # Example actions
# no_sequences = 3  # Example number of sequences
# sequence_length = 10  # Example sequence length
#
# # Set mediapipe model
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(model_complexity=0, min_detection_confidence=0.5, min_tracking_confidence=0.5)
#
# # Loop through actions
# for action in actions:
#     # Loop through sequences
#     for sequence in range(no_sequences):
#         # Loop through sequence length
#         for frame_num in range(sequence_length):
#             # Read image
#             frame_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.png')
#             frame = cv2.imread(frame_path)
#             # Check if image is valid
#             if frame is None:
#                 print(f"Error: Unable to read image '{frame_path}'.")
#                 continue
#
#             # Make detections using Mediapipe
#             try:
#                 image, results = mediapipe_detection(frame, hands)
#             except cv2.error as e:
#                 print(f"Error: {e}")
#                 continue
#
#             # Debugging: Print results or other variables
#             print(results)
#
#             # Perform other processing and saving of keypoints
#
# # Release resources
# hands.close()
