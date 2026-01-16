import cv2
import mediapipe as mp
import numpy as np
import os
import time

# ================= CONFIG =================
LETTER = "0"   # Change only this
SAVE_DIR = r"C:\Users\risha\Downloads\asl_landmarks"
SEQUENCE_LENGTH = 16
SAMPLES_PER_LETTER = 50

LETTER_DIR = os.path.join(SAVE_DIR, LETTER)
os.makedirs(LETTER_DIR, exist_ok=True)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

sample_count = 0
sequence = []

print(f"Recording letter: {LETTER}")
print("Show the gesture and hold for ~1 second")

while sample_count < SAMPLES_PER_LETTER:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand = result.multi_hand_landmarks[0]

        landmarks = np.array([
            [lm.x, lm.y, lm.z] for lm in hand.landmark
        ])

        # Normalize
        wrist = landmarks[0]
        landmarks -= wrist
        scale = np.linalg.norm(landmarks[9]) + 1e-6
        landmarks /= scale

        sequence.append(landmarks)

        if len(sequence) == SEQUENCE_LENGTH:
            filename = os.path.join(
                LETTER_DIR,
                f"{LETTER}_{sample_count:03}.npy"
            )
            np.save(filename, np.array(sequence))
            print(f"Saved: {filename}")

            sample_count += 1
            sequence = []
            time.sleep(0.5)

    cv2.imshow("Recording ASL Data", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
