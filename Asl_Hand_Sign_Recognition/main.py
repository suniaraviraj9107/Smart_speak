import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
import pyttsx3
import time
import sys
import os
from collections import deque

# 🔴 ADDED (internet translation)
from googletrans import Translator
from gtts import gTTS

# ================= CONFIG =================
CLASSES = [
    "A","B","C","D","E","F","G",
    "H","I","J","K","L","M",
    "N","O","P","Q","R","S",
    "T","U","V","W","X","Y","Z"
]

SEQ_LEN = 16
CONF_THRESH = 0.6
STABLE_FRAMES = 8
NO_HAND_FRAMES_TO_SPEAK = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================
class ASLLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(63, 256, num_layers=2, batch_first=True)
        self.fc = nn.Linear(256, len(CLASSES))

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1])

model = ASLLSTM().to(DEVICE)
model.load_state_dict(torch.load("asl_lstm_A_Z.pth", map_location=DEVICE))
model.eval()
print("✅ Model loaded successfully (A–Z)")

# ================= SPEECH =================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# 🔴 ADDED
translator = Translator()

def speak(text):
    print("🔊 EN:", text)
    engine.say(text)
    engine.runAndWait()

def speak_hindi_online(text):
    hindi = translator.translate(text, src="en", dest="hi").text
    print("🔊 HI:", hindi)

    tts = gTTS(hindi, lang="hi")
    tts.save("hi.mp3")
    os.system("start hi.mp3")   # Windows
    time.sleep(1)

# ================= MEDIAPIPE =================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)
mp_draw = mp.solutions.drawing_utils

# ================= WEBCAM =================
cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
if not cap.isOpened():
    print("❌ Webcam not accessible")
    sys.exit(1)

print("🟢 ASL A–Z → WORD → EN + HI (ONLINE) RUNNING (ESC to exit)")

# ================= STATE =================
sequence = deque(maxlen=SEQ_LEN)
prev_letter = None
stable_count = 0
word = ""
no_hand_frames = 0
letter_locked = False

# ================= MAIN LOOP =================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        no_hand_frames = 0

        if letter_locked:
            cv2.putText(frame, "REMOVE HAND", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            hand = results.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

            landmarks = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark])
            landmarks -= landmarks[0]
            landmarks /= (np.linalg.norm(landmarks[9]) + 1e-6)

            sequence.append(landmarks)

            if len(sequence) == SEQ_LEN:
                seq = np.array(sequence).reshape(1, SEQ_LEN, -1)
                seq = torch.tensor(seq, dtype=torch.float32).to(DEVICE)

                with torch.no_grad():
                    probs = torch.softmax(model(seq), dim=1)
                    conf, pred = probs.max(1)

                letter = CLASSES[pred.item()]
                confidence = conf.item()

                if confidence > CONF_THRESH:
                    if letter == prev_letter:
                        stable_count += 1
                    else:
                        prev_letter = letter
                        stable_count = 1

                    if stable_count >= STABLE_FRAMES:
                        word += letter
                        print("✅ WORD NOW:", word)

                        letter_locked = True
                        stable_count = 0
                        prev_letter = None
                        sequence.clear()

    else:
        no_hand_frames += 1
        stable_count = 0
        prev_letter = None
        sequence.clear()
        letter_locked = False

        # 🔴 ONLY ADDITION HERE (no logic changed)
        if word and no_hand_frames >= NO_HAND_FRAMES_TO_SPEAK:
            speak(word)
            speak_hindi_online(word)
            word = ""
            no_hand_frames = 0

    cv2.putText(frame, f"WORD: {word}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    cv2.imshow("ASL A–Z to Speech", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
