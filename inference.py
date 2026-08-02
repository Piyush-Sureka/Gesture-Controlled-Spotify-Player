# inference.py
import torch
import numpy as np
import cv2
import mediapipe as mp
from collections import deque
import time
import subprocess
import pyautogui

from tcn_model import TCN

WINDOW = 30
FEATURES = 63
CLASSES = ['swipe_left','swipe_right','circle','wave','none']  # example, replace with yours
MODEL_PATH = 'tcn_gesture.pth'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESH = 0.7  # confidence threshold

def normalize_seq(arr):
    seq = arr.reshape(len(arr),21,3)
    wrist = seq[:,0:1,:].copy()
    seq = seq - wrist
    scale = np.linalg.norm(seq, axis=2).max() + 1e-6
    seq = seq / scale
    return seq.reshape(len(arr), -1)

def send_android_cmd(cmd):
    # cmd is shell input command string, e.g., 'input keyevent 85' or swipe
    subprocess.run(['adb', 'shell'] + cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

def dispatch_action(pred_idx):
    label = CLASSES[pred_idx]
    print("Detected:", label)
    # Example mappings:
    if label == 'swipe_left':
        # laptop: next slide
        pyautogui.press('right')
        # android (example): go back or swipe
        # send_android_cmd('input swipe 800 500 100 500')
    elif label == 'swipe_right':
        pyautogui.press('left')
    elif label == 'circle':
        pyautogui.press('space')  # play/pause
    # add more mappings as required

def load_model():
    model = TCN(num_inputs=FEATURES, num_channels=[128,128,128], kernel_size=3, dropout=0.2, num_classes=len(CLASSES))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model

def run_inference():
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    buffer = deque(maxlen=WINDOW)
    model = load_model()
    last_dispatch = 0
    cooldown = 0.6  # seconds to avoid repeated triggers
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            vect = np.zeros(FEATURES, dtype=np.float32)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lst = []
                for lm in hand_landmarks.landmark:
                    lst.extend([lm.x, lm.y, lm.z])
                vect = np.array(lst, dtype=np.float32)
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            buffer.append(vect)
            cv2.putText(frame, f'Buffer: {len(buffer)}/{WINDOW}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
            cv2.imshow('inference', frame)
            if len(buffer) == WINDOW:
                arr = np.stack(buffer, axis=0)
                arrn = normalize_seq(arr)[None, ...]  # (1, WINDOW, FEATURES)
                x = torch.tensor(arrn, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])
                    if conf > THRESH and (time.time() - last_dispatch) > cooldown and CLASSES[pred] != 'none':
                        dispatch_action(pred)
                        last_dispatch = time.time()
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_inference()
