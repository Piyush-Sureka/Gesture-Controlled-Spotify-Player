# capture.py
"""
capture.py - Start/Stop recording controlled by keys

Controls:
  r -> start recording
  s -> stop recording and save clip (resampled to WINDOW frames)
  q -> quit

Usage:
  python capture.py --out real_data --label swipe_left --window 30
"""

import cv2
import mediapipe as mp
import numpy as np
import os
import time
from datetime import datetime
import argparse

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def landmarks_to_vector(hand_landmarks):
    v = []
    for lm in hand_landmarks.landmark:
        v.extend([lm.x, lm.y, lm.z])
    return np.array(v, dtype=np.float32)  # shape (63,)

def resample_sequence(seq, target_len):
    """
    seq: numpy array (T, F)
    Returns: numpy array (target_len, F), using linear interpolation over time
    """
    seq = np.asarray(seq)
    T, F = seq.shape
    if T == target_len:
        return seq.astype(np.float32)
    if T < 2:
        # not enough frames: pad with the same frame
        pad = np.repeat(seq, target_len, axis=0)
        return pad.astype(np.float32)
    # original frame indices
    orig = np.linspace(0, 1, T)
    target = np.linspace(0, 1, target_len)
    resampled = np.zeros((target_len, F), dtype=np.float32)
    for f in range(F):
        resampled[:, f] = np.interp(target, orig, seq[:, f])
    return resampled

def pad_or_trim(seq, target_len):
    """
    If seq shorter, pad with last frame; if longer, trim center crop.
    (Not used by default because we resample)
    """
    seq = np.asarray(seq)
    T = seq.shape[0]
    if T == target_len:
        return seq
    if T < target_len:
        pad = np.repeat(seq[-1:,...], target_len - T, axis=0)
        return np.vstack([seq, pad])
    else:
        start = (T - target_len) // 2
        return seq[start:start+target_len]

def run_capture(output_dir='data', label='wave', window=30, cam_idx=0):
    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(cam_idx)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {cam_idx}")

    recording = False
    clip_frames = []  # list of (63,) arrays
    clip_count = 0

    print("Controls: 'r' start recording, 's' stop+save, 'q' quit")
    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Camera frame not available, exiting.")
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            vect = np.zeros(window*21*3 // window, dtype=np.float32)  # placeholder shape only; we'll overwrite
            vect = np.zeros(21*3, dtype=np.float32)
            hand_present = False

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                vect = landmarks_to_vector(hand_landmarks)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_present = True
            else:
                # keep zeros for no-hand frames
                vect = np.zeros(21*3, dtype=np.float32)

            # If currently recording, append the per-frame vector (63,)
            if recording:
                clip_frames.append(vect.copy())

            # Visual overlays
            status_text = f"{'RECORDING' if recording else 'IDLE'}  Clips saved: {clip_count}"
            cv2.putText(frame, status_text, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255) if recording else (0, 255, 0), 2)
            cv2.putText(frame, "Controls: r=start, s=stop+save, q=quit", (10, frame.shape[0]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

            # Optional: show buffer length when recording
            if recording:
                cv2.putText(frame, f'Frames: {len(clip_frames)}', (10, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

            cv2.imshow('hand-capture', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('r'):
                if not recording:
                    recording = True
                    clip_frames = []
                    start_time = datetime.now().strftime("%Y%m%d%H%M%S")
                    print(f"Started recording at {start_time}.")
                else:
                    print("Already recording. Press 's' to stop and save.")

            elif key == ord('s'):
                if recording:
                    recording = False
                    T = len(clip_frames)
                    if T == 0:
                        print("No frames recorded; nothing saved.")
                    else:
                        seq = np.stack(clip_frames, axis=0)  # (T, 63)
                        # Resample to fixed window length
                        seq_rs = resample_sequence(seq, window)  # (window, 63)
                        timestamp = int(time.time())
                        filename = os.path.join(output_dir, f"{label}_{timestamp}_{clip_count}.npz")
                        np.savez_compressed(filename, X=seq_rs.astype(np.float32))
                        clip_count += 1
                        print(f"Saved clip {filename} (original frames={T} -> saved={window})")
                else:
                    print("Not recording. Press 'r' to start recording first.")

            elif key == ord('q'):
                print("Quitting.")
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=str, default='data', help='output folder to save clips')
    p.add_argument('--label', type=str, default='none', help='label name for saved clips')
    p.add_argument('--window', type=int, default=30, help='target window length in frames')
    p.add_argument('--cam', type=int, default=0, help='camera index for cv2.VideoCapture')
    args = p.parse_args()
    run_capture(output_dir=args.out, label=args.label, window=args.window, cam_idx=args.cam)
