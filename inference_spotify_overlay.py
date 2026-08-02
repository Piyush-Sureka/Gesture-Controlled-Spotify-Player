# inference_spotify_overlay.py
"""
Live inference + Spotify control + overlay.
Usage:
  python inference_spotify_overlay.py --model tcn_gesture.pth --classes swipe_left swipe_right circle wave none
"""

import argparse
from builtins import Exception, print
import time
from collections import deque
import platform
import subprocess

import cv2
import mediapipe as mp
import numpy as np
import torch

# --- Optional libs; used conditionally ---
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
    SPOTIPY_AVAILABLE = True
except Exception:
    SPOTIPY_AVAILABLE = False

try:
    import pyautogui
except Exception:
    pyautogui = None

try:
    import keyboard
except Exception:
    keyboard = None

try:
    import pygetwindow as gw
except Exception:
    gw = None

# Windows-only system mute (pycaw)
IS_WINDOWS = platform.system() == 'Windows'
if IS_WINDOWS:
    try:
        from ctypes import POINTER, cast
        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
        PYCAV_AVAILABLE = True
    except Exception:
        PYCAV_AVAILABLE = False
else:
    PYCAV_AVAILABLE = False

# import your TCN model definition
from tcn_model import TCN  # adjust import path if needed

# ------------------- Settings -------------------
WINDOW = 30
FEATURES = 63
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
THRESH = 0.65
COOLDOWN = 2  # seconds between actions
FPS_SMOOTH = 5

# ------------------------------------------------
def normalize_seq(arr):
    seq = arr.reshape(len(arr), 21, 3)
    wrist = seq[:, 0:1, :].copy()
    seq = seq - wrist
    scale = np.linalg.norm(seq, axis=2).max() + 1e-6
    seq = seq / scale
    return seq.reshape(len(arr), -1)

# ------------------ Spotify API wrapper ------------------
class SpotifyController:
    def __init__(self):
        self.client = None
        self.mode = 'none'
        if SPOTIPY_AVAILABLE and all([ 
            ('SPOTIPY_CLIENT_ID' in __import__('os').environ),
            ('SPOTIPY_CLIENT_SECRET' in __import__('os').environ),
            ('SPOTIPY_REDIRECT_URI' in __import__('os').environ)
        ]):
            try:
                scope = "user-modify-playback-state user-read-playback-state user-read-currently-playing"
                self.client = spotipy.Spotify(auth_manager=SpotifyOAuth(scope=scope))
                self.mode = 'spotify_api'
                print("[SpotifyController] Using Spotify Web API control.")
            except Exception as e:
                print("[SpotifyController] Spotify OAuth failed:", e)
                self.mode = 'none'
        else:
            self.mode = 'none'

    def play_pause(self):
        if self.mode == 'spotify_api':
            try:
                playback = self.client.current_playback()
                if playback and playback.get('is_playing'):
                    self.client.pause_playback()
                else:
                    # if no active device, attempt to start playback on last known device
                    self.client.start_playback()
                return True
            except Exception as e:
                print("Spotify API play_pause error:", e)
                return False
        return False

    def next(self):
        if self.mode == 'spotify_api':
            try:
                self.client.next_track()
                return True
            except Exception as e:
                print("Spotify API next error:", e)
                return False
        return False

    def prev(self):
        if self.mode == 'spotify_api':
            try:
                self.client.previous_track()
                return True
            except Exception as e:
                print("Spotify API prev error:", e)
                return False
        return False

# ------------------ Local control fallback ------------------
def local_play_pause():
    # Try keyboard media key
    if keyboard:
        try:
            keyboard.send('play/pause media')
            return True
        except Exception:
            pass
    # fallback: bring Spotify window to front and press space
    try:
        if gw and pyautogui:
            wins = [w for w in gw.getAllTitles() if 'Spotify' in w]
            if wins:
                win = gw.getWindowsWithTitle(wins[0])[0]
                win.activate()
                time.sleep(0.08)
                pyautogui.press('space')
                return True
    except Exception:
        pass
    # final fallback: pyautogui press (may not work if Spotify unfocused)
    try:
        if pyautogui:
            pyautogui.press('playpause')
            return True
    except Exception:
        pass
    return False

def local_next():
    if keyboard:
        try:
            keyboard.send('next track')
            return True
        except Exception:
            pass
    try:
        if pyautogui:
            pyautogui.press('nexttrack')
            return True
    except Exception:
        pass
    return False

def local_prev():
    if keyboard:
        try:
            keyboard.send('previous track')
            return True
        except Exception:
            pass
    try:
        if pyautogui:
            pyautogui.press('prevtrack')
            return True
    except Exception:
        pass
    return False

def toggle_system_mute():
    # Try pycaw first
    if PYCAV_AVAILABLE:
        try:
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            volume = cast(interface, POINTER(IAudioEndpointVolume))
            current = volume.GetMute()
            volume.SetMute(0 if current else 1, None)
            return True
        except:
            pass

    # Try nircmd if available
    try:
        subprocess.call(["nircmd.exe", "mutesysvolume", "2"])
        return True
    except:
        pass

    # Last fallback: keyboard media key
    try:
        keyboard.send("volume mute")
        return True
    except:
        pass

    print("No mute method available.")
    return False


# ------------------ Model loader that adapts ------------------
def load_model_auto(model_path, classes_list):
    ckpt = torch.load(model_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'fc.weight' in ckpt:
        out_features = ckpt['fc.weight'].shape[0]
    elif isinstance(ckpt, dict) and 'model_state_dict' in ckpt and 'fc.weight' in ckpt['model_state_dict']:
        out_features = ckpt['model_state_dict']['fc.weight'].shape[0]
        ckpt = ckpt['model_state_dict']
    else:
        # fallback to provided classes length
        out_features = len(classes_list)
    print(f"[model] checkpoint expects {out_features} classes; classes_list has {len(classes_list)}")
    model = TCN(num_inputs=FEATURES, num_channels=[128,128,128], kernel_size=3, dropout=0.2, num_classes=out_features)
    model.load_state_dict(ckpt)
    model.to(DEVICE).eval()
    return model

# ------------------ Dispatch action ------------------
class ActionDispatcher:
    def __init__(self, spotify_ctrl):
        self.spotify = spotify_ctrl
        self.last_action = ("", 0.0)
        self.last_time = 0

    def dispatch(self, label):
        now = time.time()
        if now - self.last_time < COOLDOWN:
            return None  # cooldown
        performed = False
        if label == 'swipe_left':
            # Next
            if self.spotify.mode == 'spotify_api':
                performed = self.spotify.next()
            else:
                performed = local_next()
            action_txt = "NEXT"
        elif label == 'swipe_right':
            if self.spotify.mode == 'spotify_api':
                performed = self.spotify.prev()
            else:
                performed = local_prev()
            action_txt = "PREV"
        elif label == 'circle':
            if self.spotify.mode == 'spotify_api':
                performed = self.spotify.play_pause()
            else:
                performed = local_play_pause()
            action_txt = "PLAY/PAUSE"
        elif label == 'wave':
            # toggle system mute
            performed = toggle_system_mute()
            action_txt = "MUTE/UNMUTE"
        else:
            performed = False
            action_txt = "NONE"

        if performed:
            self.last_action = (action_txt, now)
            self.last_time = now
            print(f"[action] {action_txt} performed at {time.strftime('%H:%M:%S')}")
            return action_txt
        else:
            return None

# ------------------ Main inference + overlay ------------------
def run_inference(model_path, classes, cam_idx=0):
    model = load_model_auto(model_path, classes)
    spotify_ctrl = SpotifyController()
    dispatcher = ActionDispatcher(spotify_ctrl)

    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(cam_idx)
    buffer = deque(maxlen=WINDOW)
    prev_time = time.time()
    fps_hist = deque(maxlen=FPS_SMOOTH)

    with mp_hands.Hands(min_detection_confidence=0.6, min_tracking_confidence=0.6, max_num_hands=1) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)
            vect = np.zeros(FEATURES, dtype=np.float32)
            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                arr = []
                for p in lm.landmark:
                    arr.extend([p.x, p.y, p.z])
                vect = np.array(arr, dtype=np.float32)
                mp.solutions.drawing_utils.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
            buffer.append(vect)

            # inference
            current_label = "idle"
            conf = 0.0
            action_text = dispatcher.last_action[0] if dispatcher.last_action[0] else "—"
            time_since_action = time.time() - dispatcher.last_action[1] if dispatcher.last_action[1] else 999

            if len(buffer) == WINDOW:
                seq = np.stack(buffer, axis=0)
                seqn = normalize_seq(seq)[None, ...]
                x = torch.tensor(seqn, dtype=torch.float32).to(DEVICE)
                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred = int(np.argmax(probs))
                    conf = float(probs[pred])
                    if conf > THRESH:
                        cls = classes[pred] if pred < len(classes) else f"class_{pred}"
                        current_label = cls
                        acted = dispatcher.dispatch(cls)
                        if acted:
                            action_text = acted

            # overlay text
            now = time.time()
            dt = now - prev_time
            prev_time = now
            fps = 1.0 / dt if dt > 0 else 0.0
            fps_hist.append(fps)
            fps_avg = sum(fps_hist)/len(fps_hist)

            overlay = f"Gesture: {current_label}  ({conf:.2f})"
            mode = f"Backend: {spotify_ctrl.mode}"
            cv2.putText(frame, overlay, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
            cv2.putText(frame, f"Action: {action_text}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,200,255), 2)
            cv2.putText(frame, f"{mode}  FPS: {fps_avg:.1f}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1)

            cv2.imshow('gesture-spotify', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

# ------------------ CLI ------------------
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='path to tcn model .pth')
    p.add_argument('--classes', nargs='+', required=True, help='ordered class list, same as training')
    p.add_argument('--cam', type=int, default=0)
    args = p.parse_args()
    run_inference(args.model, args.classes, cam_idx=args.cam)
