# generate_synthetic.py
"""
Synthetic dynamic hand gesture dataset generator
Creates gesture sequences (swipe_left, swipe_right, circle, wave, none)
Each clip saved as .npz (X = [30,63]) in output folder.
"""

import os
import numpy as np
from tqdm import tqdm
import random
import math

WINDOW = 30
N_POINTS = 21
FEATURES = N_POINTS * 3

# ---------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------

def make_base_hand():
    """Rough static hand layout normalized to 0–1 space"""
    hand = np.zeros((N_POINTS, 3), dtype=np.float32)
    # Spread fingers roughly
    for i in range(N_POINTS):
        hand[i, 0] = (i % 4) * 0.05  # x
        hand[i, 1] = (i // 4) * 0.05  # y
        hand[i, 2] = 0.0
    return hand

def apply_translation(hand, tx, ty):
    hand[:, 0] += tx
    hand[:, 1] += ty
    return hand

def apply_rotation(hand, angle_deg):
    theta = np.deg2rad(angle_deg)
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    xy = hand[:, :2] @ rot.T
    hand[:, :2] = xy
    return hand

def add_noise(hand, std=0.01):
    return hand + np.random.normal(0, std, hand.shape)

# ---------------------------------------------------------------------
# Gesture generators
# ---------------------------------------------------------------------

def gen_swipe(direction='left'):
    """Swipe left/right linearly across frames"""
    start_x = 0.25 if direction == 'left' else -0.25
    end_x = -0.25 if direction == 'left' else 0.25
    seq = []
    base = make_base_hand()
    for t in range(WINDOW):
        alpha = t / (WINDOW - 1)
        x = (1 - alpha) * start_x + alpha * end_x
        frame = apply_translation(base.copy(), x, 0)
        frame = add_noise(frame, 0.01)
        seq.append(frame)
    return np.stack(seq, axis=0).reshape(WINDOW, -1)

def gen_circle(clockwise=True):
    """Move in circular motion"""
    seq = []
    base = make_base_hand()
    r = 0.2
    for t in range(WINDOW):
        theta = 2 * math.pi * (t / WINDOW)
        if not clockwise:
            theta = -theta
        cx = r * math.cos(theta)
        cy = r * math.sin(theta)
        frame = apply_translation(base.copy(), cx, cy)
        frame = add_noise(frame, 0.01)
        seq.append(frame)
    return np.stack(seq, axis=0).reshape(WINDOW, -1)

def gen_wave():
    """Oscillating left-right motion like waving"""
    seq = []
    base = make_base_hand()
    amp = 0.15
    for t in range(WINDOW):
        x = amp * math.sin(2 * math.pi * t / 10)
        frame = apply_translation(base.copy(), x, 0)
        frame = add_noise(frame, 0.01)
        seq.append(frame)
    return np.stack(seq, axis=0).reshape(WINDOW, -1)

def gen_none():
    """Stationary hand with small jitter"""
    seq = []
    base = make_base_hand()
    for t in range(WINDOW):
        frame = add_noise(base.copy(), 0.02)
        seq.append(frame)
    return np.stack(seq, axis=0).reshape(WINDOW, -1)

# ---------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------

def generate_dataset(out_dir='synthetic_data', n_per_class=300):
    os.makedirs(out_dir, exist_ok=True)
    gestures = {
        'swipe_left': lambda: gen_swipe('left'),
        'swipe_right': lambda: gen_swipe('right'),
        'circle': lambda: gen_circle(random.choice([True, False])),
        'wave': gen_wave,
        'none': gen_none,
    }

    for label, func in gestures.items():
        print(f"Generating {n_per_class} samples for {label}")
        for i in tqdm(range(n_per_class)):
            seq = func()  # (30,63)
            # Add small random rotation and noise to the *entire* sequence
            seq = seq.reshape(WINDOW, N_POINTS, 3)
            angle = random.uniform(-15, 15)
            theta = np.deg2rad(angle)
            rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            xy = seq[:, :, :2].reshape(-1, 2) @ rot.T
            seq[:, :, :2] = xy.reshape(WINDOW, N_POINTS, 2)
            seq += np.random.normal(0, 0.01, seq.shape)
            np.savez_compressed(os.path.join(out_dir, f"{label}_{i}.npz"), X=seq.reshape(WINDOW, -1))
    print(f"Synthetic dataset generated in: {out_dir}")

if __name__ == '__main__':
    generate_dataset(out_dir='synthetic_data', n_per_class=300)
