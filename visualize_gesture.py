# visualize_gesture.py
"""
Visualize a synthetic gesture (.npz) as a 3D animated skeleton.
Usage:
    python visualize_gesture.py --file synthetic_data/swipe_left_0.npz
"""

import numpy as np
import argparse
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation

N_POINTS = 21
WINDOW = 30

# simple hand connectivity (based on MediaPipe Hands)
CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17)
]

def load_sequence(path):
    data = np.load(path)['X']  # (30,63)
    seq = data.reshape(WINDOW, N_POINTS, 3)
    return seq

def animate_hand(seq, interval=200):
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-0.4, 0.4)
    ax.set_ylim(-0.4, 0.4)
    ax.set_zlim(-0.2, 0.2)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.view_init(elev=20., azim=60)

    scat = ax.scatter([], [], [], s=25, c='r')
    lines = [ax.plot([], [], [], 'b-', lw=2)[0] for _ in CONNECTIONS]

    def init():
        scat._offsets3d = ([], [], [])
        for line in lines:
            line.set_data([], [])
            line.set_3d_properties([])
        return [scat] + lines

    def update(frame):
        pts = seq[frame]
        x, y, z = pts[:,0], pts[:,1], pts[:,2]
        scat._offsets3d = (x, y, z)
        for i, (a, b) in enumerate(CONNECTIONS):
            lines[i].set_data([x[a], x[b]], [y[a], y[b]])
            lines[i].set_3d_properties([z[a], z[b]])
        ax.set_title(f"Frame {frame+1}/{len(seq)}")
        return [scat] + lines

    ani = animation.FuncAnimation(fig, update, frames=len(seq),
                                  init_func=init, interval=interval, blit=True)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file G:\hand_gesture\synthetic_data\wave_90.npz")
    args = parser.parse_args()
    seq = load_sequence("G:\\hand_gesture\\synthetic_data\\wave_90.npz")
    animate_hand(seq)
