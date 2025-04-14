import numpy as np

def generate_direction_map(Q):
    directions = {0: '↑', 1: '↓', 2: '←', 3: '→'}
    height, width, _ = Q.shape
    direction_map = []

    for i in range(height):
        row = []
        for j in range(width):
            row.append(directions[np.argmax(Q[i, j])])
        direction_map.append(row)
    return direction_map

def print_direction_map(direction_map):
    for row in direction_map:
        print("  ".join(row))

def save_q_table(Q, filename):
    np.save(filename, Q)

def load_q_table(filename):
    return np.load(filename)
