import numpy as np

qtable = np.load("qtable_sarsa.npy")
print("Shape:", qtable.shape)
print("Sample value at [0,0]:", qtable[0, 0])  # ganti sesuai kebutuhan
np.set_printoptions(suppress=True, linewidth=200)
print(qtable)

best_action = np.argmax(qtable[0, 0])
print("Aksi terbaik dari (0,0):", best_action)

