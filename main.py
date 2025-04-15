from environment import GridWorld
from sarsa import sarsa
from q_learning import q_learning
from utils import generate_direction_map, print_direction_map, save_q_table

# Konfigurasi MAP
width = 4
height = 4
start = (0, 0)
goal = (3, 3)
obstacles = [(1, 1), (2, 1)]

env = GridWorld(width, height, start, goal, obstacles)

# Parameter SARSA
episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Latih Algoritma
Q, waktu, actions = sarsa(env, episodes, alpha, gamma, epsilon)

# Tampilkan hasil
print(f"Waktu pelatihan: {waktu:.2f} detik")
print("Jumlah aksi (↑ ↓ ← →):", actions)

# Direction map
direction_map = generate_direction_map(Q)
print("\nDirection Map:")
print_direction_map(direction_map)

# Simpan Q-table
save_q_table(Q, "qtable_sarsa.npy")
