from environment import GridWorld
from sarsa import sarsa
from q_learning import q_learning
from utils import generate_direction_map, print_direction_map, save_q_table, load_q_table

# Konfigurasi MAP

# Bentuk 4 X 4
# width = 4
# height = 4
# start = (0, 0)
# goal = (3, 3)

# obstacles = [
#     (1, 1),  
#     (2, 2)
# ]

# Bentuk 5 X 5
width = 5
height = 5
start = (0, 0)
goal = (4, 4)

obstacles = [
    (3, 0),  
    (1, 1),
    (4, 2),
    (2, 2),
    (3, 4),
    (0, 4),
    (1, 4)
]

# Bentuk 8 X 8
# width = 8
# height = 8
# start = (0, 1)
# goal = (7, 7)

# obstacles = [
#     (2, 3),
#     (3, 5),
#     (4, 3),
#     (5, 1), (5, 2), (5, 6),
#     (6, 1), (6, 4), (6, 6),
#     (7, 3)
# ]

env = GridWorld(width, height, start, goal, obstacles)

# Parameter SARSA
episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Latih Algoritma
Q, waktu, actions = q_learning(env, episodes, alpha, gamma, epsilon)

# Tampilkan hasil
print(f"Waktu pelatihan: {waktu:.2f} detik")
print("Jumlah aksi (↑ ↓ ← →):", actions)

# Direction map
direction_map = generate_direction_map(Q, env)
print("\nDirection Map:")
print_direction_map(direction_map)

# Simpan Q-table
save_q_table(Q, "qtable_sarsa.npy")
print("Q-table saved to file: qtable_sarsa.npy")

# Muat Q-table dari file
q_table_loaded = load_q_table("qtable_sarsa.npy")

print("Ukuran tabel:", q_table_loaded.shape)
print("Q-table loaded from file:")
print(q_table_loaded)