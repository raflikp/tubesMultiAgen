from environment import GridWorld
from sarsa import sarsa
from q_learning import q_learning
from utils import generate_direction_map, print_direction_map, save_q_table

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
    (4, 0),  
    (1, 1),
    (4, 1),
    (2, 2), 
    (0, 3),
    (4, 3),
    (2, 4)
]

# # Bentuk 8 X 8
# width = 8
# height = 8
# start = (0, 1)
# goal = (7, 7)

# obstacles = [
#     (3, 2),  
#     (5, 3),
#     (3, 4),
#     (1, 5), (2, 5), (6, 5),
#     (1, 6), (4, 6),(6, 6),
#     (3, 7)
# ]
env = GridWorld(width, height, start, goal, obstacles)

# Parameter SARSA
episodes = 1000
alpha = 0.1
gamma = 0.9
epsilon = 0.1

# Latih Algoritma
Q, waktu, actions = sarsa(env, episodes, alpha, gamma, epsilon)

# Tampilkan hasilS
print(f"Waktu pelatihan: {waktu:.2f} detik")
print("Jumlah aksi (↑ ↓ ← →):", actions)

# Direction map
direction_map = generate_direction_map(Q)
print("\nDirection Map:")
print_direction_map(direction_map)

# Simpan Q-table
save_q_table(Q, "qtable_sarsa.npy")