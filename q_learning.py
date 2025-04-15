import numpy as np
import random
import time

def epsilon_greedy_policy(Q, state, epsilon):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, 3)
    else:
        return np.argmax(Q[state[0], state[1]])

def q_learning(env, episodes, alpha, gamma, epsilon):
    Q = np.zeros((env.height, env.width, 4))  # 4 actions
    action_counts = [0, 0, 0, 0]
    start_time = time.time()

    for episode in range(episodes):
        state = env.reset()
        done = False

        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done = env.step(action)

            best_next_action = np.argmax(Q[next_state[0], next_state[1]])
            td_target = reward + gamma * Q[next_state[0], next_state[1], best_next_action]
            td_delta = td_target - Q[state[0], state[1], action]

            Q[state[0], state[1], action] += alpha * td_delta

            action_counts[action] += 1
            state = next_state

    training_time = time.time() - start_time
    return Q, training_time, action_counts