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
        action = epsilon_greedy_policy(Q, state, epsilon)
        done = False

        while not done:
            next_state, reward, done = env.step(action)
            next_action = epsilon_greedy_policy(Q, next_state, epsilon)

            Q[state] += alpha * (reward + gamma * max(Q[next_state]) - Q[state])

            action_counts[action] += 1
            state = next_state
            action = next_action

    training_time = time.time() - start_time
    return Q, training_time, action_counts