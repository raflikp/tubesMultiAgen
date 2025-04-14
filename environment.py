class GridWorld:
    def __init__(self, width, height, start, goal, obstacles):
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.obstacles = obstacles
        self.state = start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 0:  # Up
            x = max(x - 1, 0)
        elif action == 1:  # Down
            x = min(x + 1, self.height - 1)
        elif action == 2:  # Left
            y = max(y - 1, 0)
        elif action == 3:  # Right
            y = min(y + 1, self.width - 1)

        next_state = (x, y)

        if next_state in self.obstacles:
            reward = -10
            done = True
        elif next_state == self.goal:
            reward = 10
            done = True
        else:
            reward = -1
            done = False

        self.state = next_state
        return next_state, reward, done
