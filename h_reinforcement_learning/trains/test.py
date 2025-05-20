import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class GridworldEnv:
    def __init__(self, num_row, num_col, max_reward=40, max_step=40):
        self.num_row = num_row
        self.num_col = num_col
        self.max_reward = max_reward
        self.max_step = max_step
        self.n_step = 0
        self.obs_shape = (num_row, num_col)
        self.action_shape = 4  # up, right, down, left
        self._setup()
        self._init_render_vars()

    def _setup(self):
        self.agent_pos = self._random_empty()
        self.goal_pos = self._random_empty(exclude=[self.agent_pos])
        self.trap_pos = self._random_empty(exclude=[self.agent_pos, self.goal_pos])

    def _random_empty(self, exclude=None):
        if exclude is None:
            exclude = []
        while True:
            pos = np.array([np.random.randint(0, self.num_row), np.random.randint(0, self.num_col)])
            if all(not np.array_equal(pos, ex) for ex in exclude):
                return pos

    def _init_render_vars(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self._render_static()

    def _render_static(self):
        self.ax.clear()
        self.ax.set_xlim(0, self.num_col)
        self.ax.set_ylim(0, self.num_row)
        self.ax.set_xticks(np.arange(0, self.num_col + 1, 1))
        self.ax.set_yticks(np.arange(0, self.num_row + 1, 1))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.ax.add_patch(patches.Rectangle((self.trap_pos[1], self.trap_pos[0]), 1, 1, color='red'))
        self.ax.add_patch(patches.Circle((self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5), 0.3, color='yellow'))

    def reset(self):
        self.agent_pos = self._random_empty(exclude=[self.goal_pos, self.trap_pos])
        self.n_step = 0
        return self.agent_pos.copy()

    def step(self, action):
        self.n_step += 1
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.num_col - 1:
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.num_row - 1:
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1

        done = False
        reward = -1

        if np.array_equal(self.agent_pos, self.goal_pos):
            reward = self.max_reward
            done = True
        elif np.array_equal(self.agent_pos, self.trap_pos):
            reward = -self.max_reward
            done = True
        elif self.n_step >= self.max_step:
            reward = -self.max_reward
            done = True

        return self.agent_pos.copy(), reward, done, {}

    def render(self, delay=0.3):
        self._render_static()
        self.ax.add_patch(patches.Rectangle((self.agent_pos[1], self.agent_pos[0]), 1, 1, color='green'))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(delay)

    def render_all_q_values(self, q_table):
        self._render_static()
        directions = {
            0: (0.5, 0.2),  # up
            1: (0.8, 0.5),  # right
            2: (0.5, 0.8),  # down
            3: (0.2, 0.5)   # left
        }
        for r in range(self.num_row):
            for c in range(self.num_col):
                for a in range(self.action_shape):
                    dx, dy = directions[a]
                    self.ax.text(c + dx, r + dy, f"{q_table[r, c, a]:.2f}",
                                 ha='center', va='center', fontsize=7, color='blue')
        self.ax.add_patch(patches.Rectangle((self.agent_pos[1], self.agent_pos[0]), 1, 1, color='green'))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class QLearner:
    def __init__(self, state_shape, num_action, epsilon=0.1, learning_rate=0.1, discount_factor=0.95):
        self.q_table = np.zeros(state_shape + (num_action,))
        self.epsilon = epsilon
        self.lr = learning_rate
        self.gamma = discount_factor
        self.num_action = num_action

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_action)
        return np.argmax(self.q_table[tuple(state)])

    def update(self, state, next_state, reward, action):
        state = tuple(state)
        next_state = tuple(next_state)
        predict = self.q_table[state + (action,)]
        target = reward + self.gamma * np.max(self.q_table[next_state])
        self.q_table[state + (action,)] += self.lr * (target - predict)


if __name__ == "__main__":
    env = GridworldEnv(5, 5)
    agent = QLearner(env.obs_shape, env.action_shape)

    episodes = 200
    for ep in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, next_state, reward, action)
            state = next_state
            total_reward += reward

    print("Training complete.")
    env.render_all_q_values(agent.q_table)
    env.render()
