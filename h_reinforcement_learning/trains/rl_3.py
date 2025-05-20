import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import sleep

class gridworldenv:
    def __init__(self, num_row, num_col, max_reward=40, max_step=40):
        self.num_row = num_row
        self.num_col = num_col
        self.max_reward = max_reward
        self.max_step = max_step
        self.n_step = 0
        self.obs_shape = (num_row, num_col)
        self.action_shape = 4  # up, down, left, right
        self._setup()
        self._init_render_vars()

    def _setup(self):
        # initialize agent, goal, and trap positions
        self.agent_pos = self._random_empty()
        self.goal_pos = self._random_empty(exclude=[self.agent_pos])
        self.trap_pos = self._random_empty(exclude=[self.agent_pos, self.goal_pos])

    def _random_empty(self, exclude=None):
        if exclude is None:
            exclude = []
        while True:
            pos = np.array([np.random.randint(0, self.num_row),
                            np.random.randint(0, self.num_col)])
            if all(not np.array_equal(pos, ex) for ex in exclude):
                return pos

    def _init_render_vars(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.num_col)
        self.ax.set_ylim(0, self.num_row)
        self.ax.set_xticks(np.arange(0, self.num_col + 1, 1))
        self.ax.set_yticks(np.arange(0, self.num_row + 1, 1))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.agent_patch = patches.Rectangle((0, 0), 1, 1, color='green')
        self.goal_patch = patches.Circle((self.goal_pos[1] + 0.5,
                                         self.goal_pos[0] + 0.5), 0.3, color='yellow')
        self.trap_patch = patches.Rectangle((self.trap_pos[1],
                                            self.trap_pos[0]), 1, 1, color='red')
        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.goal_patch)
        self.ax.add_patch(self.trap_patch)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        self.agent_pos = self._random_empty(exclude=[self.goal_pos, self.trap_pos])
        self.n_step = 0
        return self.agent_pos.copy()

    def step(self, action: int):
        # move agent
        self.n_step += 1
        if action == 0 and self.agent_pos[0] < self.num_row - 1:  # down
            self.agent_pos[0] += 1
        elif action == 1 and self.agent_pos[1] < self.num_col - 1:  # right
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] > 0:  # up
            self.agent_pos[0] -= 1
        elif action == 3 and self.agent_pos[1] > 0:  # left
            self.agent_pos[1] -= 1

        done_win = np.array_equal(self.agent_pos, self.goal_pos)
        done_lose = np.array_equal(self.agent_pos, self.trap_pos)
        done_time = self.n_step > self.max_step
        done = done_win or done_lose or done_time

        reward = -1
        if done_win:
            reward = self.max_reward
        elif done_lose or done_time:
            reward = -self.max_reward

        return self.agent_pos.copy(), reward, done, {}

    def render(self, delay=0):
        # draw agent only
        self.agent_patch.set_xy((self.agent_pos[1], self.agent_pos[0]))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def render_all_q_values(self, q_table):
        # visualize full Q-table values on grid
        self.ax.clear()
        self.ax.set_xlim(0, self.num_col)
        self.ax.set_ylim(0, self.num_row)
        self.ax.set_xticks(np.arange(0, self.num_col + 1, 1))
        self.ax.set_yticks(np.arange(0, self.num_row + 1, 1))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()

        # draw goal and trap
        self.ax.add_patch(patches.Circle((self.goal_pos[1] + 0.5,
                                         self.goal_pos[0] + 0.5), 0.3, color='yellow'))
        self.ax.add_patch(patches.Rectangle((self.trap_pos[1],
                                            self.trap_pos[0]), 1, 1, color='red'))

        # draw Q-values
        directions = {
            0: (0.5, 0.8),  # down
            1: (0.8, 0.5),  # right
            2: (0.5, 0.2),  # up
            3: (0.2, 0.5)   # left
        }
        for r in range(self.num_row):
            for c in range(self.num_col):
                for a in range(self.action_shape):
                    dx, dy = directions[a]
                    qv = q_table[r, c, a]
                    self.ax.text(c + dx, r + dy, f"{qv:.4f}",
                                  ha='center', va='center', fontsize=7, color='blue')

        # draw agent
        self.ax.add_patch(patches.Rectangle((self.agent_pos[1],
                                            self.agent_pos[0]), 1, 1, color='green'))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()


class tabular_qlearning:
    def __init__(self, state_shape, num_action, epsilon=0.1,
                 learning_rate=0.001, discount_factor=0.9):
        self.state_shape = state_shape
        self.num_action = num_action
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.q_table = np.zeros((*state_shape, num_action))
        self.n_table = np.zeros((*state_shape, num_action))

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(0, self.num_action)
        return np.argmax(self.q_table[state[0], state[1], :])

    def explore(self, state):
        a = np.argmin(self.n_table[state[0], state[1], :])
        self.n_table[state[0], state[1], a] += 1
        return a

    def update(self, state, next_state, reward, action):
        cur = self.q_table[state[0], state[1], action]
        nxt = np.max(self.q_table[next_state[0], next_state[1], :])
        self.q_table[state[0], state[1], action] = (
            cur + self.learning_rate *
            (reward + self.discount_factor * nxt - cur)
        )


if __name__ == '__main__':
    env = gridworldenv(6, 6)
    episodes = 10
    agent = tabular_qlearning(env.obs_shape, env.action_shape)
    total_rewards = []

    for ep in range(episodes):
        state = env.reset()
        rewards = []
        while True:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, next_state, reward, action)
            state = next_state
            rewards.append(reward)
            # show full Q-table after each step
            env.render_all_q_values(agent.q_table)
            if done:
                break
        total_rewards.append(sum(rewards))

    plt.ioff()
    plt.figure()
    smoothed = np.convolve(total_rewards, np.ones(min(5, len(total_rewards)))/min(5, len(total_rewards)), mode='valid')
    plt.plot(smoothed)
    plt.title("Smoothed Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.grid(True)
    plt.show()
