import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

class QLearner:
    def __init__(self, state_dim, num_actions, epsilon=0.99, learning_rate=0.01, discount_factor=0.99):
        self.epsilon = epsilon
        self.lr = learning_rate
        self.gamma = discount_factor
        self.num_actions = num_actions
        # Weights for linear function approximation: W ∈ ℝ^(state_dim × num_actions)
        self.weights = np.zeros((state_dim, num_actions))

    def select_action(self, state):
        state = np.asarray(state).flatten()  # ensure 1D array
        q_values = np.dot(state, self.weights)  # Q(s, a) for all actions
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_actions)
        return np.argmax(q_values)


    def update(self, state, next_state, reward, action, done):
        # Q(s, a) ≈ sᵀ · W[:, a]
        q_values = np.dot(state, self.weights)
        # Q(s', a') ≈ s'ᵀ · W[:, a']
        next_q_values = np.dot(next_state, self.weights)
        # print(f'state:{state}\nweight:{self.weights}\nnext_q_val = {next_q_values}')

        # Target = r + γ · max_a' Q(s', a') if not done
        target = reward + self.gamma * np.max(next_q_values) * (not done)
        # Temporal Difference (TD) Error: δ = target - Q(s, a)
        td_error = target - q_values[action]
        # Weight update rule (SGD): W[:, a] ← W[:, a] + α · δ · s
        self.weights[:, action] += self.lr * td_error * state


# Initialize environment and agent
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
num_actions = env.action_space.n

# Create Q-learning agent with linear function approximation
agent = QLearner(state_dim, num_actions, epsilon=.995, discount_factor=0.99, learning_rate=0.004)

# Training loop
episodes = 1000
reward_history = []

for ep in range(episodes):
    state, _ = env.reset()
    state = np.asarray(state)
    total_reward = 0

    while True:
        # Select action using epsilon-greedy policy
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        next_state = np.asarray(next_state)

        # Update Q-function using TD(0) with function approximation
        agent.update(state, next_state, reward, action, done)

        state = next_state
        total_reward += reward

        if done:
            break
    agent.epsilon = max(0.05, agent.epsilon * 0.97 )
    reward_history.append(total_reward)
    if (ep + 1) % 50 == 0:
        print(f"Episode {ep+1}: Total Reward = {total_reward}")

# Plot reward history
#the smoothed is different with actual plot
smoothed = np.convolve(reward_history, np.ones(100)/100   , mode='valid')
plt.plot(smoothed)
# plt.plot(reward_history)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Q-Learning with Linear Approximation on CartPole")
plt.show()
env.close()
