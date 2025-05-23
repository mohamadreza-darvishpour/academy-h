import gymnasium as gym
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

class Policy(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=16):
        super(Policy, self).__init__()
        self.estimator = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, state):
        return self.estimator(state)

class PG_agent:
    def __init__(self, state_size, action_size, learning_rate=0.01, discount_factor=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor
        self.policy_model = Policy(input_dim=state_size, output_dim=action_size, hidden_dim=16)
        self.optimizer = torch.optim.Adam(self.policy_model.parameters(), lr=learning_rate)

    def sample_action(self, state):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # add batch dim
        logits = self.policy_model(state)
        probs = torch.softmax(logits, dim=-1)
        action = torch.multinomial(probs, num_samples=1)
        return int(action.item())

    def calc_reward_to_go(self, rewards):
        rewardsToGo = np.zeros_like(rewards, dtype=np.float32)
        running_sum = 0
        for t in reversed(range(len(rewards))):
            running_sum = rewards[t] + self.discount_factor * running_sum
            rewardsToGo[t] = running_sum
        return rewardsToGo

    def update(self, states, actions, rewards):
        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewardToGo = self.calc_reward_to_go(rewards)
        rewardToGo = torch.tensor(rewardToGo, dtype=torch.float)

        # Normalize rewards to reduce variance
        rewardToGo = (rewardToGo - rewardToGo.mean()) / (rewardToGo.std() + 1e-9)

        self.optimizer.zero_grad()
        logits = self.policy_model(states)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs[range(len(actions)), actions]

        loss = -(selected_log_probs * rewardToGo).mean()
        loss.backward()
        self.optimizer.step()

# Training loop
env = gym.make('CartPole-v1')
agent = PG_agent(env.observation_space.shape[0], env.action_space.n, learning_rate=0.01, discount_factor=0.95)

n_episodes = 1500
total_rewards = []

for episode in range(n_episodes):
    state = env.reset()[0]
    episode_rewards = []
    episode_states = []
    episode_actions = []

    done = False
    while not done:
        action = agent.sample_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)

        episode_rewards.append(reward)
        episode_states.append(state)
        episode_actions.append(action)

        state = next_state
        done = terminated or truncated

    agent.update(episode_states, episode_actions, episode_rewards)
    total_rewards.append(sum(episode_rewards))

    if (episode + 1) % 100 == 0:
        avg_reward = np.mean(total_rewards[-100:])
        print(f"Episode {episode+1}, Average reward (last 100): {avg_reward:.2f}")

# Plot moving average reward
plt.plot(np.convolve(total_rewards, np.ones(100) / 100, mode='valid'))
plt.xlabel('Episode')
plt.ylabel('Average Reward (100 episodes)')
plt.title('Policy Gradient on CartPole')
plt.show()
