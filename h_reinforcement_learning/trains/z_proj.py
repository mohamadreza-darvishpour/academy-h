import torch
from torch import nn
import numpy as np
from matplotlib import pyplot as plt

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=32):
        super(Actor, self).__init__()

        '''
        TODO: Initialize layers for the actor network.
        Use nn.Linear for linear layers and nn.ReLU for activation.
        '''
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)


    def forward(self, state):

        '''
        TODO: Convert the input state to a torch tensor if it is not already,
        and pass it through the sequential model to get the prediction.
        '''
        pred = self.estimator(state)
        # print('\n342332432 stkate : ' , state)
        return pred 



class Critic(nn.Module):
    def __init__(self, input_dim, output_dim=1, hidden_dim=32):
        super(Critic, self).__init__()

        '''
        TODO: Initialize layers for the critic network.
        Use nn.Linear for linear layers and nn.ReLU for activation.
        '''
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)

    def forward(self, state):
        '''
        TODO: Convert the input state to a torch tensor if it is not already,
        and pass it through the model to get the value prediction.
        '''
        v_pred = self.estimator(state)
        return v_pred
    
    
    
    
    
class PG_Agent:
    def __init__(self, state_size, action_size, learning_rate=0.001, discount_factor=0.98):
        self.state_size = state_size
        self.action_size = action_size
        self.discount_factor = discount_factor

        '''
        TODO: Initialize the actor and critic networks with state_size and action_size.
        '''


        '''
        TODO: Initialize the loss function and optimizers for the actor and critic networks using torch.optim.Adam.
        '''


    def sample_action(self, state):

        '''
        TODO:
        Get action probabilities from the actor network, apply softmax,
        and sample an action based on the probabilities.
        '''

        return action

    def calc_reward_to_go(self, rewards):
        '''
        TODO: Calculate discounted future rewards.
        Initialize a running sum and iterate through rewards in reverse to compute rewards-to-go.
        '''


        return rewards2go


    def update(self, states, actions, rewards, max_batch=512):
        # Process batch if needed
        if len(states) > max_batch:
            st_ind = np.random.choice(len(states) - max_batch, 1).item()
            end_ind = st_ind + max_batch
            actions = actions[st_ind: end_ind]
            rewards = rewards[st_ind: end_ind]
            states = states[st_ind: end_ind]

        actions = torch.tensor(actions)
        rewards2go = self.calc_reward_to_go(rewards)
        rewards2go = torch.tensor(rewards2go)

        # Update Critic network
        '''
        TODO: Zero the gradients, perform a forward pass to get values, compute the loss,
        and perform backward pass and optimization step for the critic network.
        '''


        # Update Actor network
        '''
        TODO: Zero the gradients, perform a forward pass to get logits,
        compute the advantages, log probabilities, actor loss, and perform backward pass and optimization step for the actor network.
        '''





from gym_trading_env.downloader import download
import datetime
import os

data_dir = 'data'
os.makedirs(data_dir, exist_ok=True)

# Download and prepare your data
# Fetch historical BTC/USDT data from the Bitfinex exchange
# Timeframe is set to 1 hour and data is saved in the 'data' directory
# Data collection starts from January 1, 2021

download(exchange_names=["bitfinex2"], symbols=["BTC/USDT"], timeframe="1h", dir="data",
         since=datetime.datetime(year=2021, month=1, day=1))




import gymnasium as gym
import pandas as pd
from tqdm import tqdm
import os

# Load dataset
df = pd.read_pickle("./data/bitfinex2-BTCUSDT-1h.pkl")

# Create features for the trading environment
df["feature_pct_change"] = df["close"].pct_change()
df["feature_high"] = df["high"] / df["close"] - 1
df["feature_low"] = df["low"] / df["close"] - 1
df.dropna(inplace=True)


# Initialize the trading environment
env = gym.make("TradingEnv",
        name= "BTCUSD",
        df = df, # Your dataset with your custom features
        positions = [ -1, 0, 1], # -1 (=SHORT), 0(=OUT), +1 (=LONG)
        trading_fees = 0.01/100, # 0.01% per stock buy / sell (Binance fees)
        borrow_interest_rate = 0.0003/100, # 0.0003% per timestep (one timestep = 1h here)
        windows = 20) # Past N observations

obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
# Initialize the policy gradient agent
agent = PG_Agent(obs_dim, env.action_space.n)

n_episodes = 500
total_reward = []

for i in tqdm(range(n_episodes)):
    state, info = env.reset()

    episode_states = []
    episode_rewards = []
    episode_actions = []

    done, truncated = False, False
    max_steps = 10000 # Maximum number of steps per episode
    step = 0
    while not done and not truncated and step < max_steps:

        '''
        TODO: Flatten the state for the agent, sample an action from the policy,
        interact with the environment using the sampled action,
        and store the state, action, and reward.
        '''


        state = next_state
        step += 1

    # Update the agent after the episode
    agent.update(np.array(episode_states), np.array(episode_actions), episode_rewards)
    total_reward.append(np.sum(episode_rewards))

# Plot the smoothed total rewards over episodes
plt.plot(np.convolve(total_reward, np.ones(20) / 20)[10: -10])
plt.title('Smoothed Rewards')
plt.show()








