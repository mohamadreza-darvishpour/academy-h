import torch 
from torch import nn 
import numpy as np 
from matplotlib import pyplot as plt 
from sklearn.preprocessing import StandardScaler 
# from tqdm import tqdm 
import gymnasium as gym 
import gym_trading_env 
import pandas as pd 
# from gym_trading_env.downloader import download  




data_dir = 'data' # to download dataset and save 
'''download the datas '''

df = pd.read_pickle("./data/btcusdt.pkl")
df['feature_pct_change'] = df["close"].pct_change()
df['feature_high'] = df['high']/df['close']
df['feature_low'] = df['low']/df['close']
df.dropna(inplace=True)
# plt.figure()
# plt.plot(df['feature_pct_change'] , label='pct_change')
# plt.plot(df['feature_high'] , label='high')
# plt.plot(df['feature_low'] , label='low')
# plt.legend()
# plt.show()


# _____________________ env 


env = gym.make("TradingEnv",
        name="BTCUSD",
        df=df,
        positions=[-1, 0, 1],
        trading_fees=0.01/100,
        borrow_interest_rate=0.0003/100,
        windows=20)

obs_dim = env.observation_space.shape[0] * env.observation_space.shape[1]
scalar = StandardScaler()
obs_sample = []
for _ in range(1500):
    state, _ = env.reset()
    flat_state = state.flatten()
    obs_sample.append(flat_state)
obs_sample = np.array(obs_sample)
scalar.fit(obs_sample)

# _____________________ actor critic 


class Actor(nn.Module):
    def __init__(self , input_dim , output_dim =1 , hidden_dim=64):
        super(Actor ,self).__init__()
        l_1 = nn.Linear(input_dim , hidden_dim)
        r1  = nn.ReLU()
        l_2 = nn.Linear(hidden_dim , hidden_dim)
        r2  = nn.ReLU()
        l_3 = nn.Linear(hidden_dim , output_dim)
        self.estimator = nn.Sequential(
            l_1 , 
            r1,
            l_2,
            r2,
            l_3 
        )
        
    def forward(self , state):
        if not isinstance(state , torch.Tensor):
            state = torch.tensor(state , dtype = torch.float32)
        return self.estimator(state) 
    

class Critic(nn.Module):
    def __init__(self , input_dim , output_dim =1 , hidden_dim=64):
        super(Critic ,self).__init__()
        l_1 = nn.Linear(input_dim , hidden_dim)
        r1  = nn.ReLU()
        l_2 = nn.Linear(hidden_dim , hidden_dim)
        r2  = nn.ReLU()
        l_3 = nn.Linear(hidden_dim , output_dim)
        self.estimator = nn.Sequential(
            l_1 , 
            r1,
            l_2,
            r2,
            l_3 
        )
        
    def forward(self , state):
        if not isinstance(state , torch.Tensor):
            state = torch.tensor(state , dtype = torch.float32)
        return self.estimator(state) 
    
    
    
    
# --------------- agent  

class PG_agent:
    def __init__(self , state_size , action_size , hidden_dim = 64 , learning_rate = .005 , discount_factor = .99):
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor
        self.state_size = state_size 
        self.action_size = action_size
        self.actor = Actor(input_dim=state_size , output_dim=action_size ,hidden_dim=hidden_dim)
        self.critic = Critic(input_dim=state_size , output_dim=1 , hidden_dim=hidden_dim)
        
        
        #optimizer 
        self.critic_loss_fn = nn.MSELoss()
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters() , lr= learning_rate)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters() , lr= learning_rate)
        
        
    def sample_action(self , state):
        logits = self.actor(state)
        logits = logits - logits.max() #numerical stability ? ????
        probs = torch.softmax(logits , dim=-1)
        dist = torch.distributions.Categorical(probs= probs)
        return dist.sample().item() , dist 
    
    def calc_reward_to_go(self , rewards):
        rewards2go = np.zeros_like(rewards ,dtype=np.float32)
        running_add = 0 
        for t in reversed(range(len(rewards))):
            running_add = rewards[t] + self.discount_factor * running_add
            rewards2go[t] = running_add
        return rewards2go
    
    
    def update(self , states , actions , rewards):
        actions = torch.tensor(actions ) 
        rewards2go = torch.tensor(self.calc_reward_to_go(rewards=rewards) , dtype=torch.float32)
        states = torch.tensor(states , dtype =torch.float32)
        
        self.critic_optimizer.zero_grad()
        values  = self.critic(states).squeeze() 
        critic_loss = self.critic_loss_fn(values , rewards2go)
        critic_loss.backward()
        #************
        torch.nn.utils.clip_grad_norm_(self.critic.parameters() , 1.0)
        self.critic_optimizer.step()
        
        self.actor_optimizer.zero_grad()
        logits = self.actor(states)
        probs = torch.softmax(logits ,dim=-1)
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)
        
        with torch.no_grad():
            baseline   = self.critic(states).squeeze()
            advantages = rewards2go - baseline 
            
        entropy = dist.entropy().mean()
        actor_loss = -(log_probs * advantages ).mean() + 0.01 * entropy #entropy
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters() ,max_norm=1.0) 
        self.actor_optimizer.step()
        
        
        
# training_loop 
n_episod = 500 
max_step = 300
lr = 0.001
df = .99
hidden_dim = 32    #64
total_reward = [] 
agent = PG_agent(obs_dim , env.action_space.n , hidden_dim=hidden_dim  , learning_rate= lr , discount_factor= df)

for i in range(n_episod):
    if i%50 ==0 :
        print('ep: ',i)
    state  , _ = env.reset() 
    episod_states , episod_actions , episod_rewards = [] , [] , [] 
    step = 0 
    while True : 
        flat_state = scalar.transform([np.array(state).flatten()])[0]
        episod_states.append(flat_state)
        
        action , dist = agent.sample_action(flat_state)
        episod_actions.append(action)
        
        next_state , reward , done , truncated , info = env.step(action)
        episod_rewards.append(reward)
        
        state = next_state 
        step+= 1 
        if done or truncated  or step>max_step : 
            break
    
    agent.update(np.array(episod_states , dtype=np.float32) , np.array(episod_actions , dtype=np.float32) , episod_rewards)
    total_reward.append(np.sum(episod_rewards))
    
    
    
#___________plot 
plt.plot(np.convolve(total_reward , np.ones(100)/100 , mode='valid'))
plt.title('smoothed total rewards.')
plt.xlabel('episode')
plt.ylabel('total reward')
plt.grid(True)
plt.show()