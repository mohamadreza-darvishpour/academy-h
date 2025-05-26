import numpy as np 
import gymnasium as gym 
from matplotlib import pyplot as plt 
from torch import nn 
import torch



class TransitionModel(nn.Module):
    #make network
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 32):
        #bigger hidden dim, more powerful model
        super(TransitionModel , self ).__init__()
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
    def forward(self , x): 
        # input as tensor
        '''because states space is limited from -8 to +8 
        its better to limit it with activation function 
        and for the reason the amounts is in  (-8 ,8) and delta is in (-16 , 16) 
        thus we decide to multiply to 20 for (\u062f\u0633\u062a \u0628\u0627\u0644\u0627 \u062f\u0631 \u0646\u0638\u0631 \u0628\u06af\u06cc\u0631\u06cc\u0645) 
        . and 'tanh' is in (-1 , +1)
        '''
        pred = 20 * torch.nn.functional.tanh(self.estimator(x))
        return pred 


class Dataset: 
    def __init__(self, max_size=100_000):  # محدودیت روی حافظه برای جلوگیری از پر شدن RAM
        self.memory = []
        self.max_size = max_size

    def append(self , data):
        self.memory.append(data)
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def sample(self , batch_size=128):
        inds = np.random.choice(len(self.memory) , batch_size , replace=False)

        inputs = [] 
        next_states = []
        rewards = []

        for ind in inds:
            sample = self.memory[ind]
            state,action,reward,next_state = sample[0],sample[1],sample[2],sample[3]
            '''
            the input of neural network must be concatenated of state and action.
            means with given current state and action where to go next . 
            '''
            inputs.append(np.concatenate([state , action]))
            next_states.append(next_state)
            rewards.append(np.array([reward]))

        return np.array(inputs) , np.array(next_states) , np.array(rewards)


#**** gather datas from env  ******
n_episod = 500
env = gym.make('Pendulum-v1')

total_reward = [] 
dataset = Dataset()
for episod in range(n_episod):
    state , _ = env.reset()
    state = np.array(state)
    episod_rewards = []

    while True:
        action = env.action_space.sample()
        next_state  , reward , terminated , truncated , _ = env.step(action)
        dataset.append([state , action , reward , next_state])  # state before step
        state = next_state
        episod_rewards.append(reward) 
        if terminated or truncated:
            break 
    total_reward.append(np.sum(episod_rewards))
#**** end gather datas from env ******


#transition model training
transition_model = TransitionModel(input_dim=4 , output_dim=3)
loss_fn = nn.MSELoss()
optim  = torch.optim.SGD(transition_model.parameters(), lr=0.01)
loss_vals = []

for e  in range(400):
    inputs_np , next_states_np , rewards_np = dataset.sample()
    delta_states_np =  next_states_np - inputs_np[: , :3]

    inputs_torch = torch.tensor(inputs_np).float()
    delta_states_torch = torch.tensor(delta_states_np).float()

    optim.zero_grad()
    next_states_pred = transition_model(inputs_torch)
    loss = loss_fn(next_states_pred , delta_states_torch)
    loss_vals.append(loss.item())
    loss.backward()
    optim.step()
    if e % 50 == 0:
        print(f"[Transition] Epoch {e}, Loss: {loss.item():.4f}")


# reward model
class RewardModel(nn.Module):
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 32):
        super(RewardModel , self ).__init__()
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)

    def forward(self , x): 
        pred = -20 * nn.functional.sigmoid(self.estimator(x))
        return pred 


reward_model = RewardModel(input_dim=4 , output_dim=1) 
loss_fn = nn.MSELoss()
optim = torch.optim.SGD(reward_model.parameters(), lr=0.01)
loss_vals = [] 

for e  in range(400):
    inputs_np , _ , rewards_np = dataset.sample()
    inputs_torch = torch.tensor(inputs_np).float()
    rewards_torch = torch.tensor(rewards_np).float()

    optim.zero_grad()
    reward_pred = reward_model(inputs_torch)
    loss = loss_fn(reward_pred , rewards_torch)
    loss_vals.append(loss.item())
    loss.backward()
    optim.step()
    if e % 50 == 0:
        print(f"[Reward] Epoch {e}, Loss: {loss.item():.4f}")



# MPC based agent
class ModelBasedAgent:
    def __init__(self , reward_model , transition_model , num_random_shoots=20 , horizon_shoot = 3):
        self.reward_model = reward_model 
        self.transition_model = transition_model 
        self.horizon  = horizon_shoot 
        self.num_random_shoots = num_random_shoots

    def select_action_MPC(self , state ):
        action_seqs =  self.get_action_seqs(self.num_random_shoots)
        action_seqs_return = self.calc_action_seq_return(state , action_seqs)
        best_action_seq_ind = np.argmax(action_seqs_return)
        best_action_seq = action_seqs[best_action_seq_ind] 
        return np.array([best_action_seq[0]])

    def get_action_seqs(self , n_seqs):
        action_seqs = []
        for i in range(n_seqs):
            action_seq = []
            for j in range(self.horizon):
                action = self.sample_action()
                action_seq.append(action)
            action_seqs.append(action_seq)  # اصلاح این خط
        return action_seqs

    def sample_action(self ,low =-2 , high = 2 ):
        return low + (high-low)*np.random.rand()

    def calc_action_seq_return(self , state , action_seqs):
        returns = []
        for i in range(len(action_seqs)):
            rewards = self.calc_return( state , action_seqs[i])
            returns.append(rewards) 
        return returns

    def calc_return(self , state , action_seq):
        reward_preds = []
        cur_state = state.copy()
        with torch.no_grad():  # مهم: جلوگیری از استفاده گرادیان
            for action in action_seq:
                input_np = np.concatenate([cur_state , np.array([action])])
                input_torch = torch.tensor(input_np).float().unsqueeze(0)

                delta_state = self.transition_model(input_torch).detach().numpy().squeeze()
                reward_pred = self.reward_model(input_torch).detach().numpy().squeeze()

                cur_state = cur_state + delta_state 
                reward_preds.append(reward_pred)
        return np.sum(reward_preds)



# ارزیابی agent
n_episod = 500
env = gym.make('Pendulum-v1')

total_reward = [] 
MB_agent = ModelBasedAgent(reward_model , transition_model)

for episod in range(n_episod):
    print('ep: ' , episod)
    state , _ = env.reset()
    state = np.array(state)
    episod_rewards = []

    while True:
        action = MB_agent.select_action_MPC(state)
        next_state  , reward , terminated , truncated , _ = env.step(action)
        dataset.append([state , action , reward , next_state])
        state = next_state
        episod_rewards.append(reward) 
        if terminated or truncated:
            break 
    total_reward.append(np.sum(episod_rewards))


# نمایش پاداش‌ها
plt.plot(total_reward)
plt.title('reward values')
plt.xlabel('Episode')
plt.ylabel('Total reward')
plt.show()
