'''
model based rl : 
in last trains we needed the choosing policy for maximize the return 
but 
in model based rl we want to learn the environment dynamics not 
the policy.
and 
after learning the env we start the planning for 
choose the optimail choice.


first step :
fit a model for transition probability of model
fit a model for reward function of model 



'''

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
        pred = self.estimator(x)
        return pred 


class Dataset: 
    def __init__(self):
        self.memory = []
        
    def append(self , data):
        self.memory.append(data)
        
    def sample(self , batch_size=128):
        inds = np.random.choice(len(self.memory) , batch_size , replace=False)
        
        inputs = [] 
        next_states = []
        
        for ind in inds:
            sample = self.memory[ind]
            state,action,reward,next_state = sample[0],sample[1],sample[2],sample[3]
            '''
            the input of neural network must be concatenated of state and action.
            means with given current state and action where to go next . 
            '''
            inputs.append(np.concatenate([state , action]))
            next_states.append(next_state)
        return np.array(inputs) , np.array(next_states)
'''
so its needed to gather datas from env with default or 
random policy to learn the trasition function
transition model need to get state-action 
and output is the next state
'''



#gathering datas from env 
n_episod = 500
env = gym.make('Pendulum-v1')

total_reward = [] 
dataset = Dataset()
for episod in range(n_episod):
    state , _ =env.reset()
    state = np.array(state)
    episod_rewards = []
    episod_states = []
    episod_actions = []
    
    while True:
        action = env.action_space.sample()
        next_state  , reward , terminated , truncated , _ = env.step(action)
        episod_rewards.append(reward) 
        episod_actions.append(action) 
        episod_states.append(state) 
        state = next_state 
        dataset.append([state , action , reward , next_state])
        if terminated or truncated :
            break 
        
    total_reward.append(np.sum(episod_rewards))  

#end **   gathering datas from env 




    
    
transition_model = TransitionModel(input_dim=4 , output_dim=3)
#for input obs states is 3dim and action is 1 that makes input '4'
#output dim is 3 that is the output space 



loss_fn = nn.MSELoss()
optim  = torch.optim.SGD(transition_model.parameters())
loss_vals = []
# for e  in range(400) :
#     inputs_np , next_states_np = dataset.sample()
#     inputs_torch = torch.tensor(inputs_np).float().float()
#     next_states_torch = torch.tensor(next_state).float().float()
    
#     optim.zero_grad()
#     next_states_pred = transition_model(inputs_torch)
#     loss = loss_fn(next_states_pred , next_states_torch)  #is 2 should be torch? 
#     loss_vals.append(loss.item())
#     loss.backward()
#     optim.step()
'''
notice for predicting the next state : 
when we get current state as input usually next state not differ very from current state
and just need in output predict current one . and neural network will be lazy in these items and just predict the inputed datas 
for learning.
and 
next state may be not have very different datas . for the reason the states space is close to each other and concrete and we don't see huge differents  with any action 
so
for solving this problem we could use the differents of states not the states . (delta_states_np)
'''
for e  in range(400) :
    inputs_np , next_states_np = dataset.sample()
    delta_states_np =  next_states_np - inputs_np[: , :3]
    
    inputs_torch = torch.tensor(inputs_np).float().float()
    delta_states_torch = torch.tensor(delta_states_np).float().float()
    
    
    next_states_torch = torch.tensor(next_state).float().float()
    
    optim.zero_grad()
    next_states_pred = transition_model(inputs_torch)
    loss = loss_fn(next_states_pred , delta_states_torch)  #is 2 should be torch? 
    loss_vals.append(loss.item())
    loss.backward()
    optim.step()



plt.plot( loss_vals )
plt.title('transition model loss values')
plt.show()








