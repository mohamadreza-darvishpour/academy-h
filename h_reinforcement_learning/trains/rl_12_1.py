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



we sayed model based methods are sample efficient but not compoutationally efficient 
and 
for any state and stet to do thousends of time of calling their neural network needs.
but in model free for each step need 'one' time call its neural network.
but may be both method model free and model base are same computaional but
model free when train
model base when planning

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
        '''because states space is limited from -8 to +8 
        its better to limit it with activation function 
        and for the reason the amounts is in  (-8 ,8) and delta is in (-16 , 16) 
        thus we decide to multiply to 20 for (دست بالا در نظر بگیریم) 
        . and 'tanh' is in (-1 , +1)
        
        '''
        pred = 20 * torch.nn.functional.tanh(self.estimator(x))
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
            
            #because reward are sacaler and in future we need to change it to tensor 
            rewards.append(np.array([reward]))
        return np.array(inputs) , np.array(next_states) , np.array(rewards)
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
    inputs_np , next_states_np , rewards_np = dataset.sample()
    delta_states_np =  next_states_np - inputs_np[: , :3]
    
    inputs_torch = torch.tensor(inputs_np).float().float()
    delta_states_torch = torch.tensor(delta_states_np).float()
    
    
    # next_states_torch = torch.tensor(next_state).float()
    
    optim.zero_grad()
    next_states_pred = transition_model(inputs_torch)
    loss = loss_fn(next_states_pred , delta_states_torch)  #is 2 should be torch? 
    loss_vals.append(loss.item())
    loss.backward()
    optim.step()






'''
need reward model too
'''
class RewardModel(nn.Module):
    #make network
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 32):
        #bigger hidden dim, more powerful model
        super(RewardModel , self ).__init__()
        
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
        
    def forward(self , x): 
        # input as tensor
        #reward amount is about  in (-16 , 0)  we use sigmoid as activation function
        pred = -20 * nn.functional.sigmoid(self.estimator(x))
        return pred 


#reward model 
reward_model = RewardModel(input_dim=4 , output_dim=1) 
loss_fn = nn.MSELoss()
optim = torch.optim.SGD(reward_model.parameters())
loss_vals = [] 


for e  in range(400) :
    inputs_np , _ , rewards_np = dataset.sample()
    
    inputs_torch = torch.tensor(inputs_np).float().float()
    rewards_torch = torch.tensor(rewards_np).float()
    
    
    # next_states_torch = torch.tensor(next_state).float()
    
    optim.zero_grad()
    reward_pred = reward_model(inputs_torch)
    loss = loss_fn(reward_pred , rewards_torch)  #is 2 should be torch? 
    loss_vals.append(loss.item())
    loss.backward()
    optim.step()








# the important part of model based rl class 
'''
for planning we use random shooting .

'''
class ModelBasedAgent:
    def __init__(self , reward_model , transition_model , num_random_shoots=20 , horizon_shoot = 3):
        #bacause out models usually make error better to use little horizon
        self.reward_model = reward_model 
        self.transition_model = transition_model 
        self.horizon  = horizon_shoot 
        self.num_random_shoots = num_random_shoots
        
        
        
    
    def select_action_MPC(self , state ):
        #the most important function is this function that w e
        # use model predict control(mpc) to define that .
        #it is openloop
        #it plan in base of input state and the type of planning is open loop 
        #but 
        #in mpc return the first action as output . 
        #action sequences 
        action_seqs =  self.get_action_seqs(self.num_random_shoots)
        action_seqs_return = self.calc_action_seq_return(state , action_seqs)
        best_action_seq_ind = np.argmax(action_seqs_return)
        best_action_seq = action_seqs[best_action_seq_ind] #best open loop plan 
        #because this loop has much more error just use first action 
        first_action = best_action_seq[0]
        return np.array([first_action])

    def get_action_seqs(self , n_seqs):
        action_seqs = []
        
        for i in range(n_seqs):
            action_seq = []
            for j in range(self.horizon):
                action = self.sample_action()
                action_seq.append(action)
            action_seqs.append(action_seqs)
        return action_seqs


    def sample_action(self ,low =-2 , high = 2 ):
        #we use random sampling and amount in (-2 , 2)
        return low + (high-low)*np.random.rand()

    def calc_action_seq_return(self , state , action_seqs):
        returns = []   # is type of reward 
        for i in range(len(action_seqs)):
            rewards = self.calc_return( state , action_seqs[i])
            returns.append(rewards) 
        return returns


    def calc_return(self , state , action_seq):
        #another most important function here. 
        reward_preds = []
        cur_state = state.copy()
        
        for action in action_seq:
            input_np = np.concatenate([cur_state , np.array([action])])
            input_torch = torch.tensor(input_np).float().unsqueeze(0) #get another batch size that is 0 size 
            
            delta_state = self.transition_model(input_torch).detach().numpy().squeeze()
            reward_pred = self.reward_model(input_torch).detach().numpy().squeeze()
            
            cur_state = cur_state + delta_state 
            reward_preds.append(reward_pred)
        
        return np.sum(reward_preds)




n_episod = 500
env = gym.make('Pendulum-v1')

total_reward = [] 
MB_agent = ModelBasedAgent(reward_model , transition_model)




for episod in range(n_episod):
    print('ep: ' ,episod )
    state , _ =env.reset()
    state = np.array(state)
    episod_rewards = []
    episod_states = []
    episod_actions = []
    
    while True:
        print(',' , sep='')
        action = MB_agent.select_action_MPC(state)
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








plt.plot( total_reward )
plt.title('reward values')
plt.show()








