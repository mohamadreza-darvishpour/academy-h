import torch 
from torch import nn 
from matplotlib import pyplot as plt 
import numpy as np 
import gymnasium as gym

'''
when we use class with inhertences of nn we can make something more 
automaitc like forward but need some notice.
'''
'''
write qlearner that use this neural network as approximator
it looks like before code but some notice . 
article : atary game deep reinforcement learning.
'''
'''
*** when we use neural network we can't use one sample updating 
. because one sample updating is lazy process and make optimization 
unstable.  
so its better to use random mini batch of data and buffer . 
should to use random samples because our data isn't i.i.d and 
why? 
if samples are sequential and Data isn't i.i.d and our 
process is regression it caused a problem so we choose data for 
minibatch randomly

'''
class qestimator(nn.Module ):
    def __init__(self , input_dim , output_dim=1 , hidden_dim=16):
        super(qestimator , self).__init__()
        
        #its important to notice dim in layers
        layer1 = nn.Linear(input_dim , hidden_dim ,bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim ,bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
        
    def forward(self , state):
        state = torch.Tensor(state)
        q_pred = self.estimator(state)
        return q_pred
        
        
        
        
class qlearner:
    def __init__(self , state_size , action_size , learning_rate=0.01 , discount_factor=0.95 , epsilon=0.1 ):   
        self.state_size = state_size 
        self.epsilon = epsilon
        self.action_size = action_size 
        self.discount_factor = discount_factor
        
        self.model = qestimator(state_size , output_dim = action_size) 
        
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters() , lr = learning_rate)
        
        self.buffer = []
        
    
    def select_action(self , state):
        # q_values = np.dot(state , self.weight)
        q_values = self.model(state).detach().numpy()
        
        p = np.random.rand()
        
        if p< self.epsilon:
            action = np.random.randint(0 , self.action_size)
        else: 
            action = np.argmax(q_values)
            
        return action 
    
    
    def add_to_buffer(self , state , action , reward , next_state , done):
        self.buffer.append([   state , action , reward , next_state , done  ])
        
    
    def update(self , batch_size = 16):
        
        sampled_inds = np.random.choice(len(self.buffer) ,min(len(self.buffer), batch_size) , replace=False).ravel()
        
        states = []
        targets = []
        
        for ind in sampled_inds:
            sample = self.buffer[ind]
            state , action , reward , next_state , done = sample 
            
            target = reward
            if done:
                target = reward 
                
            else :
                target = reward + self.discount_factor * torch.max(self.model(next_state)).item()
                
            #alter state value to torch.tensor to make code in frame of tensor. not important tooo.
            states.append(torch.tensor(state))  
        
            #for adding targets its important ***
            # update should do for that action currently does.
            #and it means the path of gradient in predictions of 
            #q values must do from that action . 
            # so for this 
            #we need to select the action in the tensor of output of model like when we use in tabular linear approximators in last codes. 
            #here we specify all q values that for all actions as 'target'
            
            #so we predict target values for all actions here. 
            target_q_values = self.model(state)
            #but we change thet target amount of the action chose.
            target_q_values[action] = target 
            #it means when for example the action space is 10 all the prediction of 
            #the border is same just the target we change in a temporal difference way. 
            #so when we use 'mse' all the target difference amount is '0' just this action's target.'
            targets.append(target_q_values)

        #so we should make the made list to torch.tensor again.
        states = torch.vstack(states) # so we have input as tensor in minibatch that could give to model and predict
        # vstack to stick the tensors vertically like np.vstack()
        targets = torch.vstack(targets) #and use this for mse error. 


        self.optimizer.zero_grad()
        loss = self.loss_fn(self.model(state) , targets)
        loss.backward()
        self.optimizer.step()


#learning loop 
n_episod = 500
env = gym.make('CartPole-v1' )
# env = gym.make('CartPole-v1' , render_mode='human')

agent = qlearner(env.observation_space._shape[0] , env.action_space.n ,discount_factor=0.95 , epsilon=0.1 ,  learning_rate=0.01)
total_reward = [] 
    
for i in range(n_episod):
    state , _ = env.reset()
    episode_rewards = []

    action = agent.select_action(state)
    while True:
        next_state , reward , terminated , truncated , _ = env.step(action)
        next_state = np.array(next_state)
        agent.add_to_buffer(state , action , reward , next_state , terminated)
        
        episode_rewards.append(reward)
        state = next_state.copy()
        
        # if i > 200:
        #     env.render()
        
        if terminated or truncated:
            break
    agent.update(  batch_size=16)
    total_reward.append(np.sum(episode_rewards))
    
plt.plot(np.convolve(total_reward , np.ones(100)/100))            
plt.show()     


''''So the result arent good and we should use 
learning rate and discount factore to make it better. 

the *** important notice in deep q learning is 
using target network
when we calculate target values in a temporal difference way(line 125) 
#         loss = self.loss_fn(self.model(state) , targets)
that this contains the predictio of the model itself line(96)
#                target = reward + self.discount_factor * torch.max(self.model(next_state)).item()
these makes the learning unstable
to solve this *****
usually use target network 
that the 'target network' is a clone of our neural network
but its 'weghts' updates in a moving average way *** and we 
don't use the last update to estimate target 
. so this makes our learning more stable. and should notice ...



'''
