#policy gradient.    reinforce: simplest method of policy gradient
#usually policy gradient has high learning variance that could decrease
#with base line (future trains)
'''
we need a parametrical function for policy. usually use 
neural network for that. 


'''

import gymnasium as gym
import torch 
from torch import nn 
import numpy as np 
from matplotlib import pyplot as plt

class Policy(nn.Module):
    #make network
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 16):
        super(Policy , self ).__init__()
        
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
        
    def forward(self , state):
        # print('\n3423 state : ' , state)
        pred = self.estimator(state)
        # print('\n342332432 stkate : ' , state)
        return pred 



class PG_agent:
    #make agent
    def __init__(self , state_size , action_size , learning_rate=0.001  , discount_factor=0.99  ):
        self.state_size = state_size 
        self.action_size = action_size 
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor 
        #????
        self.policy_model = Policy(input_dim=state_size , output_dim=action_size , hidden_dim=16)
        self.loss_fn = nn.CrossEntropyLoss(reduction='none')
        #Adam , SGD : could use adam as optimizer too.
        self.optimizer = torch.optim.Adam(self.policy_model.parameters() , lr = self.learning_rate)
        
        # we didn't need buffer because policy gradient is on-policy method.
        # and in on-policy method not use the last gathered datas.
        # self.buffer = []
    def sample_action(self , state ):
        '''the name of sample action is more sufficient as select action'''
        #get output logits of network 
        #logit: مقادیر حقیقی قبل از تبدیل به احتمال
        # print('\n98234 : ' , state)
        state = torch.tensor(state , dtype=torch.float).unsqueeze(dim=0)
        logits = self.policy_model(state)
        logits = logits.squeeze(dim=0)
        #so we need to alter this logits to probability.
        #so we enable softmax  on logits
        probs = torch.nn.functional.softmax(logits , dim= -1)
        
        # so according the probs gotten to any actions we take 
        # sample based on that. 
    
        #sampling    input:is the probs distribution , num_samples means just select and return one of the index probs.
        action = torch.multinomial(input = probs , num_samples=1)
        # print('9134\n: ' , action)
        return int(action.item())  

    
    def calc_reward_to_go(self , rewards):
        rewardsToGo = np.array(rewards  )
        
        for i in range(2  , len(rewards)+1 ): 
            #update reverse from start to end
            rewardsToGo[-i] = rewards[-i] + self.discount_factor*rewardsToGo[-i+1]
        return rewardsToGo
    
    def update(self , states , actions , rewards):
        # def update(self , batch_size=16): its not usefull here. 
        #fo rupdate we need to get the output and could not read from buffer .
        
                
        #alter all to torch.tensor
        actions = torch.tensor(actions) 
        rewardToGo = self.calc_reward_to_go(rewards = rewards)
        rewardToGo = torch.tensor(rewardToGo)
        self.optimizer.zero_grad()
        
        states = torch.Tensor(np.array(states))
        #get states to policy and get the returned logits
        logits = self.policy_model(states) 
        
        
        #now we have all needed variables for calculating policy gradient.  
        # we use a trick here.
        # for policy gradient we used something like??? as a trick but these are theory
        
        #so policy gradient is very colse to 'imitation learning' and the likelihood of 
        # policy gradient is like likelihood in supervised learning
        # so we can use this likelihood as an objective function. 
        # because the policy gradient give us something in gradient type. 
        # but in automatic library to get gradient we need to 'define' a 'loss function' 
        
        # *** so we use this similarity to supervised learning and use 
        # loss functions for calculate gradients

        # if action space is descrete(کسسته) we use 'Cross Entropy Loss' 
        # if action space is continuous(کسسته) we use 'Mean Squared Loss' 
        
        
        #finally when we differentiating(مشتق گرفتن) an 'loss function' and
        # enable a weight on that ; we get policy gradient.

        log_probs = - self.loss_fn(logits , actions) 
        # reduction=False to not return a number amount like mean
        #this loss(log_probs) is like classification loss but in policy gradient
        # nay item of 'log probability ) multiplies to a scale( reward set of actions)
        # so our reward_to_go does credit assignment to any actions to increase 
        # probabilty of that in learning 
        loss = -  log_probs * rewardToGo
        #but in policy gradient we should increase 'log_probs * rewardToGo' and enble gradient ascend
        #but because in torch loss is minimizing we make it negative.
        
        
        loss.sum().backward()
        self.optimizer.step()





#learning
n_episod = 1500
env = gym.make('CartPole-v1')
agent = PG_agent(env.observation_space._shape[0] ,env.action_space.n  )

#its needed to gathering states ,actions , and rewards in each episod
total_reward = [] 

for episod in range(n_episod):
    state = np.array(env.reset()[0])
    # print('9999: \n'  , state)
    episod_rewards = []
    episod_states = []
    episod_actions = []
    
    while True:
        action = agent.sample_action(state= state)
        next_state  , reward , terminated , truncated , _ = env.step(action)
        episod_rewards.append(reward) 
        episod_actions.append(action) 
        episod_states.append(state) 
        state = next_state 
        
        if terminated or truncated :
            break 
        
    agent.update(states= episod_states, actions= episod_actions , rewards= episod_rewards)
    total_reward.append(np.sum(episod_rewards))  

plt.plot(np.convolve(total_reward , np.ones(100) / 100 ))
plt.show()