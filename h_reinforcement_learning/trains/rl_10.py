'''policy gradient
actor critic model 
policy gradient.    reinforce: simplest method of policy gradient
usually policy gradient has high learning variance that could decrease
with base line  
a method family that use based line is actor-critic methods that 
uses the approximator for base line 

this code is simplest actor critic code . 
'''
import gymnasium as gym
import torch 
from torch import nn 
import numpy as np 
from matplotlib import pyplot as plt

class Actor(nn.Module):
    #make network
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 16):
        super(Actor , self ).__init__()
        
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
        
    def forward(self , state):
        # print('\n3423 state : ' , state)
        pred = self.estimator(state)
        # print('\n342332432 stkate : ' , state)
        return pred 



class Critic(nn.Module):
    #make network
    def __init__(self , input_dim , output_dim=1 , hidden_dim = 16):
        super(Critic , self ).__init__()
        
        
        layer1 = nn.Linear(input_dim , hidden_dim , bias=True)
        relu = nn.ReLU()
        layer2 = nn.Linear(hidden_dim , output_dim , bias=True)
        self.estimator = nn.Sequential(layer1 , relu , layer2)
        
        
    def forward(self , state):
        # print('\n3423 state : ' , state)
        v_pred = self.estimator(state)
        # print('\n342332432 stkate : ' , state)
        return v_pred 





class AC_agent:
    #make agent
    def __init__(self , state_size , action_size , learning_rate=0.001  , discount_factor=0.99  ):
        self.state_size = state_size 
        self.action_size = action_size 
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor 
        #????
        self.actor  =  Actor(input_dim=state_size , output_dim=action_size , hidden_dim=16)
        
        #if purpose is approximate Q : output_dim = action sieze but if 
        #if purpose is v or state-value function : output_dim = 1
        #so because we saw q learning many times we use monte carlo estimation here means 
        #meanse we haven't temporal-difference  here and we want to estimate 
        # rewardtogo for one state ***not state-action
        #os output_dim = 1 to estimate output as value for input as state
        self.critic =  Critic(input_dim=state_size , output_dim=1 , hidden_dim=16)
        
        
        
        self.mse_loss_fn = nn.MSELoss(reduction='none') # for critic
        self.ce_loss_fn = nn.CrossEntropyLoss(reduction='none') # for actor


        self.actor_optim = torch.optim.Adam(self.actor.parameters() , lr = self.learning_rate)
        self.critic_optim = torch.optim.Adam(self.critic.parameters() , lr = self.learning_rate)
        
        # we didn't need buffer because policy gradient is on-policy method.
        # and in on-policy method not use the last gathered datas.
        # self.buffer = []
    def sample_action(self , state ):
        '''the name of sample action is more sufficient as select action'''
        #get output logits of network 
        #logit: مقادیر حقیقی قبل از تبدیل به احتمال
        # state = torch.tensor(state , dtype=torch.float).unsqueeze(dim=0)
        state = torch.tensor(state)
        # print(f'\n\n\n983states  : ' , state)
        logits = self.actor(state)
        # logits = logits.squeeze(dim=0)
        
        
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
        # print('states 8393 : ' , states)
        # states = torch.Tensor(states)
        
        # def update(self , batch_size=16): its not usefull here. 
        #fo rupdate we need to get the output and could not read from buffer .
        
                
        #alter all to torch.tensor
        actions = torch.tensor(actions , dtype=torch.long)
        rewardToGo = self.calc_reward_to_go(rewards = rewards)
        rewardToGo = torch.tensor(rewardToGo , dtype=torch.float32)
        
        # update value network
        self.critic_optim.zero_grad()
        states = torch.tensor(np.array(states))
        values = self.critic(states)
        loss = self.mse_loss_fn(values.squeeze() , rewardToGo) #?? squeeze
        loss.sum().backward() 
        self.critic_optim.step()
        
        #update policy network
        self.actor_optim.zero_grad()
        logits = self.actor(states) 
        
        with torch.no_grad():
            values = self.critic(states)
        
        advantages = rewardToGo - values 
        log_probs = - self.ce_loss_fn(logits , actions)
        loss = -log_probs * advantages
        loss.sum().backward()
        self.actor_optim.step()
        
        
        




#learning
n_episod = 1500
env = gym.make('CartPole-v1')
agent = AC_agent(env.observation_space._shape[0] ,env.action_space.n  )

#its needed to gathering states ,actions , and rewards in each episod
total_reward = [] 

for episod in range(n_episod):
    state , _ = env.reset()
    state = np.array(state)
    # state = torch.tensor(np.array(env.reset()[0]))
    print(f'\n ep : {episod}: \n'  )
    episod_rewards = []
    episod_states = []
    episod_actions = []
    
    while True:
        # print('\nwhile\n')
        action = agent.sample_action(state= state)
        next_state  , reward , terminated , truncated , _ = env.step(action)
        episod_rewards.append(reward) 
        episod_actions.append(action) 
        episod_states.append(state) 
        state = next_state 
        
        if terminated or truncated :
            break 
    # print(f'''
    #       ep_sta : {episod_states}
    #       ep_ac : {episod_actions}
    #       ep_rew : {episod_rewards}
          
    #       ''')    
    agent.update(states= episod_states, actions= episod_actions , rewards= episod_rewards)
    total_reward.append(np.sum(episod_rewards))  

plt.plot(np.convolve(total_reward , np.ones(100) / 100 ))
plt.show()







