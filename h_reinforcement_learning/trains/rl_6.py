#double qlearner
# qlearning , in approximate and tabular way . 
#not sufficient for complicated env 
#for complicated ones use nns

import numpy as np 
import gymnasium as gym
import matplotlib.pyplot as plt

class double_qlearner:
    def __init__(self , state_shape , num_action , epsilon=0.1 , learning_rate = 0.01 , discount_factor=0.99):
        self.state_shape = state_shape 
        self.num_action = num_action
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor 
        self.epsilon = epsilon 
        
        self.weight1 = np.zeros((self.state_shape , self.num_action))
        self.weight2= np.zeros((self.state_shape , self.num_action))
        
    def select_action(self , state ) : 
        # print(f'state:{state}\nweights:{self.weight}')
        q_values1 = np.dot(state , self.weight1)
        q_values2 = np.dot(state , self.weight2)
        q_values = q_values1 + q_values2
        # print(f'q_val = {q_values}')
        p = np.random.rand()
        
        if p<self.epsilon:
            action = np.random.randint(0 , self.num_action)
        else:
            action = np.argmax(q_values)
        return action 
    
    
    def update(self , state , next_state , reward , action , next_action ):
        # in sarsa we need next action too
        cur_q_values1 = np.dot(state , self.weight1)
        next_q_values1 = np.dot(next_state , self.weight1)
        best_next_q_value1 = max(next_q_values1)

        cur_q_values2 = np.dot(state , self.weight2)
        next_q_values2 = np.dot(next_state , self.weight2)
        best_next_q_value2 = max(next_q_values2)
        
        
        target1 = reward + self.discount_factor * best_next_q_value2
        target2 = reward + self.discount_factor * best_next_q_value1
        self.weight1[: , action ] += self.learning_rate * (target1 - cur_q_values1[action]) * state 
        self.weight1[: , action ] += self.learning_rate * (target2 - cur_q_values2[action]) * state 
        



#learning loop 
n_episod = 1000
env = gym.make('CartPole-v1')
# env = gym.make('CartPole-v1' , render_mode='human')

agent = double_qlearner(env.observation_space._shape[0] , env.action_space.n ,discount_factor=0.99 ,  learning_rate=0.01)
total_reward = [] 

for i in range(n_episod):
    state , _ = env.reset()
    episode_reward = []
    action = agent.select_action(state)

    while True:
        next_state , reward , terminated , truncated , _ = env.step(action)
        next_state = np.array(next_state)
        episode_reward.append(reward)
        #wrong maybe choose action for next not current one.
        # next_action = agent.select_action(next_state)
        next_action = agent.select_action(state)
        agent.update(state , next_state , reward , action , next_action)
        state = next_state.copy()
        action = next_action
        
        # if i > 200:
        #     env.render()
        
        if terminated or truncated:
            break
    total_reward.append(np.sum(episode_reward))
    
plt.plot(np.convolve(total_reward , np.ones(100)/100))            
plt.show()     
            