#sarsa

import numpy as np 
import gymnasium as gym
import matplotlib.pyplot as plt

class sarsa:
    def __init__(self , state_shape , num_action , epsilon=0.1 , learning_rate = 0.01 , discount_factor=0.99):
        self.state_shape = state_shape 
        self.num_action = num_action
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor 
        self.epsilon = epsilon 
        
        self.weight = np.zeros((self.state_shape , self.num_action))
        
    def select_action(self , state ) : 
        # print(f'state:{state}\nweights:{self.weight}')
        q_values = np.dot(state , self.weight)
        # print(f'q_val = {q_values}')
        p = np.random.rand()
        
        if p<self.epsilon:
            action = np.random.randint(0 , self.num_action)
        else:
            action = np.argmax(q_values)
        return action 
    
    
    def update(self , state , next_state , reward , action , next_action ):
        # in sarsa we need next action too
        cur_q_values = np.dot(state , self.weight)
        next_q_values = np.dot(next_state , self.weight)
        next_q_value = next_q_values[next_action]
        target = reward + self.discount_factor * next_q_value
        self.weight[: , action ] += self.learning_rate * (target - cur_q_values[action]) * state 
        



#learning loop 
n_episod = 800
env = gym.make('CartPole-v1' , render_mode='human')

agent = sarsa(env.observation_space._shape[0] , env.action_space.n ,discount_factor=0.99 ,  learning_rate=0.01)
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
        
        if i > 200:
            env.render()
        
        if terminated or truncated:
            break
    total_reward.append(np.sum(episode_reward))
    
plt.plot(np.convolve(total_reward , np.ones(100)/100))            
plt.show()     
            