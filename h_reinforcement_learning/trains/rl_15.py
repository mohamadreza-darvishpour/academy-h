'''
epsilon decay multi armed bandit

'''




'''
multi armed bandit 
epsilon greedy

'''
import numpy as np 
import gymnasium as gym 
from matplotlib import pyplot as plt 


class EpsilonDecay_MAB:
    def __init__(self , n_arm=5 , epsilon = 0.2 , alpha = 0.5 ):
        self.alpha = alpha 
        self.time_step = 0 #the steps of learning
        self.epsilon = epsilon
        self.n_arm = n_arm 
        self.n_obs_arm = np.zeros(n_arm)
        self.exp_reward_probs = np.zeros(n_arm)
        self.best_arm = np.argmax(self.exp_reward_probs)
        
        '''
        for any arm we need reward or expected reward(more exact)
        that has been done with a function.
        '''
        self.reward_probs = self.make_reward_probs(self.n_arm)
        
    def make_reward_probs(self , n_arm):
        '''
        we define reward probs to mean of the reward any arm can return
        and is a vector(بردار) .  
        '''
        reward_probs = np.random.uniform(size = n_arm)
        return reward_probs


    def step(self ) :
        
        p = np.random.rand() 
        cur_epsilon = self.epsilon * self.alpha **(0.01 * self.time_step)
        
        
        if p < cur_epsilon: #explore
            selected_arm = np.random.randint(0 , self.n_arm)
        else:   #exploit
            selected_arm = np.argmax(self.exp_reward_probs)
            
        reward  = np.random.binomial(n=1 , p= self.reward_probs[selected_arm])
        
        self.time_step += 1
        
        
        return selected_arm , reward
        



    def update(self , arm , reward):
        
        self.n_obs_arm[arm] += 1 
        n = self.n_obs_arm[arm] 
        self.exp_reward_probs[arm] = (reward + self.exp_reward_probs[arm]*(n-1)) / n
        
        
        
    def iterative_run(self , n_steps=100):
        rewards = []
        selected_arms = []
        for t in range(n_steps):
            arm , reward = self.step()
            self.update(arm , reward)
            rewards.append(reward)
            selected_arms.append(arm)
        return rewards ,selected_arms








def plot(rewards , name=''):
    plt.plot(rewards  ,label =name)


plt.figure()
for eps in [0.01 , 0.05 , 0.3]:
    rew_list = []
    arm_list = []
    for e in range(800 ) :
        mab = EpsilonDecay_MAB(epsilon=eps ,alpha=0.5)
        rewards , arms  = mab.iterative_run(n_steps= 800 )
        rew_list.append(rewards)
        arm_list.append(arms)
    name = f'epsilon = {eps}'
    plot(np.mean(rew_list ,  0 ) , name)
    plt.legend()

plt.show()



















