'''
multi armed bandits
this is more simple subjects in reinforcement learning
and choosing in this method is not sequential 
and are stateless
and 
we get reward just after make decision and finish trade off with environment.
.
any experience is independent from another one. 

its better to do simulation in a class. 

'''
import numpy as np 
import gymnasium as gym 
from matplotlib import pyplot as plt 


class MultiArmedBandit:
    def __init__(self , n_arm=5):
        self.n_arm = n_arm 
        self.n_obs_arm = np.zeros(n_arm)
        self.exp_reward_probs = np.zeros(n_arm)
        self.best_arm = np.argmax(self.reward_probs)
        
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
        '''
        this func to decide one arm.
        it could have a strategy or not. 
        '''
        #now just stochastic
        selected_arm = np.random.randint(0 , self.n_arm)
        
        # calculate reward with sampling from bernoulli distribution '''
        #binomial distribution get 'n' that is trial number and 
        #is probablity for bernoulli dist to be '1' . 
        #so we get p as it is below
        reward = np.random.binomial(n = 1 , p=self.reward_probs[selected_arm])
        #This line simulates a reward using a 
        # Bernoulli distribution, which
        # is a special case of the binomial distribution.
        return selected_arm ,reward



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


def plot(rewards):
    plt.plot(rewards)
    plt.show()



rew_list = []
arm_list = []
for e in range(100 ) :
    mab = MultiArmedBandit()
    rewards , arms  = mab.iterative_run(n_steps= 100 )
    rew_list.append(rewards)
    arm_list.append(arms)


plot(np.mean((rew_list), 0 ))





