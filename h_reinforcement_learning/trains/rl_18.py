'''
thompson sampling
Bayesian mathematical method
multi armed bandit
we use priors as hyperparameter (bernoulli distribution)
likelihood is type of bernoulli 
prior and postior is type of beta.
'''



import numpy as np 
import gymnasium as gym 
from matplotlib import pyplot as plt 


class Thompson_Sampling_MAB:
    def __init__(self , n_arm=5 ):
        #alpha the importance of ucb
        self.time_step = 0 #the steps of learning
        self.n_arm = n_arm 
        self.prior = np.ones((self.n_arm , 2))
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
        
        samples = []
        for a in range(self.n_arm): #a : current action
            sample = np.random.beta(a = self.prior[a , 0] ,b= self.prior[a , 1])
            samples.append(sample)
    
        selected_arm = np.argmax(samples)
        reward  = np.random.binomial(n=1 , p= self.reward_probs[selected_arm])
        
        self.time_step += 1
        
        
        return selected_arm , reward
        
        #and we should make prior to postior.


    def update(self , arm , reward):
        
        self.n_obs_arm[arm] += 1 
        n = self.n_obs_arm[arm] 
        self.exp_reward_probs[arm] = (reward + self.exp_reward_probs[arm]*(n-1)) / n
        
        if reward:
            self.prior[arm , 0] += 1 
        else:
            self.prior[arm , 1] += 1
        
        
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
rew_list = []
arm_list = []
for e in range(800 ) :
    mab = Thompson_Sampling_MAB()
    rewards , arms  = mab.iterative_run(n_steps= 800 )
    rew_list.append(rewards)
    arm_list.append(arms)
plot(np.mean(rew_list ,  0 ) )
plt.legend()

plt.show()











































