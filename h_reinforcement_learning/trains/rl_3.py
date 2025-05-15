
#make custom env 
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from time import sleep

    
    
    
    
class gridworldenv:
    def __init__(self , num_row , num_col , max_reward=40 , max_step=40):
        self.num_row = num_row 
        self.num_col = num_col 
        self.max_reward = max_reward 
        self.max_step = max_step
        
        self.n_step = 0 
        self.obs_shape =  (num_row , num_col)
        self.action_shape = 4 #up,down,left,right
        
        self._setup()
        self._init_render_vars()

    def _setup(self):
        rand_row = np.random.randint(0 , self.num_row)
        rand_col = np.random.randint(0 , self.num_col)
        self.agent_pos = np.array([rand_row, rand_col])

        while True:
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.goal_pos = np.array([rand_row, rand_col])
            if not np.array_equal(self.agent_pos , self.goal_pos):
                break

        while True:
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.trap_pos = np.array([rand_row, rand_col])
            if (not np.array_equal(self.trap_pos , self.agent_pos)) and (not np.array_equal(self.trap_pos , self.goal_pos)):
                break

    def _init_render_vars(self):
        plt.ion()
        self.fig, self.ax = plt.subplots()
        self.ax.set_xlim(0, self.num_col)
        self.ax.set_ylim(0, self.num_row)
        self.ax.set_xticks(np.arange(0, self.num_col + 1, 1))
        self.ax.set_yticks(np.arange(0, self.num_row + 1, 1))
        self.ax.grid(True)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        self.agent_patch = patches.Rectangle((0, 0), 1, 1, color='green')
        self.goal_patch = patches.Circle((self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5), 0.3, color='yellow')
        self.trap_patch = patches.Rectangle((self.trap_pos[1], self.trap_pos[0]), 1, 1, color='red')

        self.ax.add_patch(self.agent_patch)
        self.ax.add_patch(self.goal_patch)
        self.ax.add_patch(self.trap_patch)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def reset(self):
        while True:
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.agent_pos = np.array([rand_row, rand_col])
            if (not np.array_equal(self.trap_pos , self.agent_pos)) and  (not np.array_equal(self.agent_pos , self.goal_pos)):
                break
        self.n_step = 0
        return self.agent_pos

    def step(self, action:int):
        self.n_step += 1 
        if   action == 0  and self.agent_pos[0] < self.num_row -1  : 
            self.agent_pos[0] += 1
        elif action == 1  and self.agent_pos[1] < self.num_col -1     : 
            self.agent_pos[1] += 1
        elif action == 2  and self.agent_pos[0] >0      : 
            self.agent_pos[0] -= 1
        elif action == 3  and self.agent_pos[1] >0      : 
            self.agent_pos[1] -= 1

        done_win = np.array_equal(self.agent_pos , self.goal_pos)
        done_lose = np.array_equal(self.agent_pos , self.trap_pos)
        done_time = self.n_step > self.max_step
        done = done_win or done_lose or done_time

        reward = -1 
        if done_win:
            reward = self.max_reward
        if done_lose or done_time: 
            reward = - self.max_reward

        print('\nstep: ' , self.n_step , '\n')
        self.render()
        ##### self.agent_pos.copy() is very important.
        return self.agent_pos.copy() , reward , done , {}

    def render(self, delay=0):
        self.agent_patch.set_xy((self.agent_pos[1], self.agent_pos[0]))
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        # sleep(delay)




#writing tabular qlearning algorithms. 
class tabular_qlearning:
    def __init__(self , state_shape , num_action, epsilon=.1  , learning_rate=.001 , discount_factor=0.9):
        self.state_shape = state_shape 
        self.epsilon = epsilon
        self.num_action = num_action 
        self.learning_rate = learning_rate 
        self.discount_factor = discount_factor
        self.q_table = np.zeros((*state_shape , num_action))
        self.numaction = num_action
        self.n_table = np.zeros((*state_shape , num_action))
   
   
    def select_action(self , state):
        #has scalar to show q values 
        q_values = self.q_table[state[0] , state[1] , : ]
        
        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(0 , self.num_action)
        else:
            action = np.argmax(q_values)
        return action 
    
    def explore(self , state):
        action = np.argmin(self.n_table[state[0] , state[1],:])
        self.n_table[state[0] , state[1] , action] += 1 
        return action
    
    def update(self , state , next_state , reward , action):
        '''formula'''
        '''q(s,a) = q(s,a) + alpha*(r + gamma* max_a q(s',a') = q(s,a))'''
        
        cur_q_val = self.q_table[state[0] , state[1] , action]
        next_q_val = self.q_table[next_state[0] , next_state[1] , :]
        best_next_q_val = max(next_q_val)
        new_q_val = cur_q_val  + self.learning_rate*(reward + self.discount_factor* best_next_q_val - cur_q_val)
        
        self.q_table[state[0] , state[1] , action ] = new_q_val
        
        
        
        
        
        
        
env = gridworldenv(6,6) 
episods = 10
agent = tabular_qlearning(env.obs_shape , env.action_shape)
total_reward = []


for i in range(episods):
    state = env.reset()
    episod_rewards = [] 
    
    while True : 
        action = agent.explore(state)
        next_state  , reward , done  , info = env.step(action)
        episod_rewards.append(reward)
        agent.update(state , next_state , reward , action)
        state = next_state
        
        if done: 
            break
        
    total_reward.append(np.sum(episod_rewards))

for i in range(episods):
    state = env.reset()
    episod_rewards = [] 
    
    while True : 
        action = agent.select_action(state)
        next_state  , reward , done  , info = env.step(action)
        episod_rewards.append(reward)
        agent.update(state , next_state , reward , action)
        state = next_state
        
        if done: 
            break
        
    total_reward.append(np.sum(episod_rewards))
# Close previous figures and turn off interactive mode
plt.close()
plt.ioff()

window_size = min(5, len(total_reward))
smoothed = np.convolve(total_reward, np.ones(window_size)/window_size, mode='valid')

plt.plot(smoothed)
plt.title("Smoothed Episode Rewards")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.show()  # This will now BLOCK until you close the window