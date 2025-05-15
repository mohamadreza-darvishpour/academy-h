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
        #max step is how many step continue to terminate with or 
        #without result
        self.max_step = max_step
        
        self.n_step = 0 
        self.obs_shape =  (num_row , num_col)
        
        self.action_shape = 4 #up,down,left,right
        
        #setup env 
        self._setup()

        
    
    def _setup(self ):
        '''in setup we define starting and ending position of agent'''    
        #random value for agent's position 
        rand_row = np.random.randint(0 , self.num_row)
        rand_col = np.random.randint(0 , self.num_col)
        self.agent_pos = np.array([rand_row, rand_col])
        
        #random goal
        while True:
            #random value for terminal's position 
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.goal_pos = np.array([rand_row, rand_col])
        
            if not np.array_equal(self.agent_pos , self.goal_pos):
                break
    
    
        #random trap
        while True:
            #random value for traps's position 
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.trap_pos = np.array([rand_row, rand_col])
        
            if (not np.array_equal(self.trap_pos , self.agent_pos)) and  (not np.array_equal(self.trap_pos , self.goal_pos)):
                break
        
    
    def reset(self):
    
        #random new agents pos for new obs
        while True:
            #random value for agent's position 
            rand_row = np.random.randint(0 , self.num_row)
            rand_col = np.random.randint(0 , self.num_col)
            self.agent_pos = np.array([rand_row, rand_col])
        
            if (not np.array_equal(self.trap_pos , self.agent_pos)) and  (not np.array_equal(self.agent_pos , self.goal_pos)):
                break
        
        return self.agent_pos
    
    
    
    def step(self, action:int ):
        '''get action and do the work'''
        '''actions : 0,1,2,3 : right,up,left,down'''
        self.n_step += 1 
        if   action == 0  and self.agent_pos[0] < self.num_row -1  : 
            self.agent_pos[0] += 1 #right

        elif action == 1  and self.agent_pos[1] < self.num_col -1     : 
            self.agent_pos[1] += 1 #up

        elif action == 2  and self.agent_pos[0] >0      : 
            self.agent_pos[0] -= 1 #left

        elif action == 3  and self.agent_pos[1] >0      : 
            self.agent_pos[1] -= 1 #down

        
        done_win = np.array_equal(self.agent_pos , self.goal_pos)
        done_lose = np.array_equal(self.agent_pos , self.trap_pos)
        done_time = self.n_step > self.max_step
        
        done = done_win or done_lose or done_time
        
        reward = -1 
        if done_win:
            reward = self.max_reward
        if done_lose : 
            reward = - self.max_reward 
        if done_time : 
            reward = - self.max_reward
        info = {} # additional info . 
        
        #self.agent_pos : next state
        print('\nstep: ' , self.n_step , '\n')
        self.render()
        return self.agent_pos , reward , done ,info 
        
    def render(self, delay=.01):
        plt.ion()  # Turn on interactive mode
        fig, ax = plt.subplots()
        ax.set_xlim(0, self.num_col)
        ax.set_ylim(0, self.num_row)
        ax.set_xticks(np.arange(0, self.num_col + 1, 1))
        ax.set_yticks(np.arange(0, self.num_row + 1, 1))
        ax.grid(True)
        ax.set_aspect('equal')
        ax.invert_yaxis()

        # Draw agent
        agent = patches.Rectangle((self.agent_pos[1], self.agent_pos[0]), 1, 1, color='green')
        ax.add_patch(agent)

        # Draw goal
        goal = patches.Circle((self.goal_pos[1] + 0.5, self.goal_pos[0] + 0.5), 0.3, color='yellow')
        ax.add_patch(goal)

        # Draw trap
        trap = patches.Rectangle((self.trap_pos[1], self.trap_pos[0]), 1, 1, color='red')
        ax.add_patch(trap)

        plt.title("GridWorld Environment")
        plt.draw()           # Draw the figure
        plt.pause(delay)     # Keep it open for a short time
        plt.close()          # Close after the pause



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
    def select_action(self , state):
        #has scalar to show q values 
        q_values = self.q_table[state[0] , state[1] , : ]
        
        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(0 , self.num_action)
        else:
            action = np.argmax(q_values)
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
episods = 200
agent = tabular_qlearning(env.obs_shape , env.action_shape)
total_reward = []


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
        
    total_reward.append(episod_rewards)