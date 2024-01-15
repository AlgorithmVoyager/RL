import numpy as np
import copy
import time 
import random

import matplotlib.pylab as plt
import matplotlib.patches as patches

# define mode 
class World:
    def __init__(self,cols,rows):
        self._cols = cols
        self._rows = rows 
        self.target = []
        self.barrier = [] 

    # * generate stationary barrier
    def set_barrier(self):
        self.barrier.append([1,1])
        self.barrier.append([1,2])
        self.barrier.append([2,2])
        self.barrier.append([3,1])
        self.barrier.append([3,3])
        self.barrier.append([4,1])
        # self.barrier.append([0,1])
    
    def set_target(self):
         self.target = [3,2]


class ActionRewardSystem(World):
    def __init__(self, cols, rows):
        super().__init__(cols, rows)
        self.action_set = {0:"up",1:"right",2:"down",3:"left",4 : "stand"}
        self.index_change_set = {"up":[-1,0],"right":[0,1],"down":[1,0],"left":[0,-1], "stand" :[0,0]}

    def get_action_set(self):
        return self.action_set

    # * return: reward, state
    def get_reward_and_state_by_current_state(self, state_idx, action):
        cols_idx = state_idx % self._cols
        rows_idx = state_idx // self._cols
        
        index_change = self.index_change_set[action]
        next_state = [rows_idx + index_change[0], cols_idx + index_change[1]]
        next_state_idx = next_state[0] * self._cols + next_state[1]
        # print("state idex {}".format(state_idx))
        # print("next state {}".format(next_state))
        # print("next_state_idx {}".format(next_state_idx))
        # print("self.target {}".format(self.target))
        
        if(next_state[0]< 0 or next_state[0] >= self._cols or next_state[1] < 0 or next_state[1]>=self._rows):
            return -1, state_idx
        elif (next_state in self.barrier):
            return -10, next_state_idx
        elif (next_state[0] == self.target[0] and next_state[1] == self.target[1]):
            return 1, next_state_idx
        else:
            return 0,next_state_idx
    
    def generate_episode_trajectory(self, episode_length, current_state=None, current_action=None):
        episode_trajectory = []
        # * choose start state and action
        current_state = random.randint(0,self._cols*self._rows-1) if current_state is None else current_state
        current_action = random.choice(self.action_set) if current_action is None else current_action
        print("current state {} current actoin ".format(current_state, current_action))
        print(current_action)
        while current_state != self.target[0]*self._rows + self._cols:
            reward,next_state = self.get_reward_and_state_by_current_state(current_state, current_action)
            episode_trajectory.append([current_state,current_action,reward])
            print("next state {}".format(next_state))
            current_state = next_state
            print("episode generated state: {}, current action is {}, get reward: {}".format(current_state,current_action,reward))
            current_action = random.choice(self.action_set)
            

        # * generate episode trajectory 
        return episode_trajectory


class MCBasic(ActionRewardSystem):
    def __init__(self,gamma, action_size, cols, rows, episode_size,episode_length):
        super().__init__(cols, rows)
        self.gamma = gamma
        self.action_size = action_size
        self.state_size = cols * rows
        self.q_value_table = np.zeros((self.state_size, action_size))
        self.v_value = np.zeros(self.state_size)
        self.action_table = np.zeros(self.state_size)
        self.state_value_max_iter = 6
        self.episode_size = episode_size
        self.episode_len = episode_length
    
    def step(self):
        continue_search = True
        iter_step = 0
        while continue_search:
            print('-'*100)
            iter_step += 1
            old_v_value = copy.deepcopy(self.v_value)
            # * generate traj
            for current_state_idx in range(self.state_size):
                for current_action_idx in range(self.action_size):
                    episodes_returns =  self.MC_generate_q(current_state_idx,current_action_idx)
                    self.generate_qsa(episodes_returns, current_state_idx, current_action_idx)
            print("current step {} \n current q value: {}".format(iter_step, np.resize(self.q_value_table,(self.state_size,self.action_size))))
            self.v_value = np.array([np.max(self.q_value_table[i]) for i in range(self.state_size)])
            self.policy_improvement()
            print("current step {} \n old state value: {}".format(iter_step, np.resize(old_v_value,(self._rows,self._cols))))
            print("current step {} \n current state value: {}".format(iter_step, np.resize(self.v_value,(self._rows,self._cols))))
            print("current step {} \n current action value: {}".format(iter_step, np.resize(self.action_table,(self._rows,self._cols))))

            error = self.v_value - old_v_value
            print(error)
            if np.linalg.norm(error) < 1e-3:
                continue_search = False
            
            if iter_step > 1000:
                print("can't search optim res")
                return
                
        
    def MC_generate_q(self,current_state_idx,current_action_idx):
        episodes_returns = 0.0 
        for k in range(self.episode_size):
            traj = self.generate_episode_trajectory(self.episode_len, current_state_idx, self.action_set[current_action_idx])
            current_reward = 0.0
            # * calculate returns
            for state_reward_pair in reversed(traj):
                current_reward = state_reward_pair[-1] + self.gamma * current_reward
            episodes_returns += current_reward
        return episodes_returns
        
    
    def generate_qsa(self,episodes_returns,current_state_idx,current_action_idx):
        self.q_value_table[current_state_idx][current_action_idx] = (episodes_returns) / self.episode_size
        print("current loop generated episodes_returns is {},  self.episode_size is {}, q value is {}".format(episodes_returns, self.episode_size,  self.q_value_table[current_state_idx][current_action_idx]))
    
    def policy_evaluation(self):
        self.v_value = np.array([np.average(self.q_value_table[i]) for i in range(self.state_size) ])
        # self.v_value = np.array([np.max(self.q_value_table[i]) for i in range(self.state_size)])


    def policy_improvement(self):
        self.action_table =  np.array([np.argmax(self.q_value_table[i]) for i in range(self.state_size)])

if __name__ == '__main__':
    # # ars = ActionRewardSystem(5,5)
    # # ars.set_barrier()
    # # ars.set_target()
    # # traj = ars.generate_episode_trajectory(7)
    # # print(traj)
    # action_set = {0:"up",1:"right",2:"down",3:"left",4 : "stand"}
    # print(random.choice(action_set))

    mcb = MCBasic(0.5, 5, 5, 5, 6, 50)
    mcb.set_barrier()
    mcb.set_target()
    mcb.step()



 