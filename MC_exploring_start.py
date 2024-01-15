#!/usr/bin/env python3

import numpy as np
import copy
import time 
import random

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
        self.action_reverse_set = {"up":0,"right":1,"down":2,"left":3,"stand":4}
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
            return -1, next_state_idx
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
        for i in range(episode_length):
            reward,next_state = self.get_reward_and_state_by_current_state(current_state, current_action)
            episode_trajectory.append([current_state,current_action,reward])
            print("next state {}".format(next_state))
            current_state = next_state
            print("episode generated state: {}, current action is {}, get reward: {}".format(current_state,current_action,reward))
            current_action = random.choice(self.action_set)
            
        # * generate episode trajectory 
        return episode_trajectory


class MCExploringStarts(ActionRewardSystem):
    def __init__(self, gamma, action_size, cols, rows, episode_size, episode_length):
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
        self.Returns = {}
        self.Nums = {}
    
    def step(self):
        stable = False
        for i in range(self.episode_size):
            # * generate episode_len traj
            traj = self.generate_episode_trajectory(self.episode_len)
            current_reward = 0.0
            old_action = copy.deepcopy(self.action_table)
            # * calculate returns
            for  s, a, reward in reversed(traj):
                
                current_reward = reward + self.gamma * current_reward
                if (s,a) not in self.Returns:
                    self.Returns[(s,a)] = current_reward
                    self.Nums[(s,a)] = 1
                else:
                    self.Returns[(s,a)] += current_reward
                    self.Nums[(s,a)] += 1
                
                self.q_value_table[s,self.action_reverse_set[a]] =  self.Returns[(s,a)]/self.Nums[(s,a)] if (self.Nums[(s,a)]!= 0) else self.q_value_table[s,a]
                # policy improvement
                self.action_table[s] = np.argmax(self.q_value_table[s,:])
            # if all(x == y for x,y in zip(old_action, self.action_table)):
            #     stable = True
            #     break

            print("current step  \n current q_value_table value: {}".format(np.resize(self.q_value_table,(self.state_size,self.action_size))))
            print("current step  \n current action value: {}".format(np.resize(self.action_table,(self._rows,self._cols))))
        print("current step  \n current q_value_table value: {}".format(np.resize(self.q_value_table,(self.state_size,self.action_size))))
        print("current step  \n current action value: {}".format(np.resize(self.action_table,(self._rows,self._cols))))
        
if __name__ == '__main__':
    mcb = MCExploringStarts(0.8, 5, 5, 5, 200, 100000)
    mcb.set_barrier()
    mcb.set_target()
    mcb.step() 