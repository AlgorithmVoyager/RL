
#!/usr/bin/env python3

import numpy as np
import pandas as pd 
import random
from tqdm import tqdm

import copy
import time 


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
    
    def is_target(self, current_state_idx):
        target_idx = self.target[0] * self._cols +  self.target[1]
        if current_state_idx == target_idx:
            return True
        else:
            return False

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
            return -1, state_idx, False
        elif (next_state in self.barrier):
            return -100, next_state_idx, False
        elif (next_state[0] == self.target[0] and next_state[1] == self.target[1]):
            return 1, next_state_idx, True
        else:
            return 0,next_state_idx, False
    

class Sarsa(ActionRewardSystem):
    def __init__(self, cols, rows, action_size, episode_size, alpha = 0.01, epsilon=0.2, gamma=0.7):
        super().__init__(cols, rows)
        self.action_size = action_size
        self.q_sa = pd.DataFrame(columns=["up","right","down","left","stand"],dtype=np.float64)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.episode_size = episode_size
        self.action_table = np.zeros(cols*rows)
        self.state_size = cols*rows
        # self.state_value_table = np.random.randn(cols*rows,action_size)

    def check_state_exist(self,current_state):
        if current_state not in self.q_sa.index:
            print("current_state {} not in q_sa".format(current_state))
            self.q_sa.loc[str(current_state)] = 0
    
    # * epislon greedy algorithm
    def choose_action(self, current_state):
        self.check_state_exist(current_state)
        current_actions = self.q_sa.loc[str(current_state)]

        best_action = np.random.choice(current_actions[current_actions == np.max(current_actions)].index)
        
        if random.random() < self.epsilon * (self.action_size - 1)/ self.action_size:
            actions = list(self.action_set.values())
            print("actions {}".format(actions))
            print("best action {}".format(best_action))
            actions.remove(best_action)
            return np.random.choice(actions)
        return best_action
    
    def policy_improvement(self,current_state):
        current_actions = self.q_sa.loc[str(current_state)]
        if np.random.uniform(0,1) > self.epsilon:
            action = np.random.choice(current_actions[current_actions == np.max(current_actions)].index)
        else:
            action = random.choice(self.action_set)
        self.action_table[current_state] = self.action_reverse_set[action]

    
    def update_q_sa(self,s_t,a_t,r_t1,s_t1,a_t1):
        self.check_state_exist(s_t1)
        q_old = self.q_sa.loc[str(s_t),a_t]
        if not self.is_target(s_t1):
            q_predict = self.q_sa.loc[str(s_t1),a_t1]
            q_new = r_t1 + self.gamma*q_predict
            print("q_predict {} r_t1 {} self.gamma{} ".format(q_predict,r_t1,self.gamma))
        else:
            # stop here 
            q_new = r_t1 
        value = q_old - self.alpha * (q_old -q_new)
        print("s_t {} a_t {}".format(s_t,a_t))
        self.q_sa.loc[str(s_t),a_t] = value
        print("sqa st {} ".format(self.q_sa.loc[str(s_t),a_t]))
        print("q_old {} q_new {} self.alpha{} value{} ".format(q_old,q_new,self.alpha,value))
    
    # * each episold need find target
    def step(self,init_state):
        for i in tqdm(range(self.episode_size),desc="processing"):
            time.sleep(1)
            current_state = init_state
            current_action = self.choose_action(init_state)
            while not self.is_target(current_state):
                reward, next_state, done  = self.get_reward_and_state_by_current_state(current_state, current_action)
                next_action = self.choose_action(next_state)
                print("s_qa {} \n".format(self.q_sa))
                self.update_q_sa(current_state,current_action,reward,next_state,next_action)
                # self.policy_improvement(current_state)
                print("current state {} current action {} reward {} next state {} next actoin {}".format(current_state,current_action,reward,next_state,next_action))
                print("s_qa {} \n".format(self.q_sa))
                current_action = next_action
                current_state = next_state
                
                if done:
                    print(self.q_sa)
                    break
        # self.action_table = np.array([np.max(self.q_sa.loc[str(i),:]) for i in range(self.state_size)])

if __name__ == '__main__':
    saras = Sarsa(5,5,5,1)
    saras.set_barrier()
    saras.set_target()
    # print((saras.action_set))
    # print(random.choice(saras.action_set))
    saras.step(0)
    # print(saras.q_sa)
    print("final action table is {}".format(saras.action_table))
    print("final q_sa table is {}".format(saras.q_sa))
    