#!/usr/bin/env python3

import os 
import random
import time
from tqdm import tqdm

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import deque 


from utils import get_reward

font = {"family" :"Microsoft YaHei",
    "weight":"bold",
    "size":"9"}
matplotlib.rc("font",**font)

class Solver(object):
    def __init__(self, rows : int, cols :int):
        """
            rows : rows of map
            cols : cols of map
        """

        self.idx_to_action = { 0 : "↑", 1 : "->", 2 : "↓" , 3 : "<-", 4 : "o"}
        self.rows, self.cols, self.action_size = rows, cols, len(self.idx_to_action)

        # init matrix
        self.state_value_matrix = np.random.randn(rows, cols)
        self.action_value_matrix = np.random.randn(rows, cols, len(self.idx_to_action))
        self.best_policy = np.random.choice(len(self.idx_to_action), size = (rows,cols))
    
    def show_policy(self):
        for i in self.best_policy.tolist():
            print(*[self.idx_to_action[idx] for idx in i], sep=' ')

    def show_graph(self, graph):
        for i in graph:
            print(*i, sep=' ')
    
    def _clear_console(self):
        if os.name == 'nt':
            _ = os.system('cls')
        else:
            _ = os.system('clear')
    
    def show_point_to_point(self, start_point, end_point, graph):
        assert(0 <= start_point[0] < self.rows)  and (0 <= start_point[1] < self.cols), f'The start_point is {start_point}, is out of range.'
        assert(0 <= end_point[0] < self.rows)  and (0 <= end_point[1] < self.cols), f'The end_point is {end_point}, is out of range.'

        row, col = start_point
        i = 0
        while True:
            graph[row][col] = self.idx_to_action[self.best_policy[row][col]]
            self._clear_console()
            self.show_graph(graph)
            time.sleep(0.5)

            row, col, _ = get_reward((row, col), self.best_policy[row][col], graph)
            if (row, col) == end_point or i > self.rows * self.cols:
                break
            i += 1
    
    def get_epsilon_greedy_action(self, state, epsilon = 0.1):
        row, col = state 
        best_action = np.argmax(self.action_value_matrix[row][col]).item()

        if random.random() < epsilon * (self.action_size - 1)/ self.action_size:
            actions = list(self.idx_to_action.keys())
            actions.remove(best_action)
            return random.choice(actions)
        return best_action

    def mplot(self, x, y, ax, fmt, title, x_label, y_label, legend):
        ax.plot(x,y,fmt)
        ax.set_xlim(x[0],x[-1]+0.4)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(title)
        # ax.set_legend(legend)

class Sarsa(Solver):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows, cols)
    
    def _update_q_value(self, cur_state, cur_action, reward, next_state, next_action, alpha_k = 1e-1, gamma = 0.9):
        cur_row, cur_col = cur_state
        next_row, next_col = next_state

        cur_action_value = self.action_value_matrix[cur_row, cur_col, cur_action]
        next_action_value = self.action_value_matrix[next_row, next_col, next_action]

        # update 
        updated_current_action_value = cur_action_value - alpha_k * (cur_action_value - reward - gamma * next_action_value)
        self.action_value_matrix[cur_row, cur_col, cur_action] = updated_current_action_value
    
    def step(self, epoch, graph, start_state = None, alpha_k = 1e-1, gamma = 0.9):
        cur_state = start_state
        cache = []
        for i in tqdm(range(epoch),desc='processing'):
            if start_state is None:
                cur_state = (random.randint(0, self.rows -1),random.randint(0, self.cols -1))
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state)
            print("curren state {}, current_action {} ".format(cur_state,cur_action))

            j = 0
            while graph[cur_state[0]][cur_state[1]] != "#":
                *next_state, reward  = get_reward(cur_state, cur_action, graph)
                next_action = self.get_epsilon_greedy_action(next_state)
                print("next_state {}, next_action {}  reward {} ".format(next_state,next_action,reward))
                self._update_q_value(cur_state, cur_action,reward,next_state,next_action,alpha_k,gamma)
                cur_state = next_state
                cur_action = next_action
                j+=1
            cache.append(j)
        
        _,axs = plt.subplots(1,1, figsize=(4,3),dpi=150)
        self.mplot(list(range(len(cache))),cache, axs, 'b','Epislon 长度变化图','Episode index','Episode Length',['Sarsa'])

        plt.tight_layout()
        plt.show()

        self.state_value_matrix = np.max(self.action_value_matrix,axis=2)
        self.best_policy =  np.argmax(self.action_value_matrix,axis=2)

class ExpectedSarsa(Solver):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows, cols)
    
    def _get_expected_greedy_q(self, state, epsilon = 0.1):
        row, col = state 
        best_action = np.argmax(self.action_value_matrix[row][col]).item()

        expected_q = (1-epsilon * (self.action_size - 1)/ self.action_size) * self.action_value_matrix[row, col, best_action]
        actions = list(self.idx_to_action.keys())
        actions.remove(best_action)
        for action in actions:
            expected_q += (epsilon / self.action_size) *  self.action_value_matrix[row, col, action]
        return expected_q
    
    def _update_q_value(self, cur_state, cur_action, reward, next_state, alpha_k = 1e-1, gamma = 0.9):
        cur_row, cur_col = cur_state
        # next_row, next_col = next_state

        cur_action_value = self.action_value_matrix[cur_row, cur_col, cur_action]
        next_expected_action_value = self._get_expected_greedy_q(next_state)

        updated_current_action_value = cur_action_value - alpha_k * (cur_action_value - reward - gamma * next_expected_action_value)
        self.action_value_matrix[cur_row, cur_col, cur_action] = updated_current_action_value
    
    def step(self, epoch, graph, start_state = None, alpha_k = 1e-1, gamma = 0.9):
        cur_state = start_state
        cache = []
        for i in tqdm(range(epoch),desc='processing'):
            if start_state is None:
                cur_state = (random.randint(0, self.rows -1),random.randint(0, self.cols -1))
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state)
            print("curren state {}, current_action {} ".format(cur_state,cur_action))

            j = 0
            while graph[cur_state[0]][cur_state[1]] != "#":
                *next_state, reward  = get_reward(cur_state, cur_action, graph)
                # next_action = self.get_epsilon_greedy_action(next_state)
                print("next_state {}, reward {} ".format(next_state,reward))
                self._update_q_value(cur_state, cur_action,reward,next_state,alpha_k,gamma)
                cur_state = next_state
                # cur_action = next_action
                cur_action = self.get_epsilon_greedy_action(cur_state)
                j+=1
            cache.append(j)
        
        _,axs = plt.subplots(1,1, figsize=(4,3),dpi=150)
        self.mplot(list(range(len(cache))),cache, axs, 'b','Epislon 长度变化图','Episode index','Episode Length',['ExpectedSarsa'])

        plt.tight_layout()
        plt.show()

        self.state_value_matrix = np.max(self.action_value_matrix,axis=2)
        self.best_policy =  np.argmax(self.action_value_matrix,axis=2)
    

class NStepSarsa(Solver):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows, cols)
    
    def _update_q_value(self, cur_state, cur_action, total_reward, alpha_k = 1e-1, gamma = 0.9):
        cur_row, cur_col = cur_state
        cur_action_value = self.action_value_matrix[cur_row, cur_col, cur_action]

        # update 
        updated_current_action_value = cur_action_value - alpha_k * (cur_action_value - total_reward)
        self.action_value_matrix[cur_row,cur_col,cur_action] = updated_current_action_value

    def calculate_n_step_reward(self, n_step, n_step_cache, gamma):
        row, col = n_step_cache[-1][0]
        next_action = n_step_cache[-1][1]
        total_reward = self.action_value_matrix[row,col,next_action]
        for j in range(n_step-2,-1,-1):
            total_reward = gamma * total_reward + n_step_cache[j][-1]
        return total_reward
    
    def step(self, epoch, graph, start_state = None, n_step = 6, alpha_k = 1e-1, gamma = 0.9):
        cur_state = start_state
        next_state = cur_state
        n_step_cache = deque()
        cache = []
        
        for i in tqdm(range(epoch),desc='processing'):
            n_step_cache.clear()
            if start_state is None:
                cur_state = (random.randint(0, self.rows -1),random.randint(0, self.cols -1))
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state)
            n_step_cache.append([cur_state, cur_action, 0])
            print("curren state {}, current_action {} ".format(cur_state,cur_action))

            j = 0
            while graph[cur_state[0]][cur_state[1]] != "#":
                *next_state, reward  = get_reward(cur_state, cur_action, graph)
                n_step_cache[-1][-1] = reward
                next_action = self.get_epsilon_greedy_action(next_state)
                cur_state = next_state
                cur_action = next_action
                n_step_cache.append([cur_state, cur_action, 0])
                if len(n_step_cache) == n_step:
                    total_reward = self.calculate_n_step_reward(n_step, n_step_cache, gamma)
                    pop_cur_state,pop_cur_actoin,_ = n_step_cache.popleft()
                    self._update_q_value(pop_cur_state, pop_cur_actoin, total_reward, alpha_k,gamma)
                
                print("next_state {}, next_action {}  reward {} ".format(next_state,next_action,reward))
                j+=1
            cache.append(j)
            cache_len = len(n_step_cache)
            next_row, next_col = n_step_cache[-1][0]
            next_action = n_step_cache[-1][1]
            reward_sum = self.action_value_matrix[next_row,next_col,next_action]
            for j in range(cache_len-2, -1, -1):
                reward_sum = gamma * reward_sum + n_step_cache[j][-1]
                pop_cur_state,pop_cur_actoin,_ = n_step_cache[j]
                self._update_q_value(pop_cur_state, pop_cur_actoin, reward_sum, alpha_k, gamma)
        
        _,axs = plt.subplots(1,1, figsize=(4,3),dpi=150)
        self.mplot(list(range(len(cache))),cache, axs, 'b','Epislon 长度变化图','Episode index','Episode Length',['NStepSarsa'])

        plt.tight_layout()
        plt.show()

        self.state_value_matrix = np.max(self.action_value_matrix,axis=2)
        self.best_policy =  np.argmax(self.action_value_matrix,axis=2)
        self.best_policy =  np.argmax(self.action_value_matrix,axis=2)


class QLearning(Solver):
    def __init__(self, rows: int, cols: int):
        super().__init__(rows, cols)
    
    def _update_q_value(self, cur_state, cur_action, reward, next_state, alpha_k = 1e-1, gamma = 0.9):
        cur_row, cur_col = cur_state
        next_row, next_col = next_state

        cur_action_value = self.action_value_matrix[cur_row, cur_col, cur_action]
        next_action_value = np.max(self.action_value_matrix[next_row, next_col, :])

        # update 
        updated_current_action_value = cur_action_value - alpha_k * (cur_action_value - reward - gamma * next_action_value)
        self.action_value_matrix[cur_row, cur_col, cur_action] = updated_current_action_value
    
    def step(self, epoch, graph, start_state = None, alpha_k = 1e-1, gamma = 0.9):
        cur_state = start_state
        cache = []
        for i in tqdm(range(epoch),desc='processing'):
            if start_state is None:
                cur_state = (random.randint(0, self.rows -1),random.randint(0, self.cols -1))
            else:
                cur_state = start_state
            cur_action = self.get_epsilon_greedy_action(cur_state)
            print("curren state {}, current_action {} ".format(cur_state,cur_action))

            j = 0
            while graph[cur_state[0]][cur_state[1]] != "#":
                *next_state, reward  = get_reward(cur_state, cur_action, graph)
                next_action = self.get_epsilon_greedy_action(next_state)
                print("next_state {}, next_action {}  reward {} ".format(next_state,next_action,reward))
                self._update_q_value(cur_state, cur_action,reward,next_state,alpha_k,gamma)
                cur_state = next_state
                cur_action = next_action
                j+=1
            cache.append(j)
        
        _,axs = plt.subplots(1,1, figsize=(4,3),dpi=150)
        self.mplot(list(range(len(cache))),cache, axs, 'b','Epislon 长度变化图','Episode index','Episode Length',['QLearing'])

        plt.tight_layout()
        plt.show()

        self.state_value_matrix = np.max(self.action_value_matrix,axis=2)
        self.best_policy =  np.argmax(self.action_value_matrix,axis=2)


def Saras():
    sarsa = Sarsa(r,c)
    sarsa.step(1000,graph=graph,start_state=None)
    sarsa.show_policy()
    sarsa.show_point_to_point(start_state, end_state, graph)

def ExpectedSaras():
    sarsa = ExpectedSarsa(r,c)
    sarsa.step(1000,graph=graph,start_state=None)
    sarsa.show_policy()
    sarsa.show_point_to_point(start_state, end_state, graph)

def NStepSaras():
    sarsa = NStepSarsa(r,c)
    sarsa.step(2000,graph=graph,start_state=None)
    sarsa.show_policy()
    sarsa.show_point_to_point(start_state, end_state, graph)

def QLearningmain():
    sarsa = QLearning(r,c)
    sarsa.step(1000,graph=graph,start_state=None)
    sarsa.show_policy()
    sarsa.show_point_to_point(start_state, end_state, graph)


if __name__ == '__main__':
    graph = [
        ['@','@','@','@','@'],
        ['@','x','x','@','@'],
        ['@','@','x','@','@'],
        ['@','x','#','x','@'],
        ['@','x','@','@','@'],
    ]

    r = len(graph)
    c = len(graph[0])
    start_state = (0,0)
    end_state = (3,2)

   
    # Saras()
    # ExpectedSaras()
    # NStepSaras()
    QLearningmain()

