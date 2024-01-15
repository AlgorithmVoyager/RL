import numpy as np
import copy
import time 

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
        print("state idex {}".format(state_idx))
        print("next state {}".format(next_state))
        print("next_state_idx {}".format(next_state_idx))
        print("self.target {}".format(self.target))
        
        if(next_state[0]< 0 or next_state[0] >= self._cols or next_state[1] < 0 or next_state[1]>=self._rows):
            return -1, state_idx
        elif (next_state in self.barrier):
            return -1, next_state_idx
        elif (next_state[0] == self.target[0] and next_state[1] == self.target[1]):
            return 1, next_state_idx
        else:
            return 0,next_state_idx


class PolicyIteration(ActionRewardSystem):
    def __init__(self, gamma, action_size, cols, rows):
        super().__init__(cols, rows)
        self.gamma = gamma
        self.action_size = action_size
        self.state_size = cols * rows
        self.q_value_table = np.zeros((self.state_size, action_size))
        self.v_value = np.ones(self.state_size)
        self.action_table = np.zeros(self.state_size)
        self.state_value_max_iter = 6
    
    def solve(self):
        continue_search = True
        iter_step = 0
        while continue_search:
            iter_step += 1
            last_step_v_value = copy.deepcopy(self.v_value)
            # * policy evaluation step
            get_optim_value_status_at_current_step = False
            current_state_value_iter = 0
            while current_state_value_iter < self.state_value_max_iter:
                # * store old value 
                old_v_value = copy.deepcopy(self.v_value)
                current_state_value_iter += 1
                
                for i in range(self.state_size):
                    for j in range(self.action_size):
                        reward, next_state_idx = self.get_reward_and_state_by_current_state(i, self.action_set[j])
                        print(reward, next_state_idx)
                        self.q_value_table[i,j] = reward + self.gamma * old_v_value[next_state_idx]
                
                # update current iterate value state
                self.v_value = np.array([np.max(self.q_value_table[i]) for i in range(self.state_size)])
                print("current step {} \n old state value: {}".format(iter_step, np.resize(old_v_value,(self._rows,self._cols))))
                print("current step {} \n current state value: {}".format(iter_step, np.resize(self.v_value,(self._rows,self._cols))))
                error = self.v_value - old_v_value
                if np.linalg.norm(error) < 1e-3:
                    get_optim_value_status_at_current_step = True
           
            print("current step {} \n current action value: {}".format(iter_step, np.resize(self.action_table,(self._rows,self._cols))))

            # * policy improvment step
            
            # update q value table
            for i in range(self.state_size):
                    for j in range(self.action_size):
                        reward, next_state_idx = self.get_reward_and_state_by_current_state(i, self.action_set[j])
                        print(reward, next_state_idx)
                        self.q_value_table[i,j] = reward + self.gamma * old_v_value[next_state_idx]
            self.action_table =  np.array([np.argmax(self.q_value_table[i]) for i in range(self.state_size)])
            print("current step {} \n current action value: {}".format(iter_step, np.resize(self.action_table,(self._rows,self._cols))))

            error = self.v_value - last_step_v_value
            if np.linalg.norm(error) < 1e-3:
                continue_search = False
            
            if iter_step > 100000:
                print("can't search optim res")
                return
    
    def get_optim_action(self):
        return self.action_table
    
    def get_optim_v_value(self):
        return self.v_value
    

if __name__ == '__main__':
    b_optim = PolicyIteration(0.10,5,5,5)
    b_optim.set_barrier()
    b_optim.set_target()
    b_optim.solve()

    optim_action = b_optim.get_optim_action()
    optim_v = b_optim.get_optim_v_value()

    print("final action value {} \n".format(optim_action))
    print("final state value {}".format(optim_v))
    
    


