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


class Visualization:
    def drawWorld(self,ax, rows, cols, barrier, targets):
        squares = []
        for x in range(rows):
            for y in range(cols):
                facecolor = 'none'
                if [x,y] in barrier:
                    facecolor ='b'
                elif [x,y] in targets:
                    facecolor ='r'

                square = patches.Rectangle((x-0.5,y-0.5),1,1,linewidth = 1, edgecolor ='b',facecolor=facecolor)
                ax.add_patch(square)
    
    def drawActoin(self,ax, rows, cols, action_list):
        arrows = []
        cur_idx = 0
        for x in range(rows):
            for y in range(cols):
                dx = 0
                dy = 0 
                color = 'g'
                if action_list[cur_idx] == 0:
                    dx = 0
                    dy = -0.5
                    color = 'g'
                elif action_list[cur_idx] == 1:
                    dx = 0.5
                    dy = 0
                    color = 'g'
                elif action_list[cur_idx] == 2:
                    dx = 0
                    dy = 0.5
                    color = 'g'
                elif action_list[cur_idx] == 3:
                    dx = -0.5
                    dy = 0
                    color = 'g'
                elif action_list[cur_idx] == 4:
                    dx = 0.1
                    dy = 0.1
                    color = 'b'
                arrow = patches.FancyArrowPatch((x,y),(x+dx,y+dy),arrowstyle='->',mutation_scale=15,color=color)
                ax.add_patch(arrow)
                cur_idx += 1
    
    def drawStateValue(self,ax,rows,cols,state_list):
        idx = 0 
        for x in range(rows):
            for y in range(cols):
                ax.text(x-0.5,y-0.5,str(state_list[idx]),fontsize=12,ha='center',va='center')
                idx += 1

    def showfig(self,ax, x_l,x_u,y_l,y_u):
        plt.gca().set_aspect('equal',adjustable='box')
        plt.grid()
        plt.show()

        ax.set_xlim(x_l,x_u)
        ax.set_ylim(y_l,y_u)

        time.sleep(1)
        plt.close()


class BellmanOptimEquations(ActionRewardSystem):
    def __init__(self, gamma, action_size, cols, rows):
        super().__init__(cols, rows)
        self.gamma = gamma
        self.action_size = action_size
        self.state_size = cols * rows
        self.q_value_table = np.zeros((self.state_size, action_size))
        self.v_value = np.zeros(self.state_size)
        self.action_table = np.zeros(self.state_size)
        self.vis = Visualization()
    
    def solve(self):
        continue_search = True
        iter_step = 0
        while continue_search:
            iter_step += 1
            old_v_value = copy.deepcopy(self.v_value)

            fig, ax = plt.subplots()

            for i in range(self.state_size):
                for j in range(self.action_size):
                    reward, next_state_idx = self.get_reward_and_state_by_current_state(i, self.action_set[j])
                    print(reward, next_state_idx)
                    self.q_value_table[i,j] = reward + self.gamma * old_v_value[next_state_idx]
            
            self.v_value = np.array([np.max(self.q_value_table[i]) for i in range(self.state_size)])
            self.action_table =  np.array([np.argmax(self.q_value_table[i]) for i in range(self.state_size)])

            # self.vis.drawWorld(ax,self._rows,self._cols,self.barrier,self.target)
            # self.vis.drawActoin(ax,self._rows,self._cols,self.action_table)
            # self.vis.drawStateValue(ax,self._rows,self._cols,self.v_value)
            # self.vis.showfig(ax,-2,self._rows,-2,self._cols)


            print("current step {} \n old state value: {}".format(iter_step, np.resize(old_v_value,(self._rows,self._cols))))
            print("current step {} \n current action value: {}".format(iter_step, np.resize(self.action_table,(self._rows,self._cols))))
            print("current step {} \n current state value: {}".format(iter_step, np.resize(self.v_value,(self._rows,self._cols))))
            
            error = self.v_value - old_v_value
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
    b_optim = BellmanOptimEquations(0.50,5,5,5)
    b_optim.set_barrier()
    b_optim.set_target()
    b_optim.solve()

    optim_action = b_optim.get_optim_action()
    optim_v = b_optim.get_optim_v_value()

    
    


