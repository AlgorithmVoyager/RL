#!/usr/bin/env python3

import sys
import math
import random
import numpy as np

AVALIABLE_CHOICES= [1,-1,2,-2]
AVALIABLE_CHOICE_NUMBER = len(AVALIABLE_CHOICES)
MAX_ROUND_NUMBER = 10

class State(object):
    """
        记录门特卡罗游戏状态，包括某一个node的状态数据，当前游戏得分，当前游戏round，从开始到当前的执行记录
        需要实现判断当前状态是否达到游戏结束状态，支持action中随即取出操作
    """
    def __init__(self) -> None:
        self.current_value = 0.0
        # for the first root node, the index is 0 & game start from 1
        self.current_round_index = 0
        self.cumulative_choices = []
    

    def get_current_value(self):
        return self.current_value

    def get_current_round_index(self):
        return self.current_round_index
    
    def get_cumulative_choices(self):
        return self.cumulative_choices

    def set_current_value(self,current_value):
        self.current_value = current_value

    def set_current_round_index(self, current_round_index):
        self.current_round_index = current_round_index
    
    def set_cumulative_choices(self, cumulative_choices):
        self.cumulative_choices = cumulative_choices
    
    def is_terminal(self):
        return self.current_round_index == MAX_ROUND_NUMBER
    
    def compute_rewards(self):
        return -abs(1 - self.current_value)
    
    def get_next_state_with_random_choice(self):
        random_choice = random.choice([choice for choice in AVALIABLE_CHOICES])
        next_state = State()
        next_state.set_current_value(self.current_value + random_choice)
        next_state.set_current_round_index(self.current_round_index + 1)
        next_state.set_cumulative_choices(self.cumulative_choices + [random_choice])

        return next_state
    
    def __repr__(self) -> str:
        return "State: {}, value:{}, round:{}, choices{}".format(hash(self),self.current_value,self.current_round_index,self.cumulative_choices)
    


class Node(object):

    def __init__(self):
        self.parent = None
        self.children = []
        self.visited_times=0
        self.quality_value = 0
        self.state = None
    
    def get_parent(self):
        return self.parent
    
    def get_children(self):
        return self.children
    
    def get_visited_times(self):
        return self.visited_times
    
    def get_quality_value(self):
        return self.quality_value

    def get_state(self):
        return self.state
    
    def set_parent(self,parent):
        self.parent = parent
    
    def set_children(self,children):
        self.children = children
    
    def set_visited_times(self,visited_times):
        self.visited_times = visited_times
    
    def set_quality_value(self,quality_value):
        self.quality_value = quality_value
    
    def set_state(self,state):
        self.state = state
    
    def add_visited_times(self):
        self.visited_times += 1
    
    def add_quality_value(self,n):
        self.quality_value += n
    
    def is_all_expanded(self):
        return len(self.children) == AVALIABLE_CHOICE_NUMBER
    
    def add_child(self, sub_node):
        self.children.append(sub_node)
    
    def __repr__(self) -> str:
        return "Node: {}, Q/N: {}/{}, state: {}".format(hash(self),self.quality_value,self.visited_times,self.state)

def best_child(node, is_exploration):
    best_score = -sys.maxsize
    best_sub_node = None

    for sub_node in node.get_children():
        if is_exploration:
            C= 1/math.sqrt(2.0)
        else:
            C= 0.0
        
        score = -sys.maxsize
        left =sub_node.get_quality_value() / sub_node.get_visited_times()
        if is_exploration:
            #UCB quality/times+ C* sqrt(2*ln(total_tims)/times)
            print("node visited time {} , sub node visited times".format(node.get_visited_times(),sub_node.get_visited_times()))
            right = 2*math.log(node.get_visited_times())/(sub_node.get_visited_times())
            # right = 0
            score = left + C*math.sqrt(right+1e-3)
        else:
            score = left

        if score > best_score:
            best_sub_node = sub_node
            best_score = score

    return best_sub_node

def expand(node):
    """
        输入一个节点，再该节点扩展一个新的节点，使用random方法执行action，返回新增的节点，需要保证新增的节点和其他节点action不同
    """
    tried_sub_node_states=[sub_node.get_state() for sub_node in node.get_children()]
    new_state = node.get_state().get_next_state_with_random_choice()

    # check until get the new state which has the different action fromt others
    while new_state in tried_sub_node_states:
        new_state = node.get_state().get_next_state_with_random_choice()
    
    sub_node = Node()
    sub_node.set_state(new_state)
    node.add_child(sub_node)
    return sub_node

def tree_policy(node):
    """
        menta carlo tree search , selectin & expansion
        传入当前需要搜索的节点，根据exploration /exploitation 算法返回返回最好的node，如果节点是叶子节点，直接返回
        基本策略是返回为选择的节点，如果有多个随机选择，没有据选择UCB最大的返回，相等则随机返回
    """
    #check if the current is the leaf node 
    while(node.get_state().is_terminal() == False):
        if node.is_all_expanded():
            node = best_child(node,True)
        else:
            sub_node= expand(node)
            return sub_node
    
    # return the leaf node 
    return node

def default_policy(node):
    """ 
        蒙特卡洛搜索树的simulation阶段，输入一个需要expand的节点，随机操作后创建新的节点，返回新增节点的reward，
        输入的节点应该不是子结点，而是有未执行的action可以expand的节点
        基本测罗是随机选择action
    """
    # get the state of the game
    current_state = node.get_state()

    # run until the game over
    while current_state.is_terminal() == False:
        current_state = current_state.get_next_state_with_random_choice()

    final_state_reword = current_state.compute_rewards()
    return final_state_reword

def backup(node, reward):
    """
        蒙特卡洛backpropagation阶段，输入前面获取需要expand的节点和新执行action的reward，
        反馈给expand节点和上游所有节点更新数据
    """

    # update util the root node
    while node!= None:
        # update the visited times
        node.add_visited_times()
        # update the quality value 
        node.add_quality_value(reward)
        # change the node to the parent node 
        node = node.parent 

def monte_carlo_tree_search(node):
    """
        实现蒙特卡罗树搜索算法，传入一个根节点，再有限的时间内根据之前已经探索过的树结构expand新节点和新数据，
        然后返回只要explotiation最高的子结点

        门特卡罗树搜索的四个步骤：selection, expansion, simulation, backpropagation
        前两步使用tree policy 找到值的搜索的节点
        第三步使用default policy 也就是再选中的节点中随机选一个子结点，并计算reward
        最后一部使用backup把reward更新到所有的选中节点的节点上

        进行预测时，根据q值选择exploitation最大的节点即可，找到下一个最优的节点
    """

    computation_budget = 4

    # run as much as possible under the computation budget
    for i in range(computation_budget):
        # find the best node to expand 
        expand_node = tree_policy(node)

        # random run to add node and get reward
        reward = default_policy(expand_node)

        # update all passing node with reward
        backup(expand_node, reward)

    # get the best node 
    best_next_node = best_child(node, False)
    return best_next_node


def main():
    # create the initialized state and initialize node 
    init_state = State()
    init_node = Node()
    init_node.set_state(init_state)
    current_node = init_node

    # set the round to play
    for i in range(10):
        print("play ground {}".format(i+1))
        current_node = monte_carlo_tree_search(current_node)
        print("choose node: {}".format(current_node))

if __name__ == '__main__':
    main()