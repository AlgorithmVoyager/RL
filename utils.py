#!/usr/bin/env python3

def get_reward(state, action, graph):
    r, c = len(graph), len(graph[0])
    reward = -1

    row, col = state
    next_row, next_col = state 
    if action == 0:
        next_row = row - 1
    elif action == 1:
        next_col = col + 1
    elif action == 2:
        next_row = row + 1
    elif action == 3:
        next_col = col -1
    
    if next_row < 0 or next_row >= r or next_col < 0 or next_col >= c :
        reward = -1
        next_row, next_col = state
    elif graph[next_row][next_col] == 'x':
        reward = -10
    elif graph[next_row][next_col] == '#':
        reward = 1
    return next_row, next_col, reward

        
