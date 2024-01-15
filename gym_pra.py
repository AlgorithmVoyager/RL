import gym
import numpy as np
import copy


# def main():
#     env = gym.make('CartPole-v0')
#     env.reset()

#     for i_episode in range(20):
#         observation = env.reset()
#         for t in range(100):
#             env.render()
#             print(observation)
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             if done: # 如果结束, 则退出循环
#                 print("Episode finished after {} timesteps".format(t+1))
#                 break
#     env.close()

env = gym.make('CartPole-v0')
action_space = env.action_space.n
state_space = env.observation_space.shape[0]
policy = np.ones((state_space,action_space))/action_space

value_function = np.zeros(state_space)


def policy_evaluation(policy, value_function, gamma = 0.99, theta= 1e-6):
    while True:
        delta = 0
        for state in range(state_space):
            v = value_function[state]
            new_v = 0
            for action in range(action_space):
                for prob, next_state, reward, _ in env.P[state][action]:
                    new_v += policy[state][action] * prob *(reward + gamma * value_function[next_state])
            value_function[state] = new_v
            delta = max(delta,max(v-value_function[state]))
        
        if delta < theta:
            break

def policy_improvement(policy, value_function, gamma = 0.99):
    policy_stable = True
    for state in range(state_space):
        old_action = np.argmax(policy[state])
        action_values = np.zeros(action_space)
        for action in range(action_space):
            for prob, next_state, reward, _ in env.P[state][action]:
                action_values[action] += prob*(reward + gamma * value_function[next_state])
        
        best_action = np.argmax(action_values)
        policy[state] = np.eye(action_space)[best_action]

        if old_action != best_action:
            policy_stable = False
    return policy, policy_stable

def main():
    total_reword = 0
    num_episodes = 10
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            print("state {}".format(state))
            print("policy {}".format(policy))
            print("policy[state] {}".format(policy[state]))
            action = np.argmax(policy[state])
            state, reward, done, _ = env.step(action)
            total_reword += reward
    average_reward = total_reword/num_episodes

    print("average reward over {num_episodes} episodes : {average_reward}")

    env.close()


if __name__ == '__main__':
    main()