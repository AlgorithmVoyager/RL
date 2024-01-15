import random
import gym
import numpy as np
import collections
from tqdm import tqdm
import torch 
import torch.nn.functional as F
import matplotlib.pyplot as plt 
# import rl_utils 


class ExperienceRelayBuffer(object):

    def __init__(self, capacity) -> None:
        self._buffer = collections.deque(maxlen=capacity)
    
    @property
    def buffer(self):
        return self._buffer
    
    @buffer.setter
    def buffer(self,state,action,reward,next_state,done):
        self._buffer.append((state,action,reward,next_state,done))
    
    def add(self,state,action,reward,next_state,done):
        self._buffer.append((state,action,reward,next_state,done))
    
    def size(self):
        return len(self._buffer)
    
    def sample(self, batch_size):
        transitions = random.sample(self._buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done 


class QNet(torch.nn.Module):

    """
        state_dim is input_dim 
        action_dim is output_dim
    """
   
    def __init__(self, state_dim, hidden_dim, action_dim) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(state_dim,hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim,action_dim)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))
    

class DQN(object):

    def __init__(self, state_dim, hidden_dim, action_dim, learing_rate, gamma, epsilon, target_update, device):
        self._action_dim = action_dim
        self._q_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        # target net 
        self._target_net = QNet(state_dim, hidden_dim, action_dim).to(device)
        # optimizer Adam
        self._optimizer = torch.optim.Adam(self._q_net.parameters(), lr=learing_rate)
        self._gamma = gamma
        self._epsilon = epsilon
        self._target_update = target_update
        self._count = 0
        self._device = device
    
    def take_action(self,state):
        if np.random.random() < (self._epsilon * (self._action_dim - 1)/self._action_dim):
            action = np.random.randint(self._action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self._device)
            action = self._q_net(state).argmax().item()
        return action 
    
    def update(self, transition_dict):
        # get pair from experience replay buffer
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self._device)
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self._device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self._device)
        next_states =torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self._device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self._device)

        # get q_values directly from q_net by states
        q_values = self._q_net(states).gather(1,actions)

        # calculcate q_targets
        # next state max q value 
        max_next_q_values = self._target_net(next_states).max(1)[0].view(-1,1)
        q_targets = rewards + self._gamma * max_next_q_values * (1-dones)

        # calculate optimization function object function error 
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))
        # start optim, backward cal parameter 
        self._optimizer.zero_grad()
        dqn_loss.backward()
        self._optimizer.step()

        if self._count % self._target_update == 0:
            # state dict store parameters
            self._target_net.load_state_dict(self._q_net.state_dict())
        self._count +=1 


def main():
    lr = 2e-3
    num_episodes = 500

    hidden_dim = 128
    epsilon = 0.01
    gamma = 0.9
    target_update = 10
    buffer_size = 10000
    minimal_size = 500
    batch_size = 64

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    env_name = 'CartPole-v0'
    env = gym.make(env_name)

    random.seed(0)
    np.random.seed(0)
    env.seed(0)
    torch.manual_seed(0)

    replay_buffer = ExperienceRelayBuffer(buffer_size)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    agent = DQN(state_dim, hidden_dim, action_dim, lr, gamma, epsilon, target_update, device)

    return_list = []

    for i in range(10):
        with tqdm(total=(num_episodes / 10),desc = 'Iteration %d' % i) as pbar:
            for i_episode  in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                done = False
                while not done:
                    # generate data
                    action = agent.take_action(state)
                    next_state, reward, done, _ = env.step(action)
                    replay_buffer.add(state,action,reward,next_state,done)
                    state = next_state
                    episode_return += reward

                    # experience replay
                    if replay_buffer.size()  > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {
                            'states':b_s,
                            'actions':b_a,
                            'rewards':b_r,
                            'next_states':b_ns,
                            'dones':b_d
                        }
                        # optim and update network
                        agent.update(transition_dict)
                
                return_list.append(episode_return)
                if(i_episode + 1) % 10 == 0:
                    pbar.set_postfix({
                        'episode' : '%d' % (num_episodes/10 * i +i_episode +1),
                        'return':'%.3f' % np.mean(return_list[-10:])
                    })
                pbar.update(1)
    
    episode_list = list(range(len(return_list)))
    plt.plot(episode_list, return_list)
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title('DQN on {}'.format(env_name))
    plt.show()


if __name__ == '__main__':
    main()
