"""
All printic for training an AI to play Tetris
"""
import math
import random
from collections import namedtuple, deque
from model import action, state
from utils import BOARD_WIDTH, BOARD_HEIGHT

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib
import matplotlib.pyplot as plt

from itertools import count

from view import View

# set up matplotlib

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# torch.set_default_dtype(torch.float)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def _generate_conv_sequence(*channels):
    channels = [c for c in channels]
    k = 5
    while len(channels) > 1:
        cur_in = channels.pop(0)
        cur_out = channels[0]
        yield nn.Conv2d(in_channels=cur_in, out_channels=cur_out, kernel_size=k, stride=1, padding='same')
        yield nn.Softsign()
    return


class DQN(nn.Module):
    """
    Deep Q Learning conv
    """

    def __init__(self, w, h, outputs):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(*_generate_conv_sequence(4, 32, 16, 8))
        self.head = nn.Sequential(
            nn.Linear(w * h * 8, outputs),
        )

    def forward(self, x):
        x = x.to(device)
        x = self.conv(x)
        x = self.head(x.view(x.size(0), -1))
        return x


MEMORY_SIZE = 1000000
BATCH_SIZE = 2**7
GAMMA = 0.99999
EPS_START = 1.00
EPS_END = 0.00
EPS_DECAY = 1000
TARGET_UPDATE = 10

n_actions = action.N

policy_net = DQN(BOARD_WIDTH, BOARD_HEIGHT, n_actions).to(device).double()
target_net = DQN(BOARD_WIDTH, BOARD_HEIGHT, n_actions).to(device).double()
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.RMSprop(policy_net.parameters())
memory = ReplayMemory(MEMORY_SIZE)

steps_done = 0


def select_action(state, init=False) -> int:
    """
    Given state, select next action as int.
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                    math.exp(-1. * steps_done / EPS_DECAY)
    print(eps_threshold)
    if not init:
        steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            head = policy_net(state)
            print(head)
            argmax = head.argmax(1)[0].view(1, 1).item()
            print(f'{argmax=}')
            print(f'{action.Action(argmax)=}')
            return argmax
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.float32).item()


episode_fitnesses = []


def plot_fitnesses():
    plt.figure(2)
    plt.clf()
    fitnesses = torch.tensor(episode_fitnesses, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Fitness')
    plt.plot(fitnesses.numpy())
    # Take 100 episode averages and plot them too
    if len(fitnesses) >= 100:
        means = fitnesses.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool
    )
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the conv computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    state_action_values = policy_net(state_batch.double()).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).argmax(1)[0].detach().float()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    print(loss)

    # Optimize the conv
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


if __name__ == '__main__':
    # s = state.State()
    # pbar = trange(MEMORY_SIZE // 20)
    # pbar.set_description('Initializing memory')
    # for i in pbar:
    #     tensor = torch.tensor(s.get_state(), dtype=torch.float)
    #     tensor = torch.unsqueeze(tensor, 0)
    #     a = select_action(tensor.double(), init=True)
    #     _, reward, done = s.step(action.Action(a))
    #     reward = torch.tensor([reward], device=device)
    #     # Observe new state
    #     next_state = torch.tensor(s.get_state())
    #     next_state = torch.unsqueeze(next_state, 0)
    #     if done:
    #         next_state = None
    #     # Store the transition in memory
    #     memory.push(tensor, torch.tensor([[int(a)]]), next_state, reward)
    #     # Move to the next state
    #     state_tensor = next_state
    #     if done:
    #         s = state.State()

    v = View()
    num_episodes = 5000
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        s = state.State()
        v.draw(s)
        state_tensor = torch.tensor(s.get_state(), dtype=torch.float64)
        state_tensor = torch.unsqueeze(state_tensor, 0)
        for t in count():
            # Select and perform an action
            a = select_action(state_tensor.double())
            _, reward, done = s.step(action.Action(a))
            reward = torch.tensor([reward], device=device, dtype=torch.float64)

            v.draw(s)

            # Observe new state
            next_state = torch.tensor(s.get_state(), dtype=torch.float64)
            next_state = torch.unsqueeze(next_state, 0)
            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state_tensor, torch.tensor([[int(a)]]), next_state, reward)

            # Move to the next state
            state_tensor = next_state

            # Perform one step of the optimization (on the policy network)
            optimize_model()
            if done:
                episode_fitnesses.append(t + 1)
                plot_fitnesses()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')
    plt.show()
