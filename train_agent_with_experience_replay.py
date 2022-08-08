"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import statistics
from queue import PriorityQueue
import matplotlib.pyplot as plt
from random import randint
import torch.nn.functional as F
import pickle
import gym
import numpy as np
import argparse
import wimblepong
from agent.agent import Agent 
import torch
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30000)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--keep_playing_reward", action="store_true")
args = parser.parse_args()

class MemoryBuffer():
    def __init__(self, size):
        # rewards, states
        self.buffer = []
        self.size = size
    def insert(self, replay):
        if len(self.buffer) == self.size:
            self.buffer = self.buffer[1:] + [replay]
            self._sort()
        else:
            self.buffer = self.buffer + [replay]
            self._sort()

    def sample(self, num):
        indices = np.random.randint(0, high=len(self.buffer), size=num, dtype=int)
        # samples = np.random.choice(self.buffer, size=min(num, len(self.buffer)), replace=False)
        samples = [self.buffer[i] for i in indices]
        return samples

    def reset(self):
        self.buffer = []

    def _sort(self):
        self.buffer = sorted(self.buffer, key = lambda x:(x[-1], len(x[0])))

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]

def compute_keep_playing_reward(length):
    # print(max(0, (length-30)/30))
    return max(0, (length-30)/30)

def discounted_returns(rewards, gamma=0.999):
    ## Init R
    R = 0
    returns = list()
    for reward in reversed(rewards):
        R = reward + gamma * R
        #print(R)
        returns.insert(0, R)
        #returns.append(R)

    returns = torch.tensor(returns)
    
    ## normalize the returns
    returns = (returns - returns.mean()) / (returns.std() + 1e-6)
    return returns

def discount_reward(r, gamma=0.999):
    discounted_r = np.zeros_like(r, dtype=np.float32)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def roll_out(agent, env):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    values = []
    dones = []
    probs = []
    wins = 0
    is_done = False

    while not is_done:
        if not args.headless:
            env.render()
        states.append(state)
        action, value, policy_prob = agent.choose_action(torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2))
        # log_softmax_action = actor_network(Variable(torch.Tensor([state])))
        # softmax_action = torch.exp(log_softmax_action)
        # action = np.random.choice(ACTION_DIM,p=softmax_action.cpu().data.numpy()[0])
        # one_hot_action = [int(k == action) for k in range(ACTION_DIM)]
        next_state, reward, done, info = env.step(action)
        if not done and args.keep_playing_reward:
            new_reward = compute_keep_playing_reward(len(rewards)+1) + reward
        else:
            new_reward = reward
        #fix_reward = -10 if done else 1
        actions.append(action)
        rewards.append(new_reward)
        values.append(value)
        dones.append(done)
        probs.append(policy_prob.logits[0][action])
        state = next_state
        if done:
            if reward == 10:
                wins += 1
            is_done = True
            env.reset()
            break

    return states, actions, rewards, probs, values, wins

def compute_loss(rewards, values, probs):
    discounted_rewards = discount_reward(rewards)
    value_loss = F.mse_loss(torch.concat(values), torch.tensor(discounted_rewards, dtype=torch.float32))
    advantages = torch.tensor(discounted_rewards, dtype=torch.float32) - torch.concat(values).detach()
    actor_loss = - torch.mean(torch.concat(probs) * advantages)
    return actor_loss, value_loss

def compute_loss_from_replay(replay, agent):
    actor_loss_sum, value_loss_sum = [], []
    for rep in replay:
        rewards, states, actions, wins = rep
        actions, values, policy_probs = agent.choose_action(torch.stack([torch.tensor(state.copy(), dtype=torch.float32) for state in states]).permute(0, 3, 1, 2))
        discounted_rewards = discount_reward(rewards)
        value_loss = F.mse_loss(values, torch.tensor(discounted_rewards, dtype=torch.float32))
        advantages = torch.tensor(discounted_rewards, dtype=torch.float32) - values.detach()
        probs = policy_probs.logits[[i for i in range(len(actions))], actions]
        actor_loss = - torch.mean(probs * advantages)
        actor_loss_sum.append(actor_loss)
        value_loss_sum.append(value_loss)
    return torch.mean(torch.stack(actor_loss_sum)), torch.mean(torch.stack(value_loss_sum))


def train(env, agent, num_step, actor_optimizer, critic_optimizer):
    # init a task generator for data fetching
    wins1 = 0
    length = []
    buffer = MemoryBuffer(20)

    for step in range(num_step):
        if step % 5 == 4:
            env.switch_sides()
        if step % 1000 == 999:
            torch.save(agent.state_dict(), "agent_{}.pkl".format(step))
        states, actions, rewards, probs, values, wins = roll_out(agent, env)
        buffer.insert([rewards, states, actions, wins])
        length.append(len(states))
        wins1 += wins
        replay = buffer.sample(3)
        print("\tInformation on Selected replay: Mean Length {:.3f} Wins {}".format(statistics.mean([len(rep[0]) for rep in replay]), sum([rep[-1] for rep in replay])))
        actor_loss, value_loss = compute_loss_from_replay(replay, agent)
        # actor_loss, value_loss = compute_loss(rewards, values, probs)

        # train actor network
        actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm(agent.policy.parameters(), 0.5)
        actor_optimizer.step()

        # train value network
        critic_optimizer.zero_grad()
        value_loss.backward()
        torch.nn.utils.clip_grad_norm(agent.critic.parameters(), 0.5)
        critic_optimizer.step()

        to_print = " Episode {} Length {} Mean Length {:.3f} Actor Loss: {:.6f} Value Loss: {:.3f} Wins: {} Broken WR: {}".format(step, length[-1], statistics.mean(length), actor_loss, value_loss, wins, wins1/(step+1))
        print(to_print)

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
ob1 = env.reset()

# Number of episodes/games to play
episodes = 100000

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
# player = wimblepong.SimpleAi(env, player_id)
# player = Agent(3, 32, 128, 2, "vae_19.pkl", "agent_2999.pkl")
player = Agent(3, 32, 128, 2, "vae_19.pkl")
actor_optimizer = torch.optim.Adam(player.policy.parameters(), lr=1e-3)
critic_optimizer = torch.optim.Adam(player.critic.parameters(), lr=1e-5)

for param in player.cnn_vae.parameters():
    param.requires_grad = False

# Housekeeping

train(env, player, episodes, actor_optimizer, critic_optimizer)

# for i in range(0, episodes):


