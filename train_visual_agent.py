"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import statistics
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

def compute_keep_playing_reward(length):
    # print(max(0, (length-30)/30))
    return max(0, (length-30)/30)

def discounted_returns(rewards, gamma=0.9):
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

def discount_reward(r, gamma=0.9):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, len(r))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def td_discount_reward(r, values, gamma=0.9):
    discounted_r = np.zeros_like(r)
    for t in reversed(range(0, len(r))):
        if t == len(r) - 1:
            discounted_r[t] = r[t]
        else:
            discounted_r[t] = values[t+1].detach().cpu().numpy().squeeze() * gamma + r[t]
    return discounted_r

def roll_out(agent, env, device):
    state = env.reset()
    states = []
    actions = []
    rewards = []
    values = []
    dones = []
    probs = []
    entropys = []
    wins = 0
    is_done = False

    while not is_done:
        if not args.headless:
            env.render()
        states.append(state)
        action, value, policy_prob = agent.choose_action(torch.tensor(state.copy(), dtype=torch.float32).unsqueeze(0).permute(0, 3, 1, 2).to(device))
        next_state, reward, done, info = env.step(action+1)
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
        entropys.append(policy_prob.entropy())
        state = next_state
        if done:
            if reward == 10:
                wins += 1
            is_done = True
            env.reset()
            break

    return states, actions, rewards, probs, values, entropys, wins

def compute_loss(rewards, values, probs, device):
    discounted_rewards = discounted_returns(rewards)
    # discounted_rewards = td_discount_reward(rewards, values)
    value_loss = F.mse_loss(torch.concat(values), torch.tensor(discounted_rewards, dtype=torch.float32).to(device), reduction="mean")
    advantages = torch.tensor(discounted_rewards, dtype=torch.float32).to(device) - torch.concat(values)
    actor_loss = - torch.sum(torch.concat(probs) * advantages.detach())
    return actor_loss, value_loss

def train(env, agent, num_step, optimizer, device):
    # init a task generator for data fetching
    wins1 = 0
    running_length = 0.0
    running_reward = -10.0

    for step in range(num_step):
        if step % 5 == 4:
            env.switch_sides()
        if step % 1000 == 999:
            torch.save(agent.state_dict(), "agent_{}.pkl".format(step))
        states, actions, rewards, probs, values, entropys, wins = roll_out(agent, env, device)
        wins1 += wins
        running_length = 0.05 * len(states) + 0.95 * running_length
        running_reward = 0.05 * rewards[-1] + 0.95 * running_reward
        actor_loss, value_loss = compute_loss(rewards, values, probs, device)
        loss = actor_loss + value_loss - torch.cat(entropys).sum() * 0.01

        # train network
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(agent.parameters(), 0.1)
        optimizer.step()

        to_print = " Episode {} Length {} Mean Length {:.3f} Mean Reward: {:.3f} Actor Loss: {:.6f} Value Loss: {:.3f} Wins: {} Broken WR: {}".format(step, len(states), running_length, running_reward, actor_loss, value_loss, wins, wins1/(step+1))
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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
player = Agent(3, 32, 128, 2, "vae_19.pkl").to(device)
optimizer = torch.optim.Adam(player.parameters(), lr=1e-5)

# for param in player.cnn_vae.parameters():
#     param.requires_grad = False

# Housekeeping

train(env, player, episodes, optimizer, device)

# for i in range(0, episodes):


