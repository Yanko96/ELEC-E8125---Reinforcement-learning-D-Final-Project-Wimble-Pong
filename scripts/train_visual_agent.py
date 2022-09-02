"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import argparse
import warnings
import os

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import wimblepong
from agent.agent import Agent
from utils.utils import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30000)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--load_model", type=str, help="Load Model Checkpoint", default=None)
parser.add_argument("--save_path", type=str, help="Path to Save Model Checkpoint", default=None)
parser.add_argument("--pretrain", type=str, help="Pretrained Encoder from VAE", default=None)
parser.add_argument("--keep_playing_reward", action="store_true")
args = parser.parse_args()

def train(env, agent, num_step, optimizer, device):
    # init a task generator for data fetching
    wins1 = 0
    running_length = 0.0
    running_reward = -10.0

    for step in range(num_step):
        if step % 5 == 4:
            env.switch_sides()
        if step % 1000 == 999:
            torch.save(agent.state_dict(), os.path.join(args.save_path, "agent_{}.pkl".format(step)))
        states, actions, rewards, probs, values, entropys, wins = roll_out(agent, env, device, args.headless, args.keep_playing_reward)
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

        if step % 1000 == 999:
            if args.pretrain:
                optimizer = torch.optim.Adam(
                    [
                        {"params": player.cnn_vae.encoder.parameters(), "lr": min(1e-12*pow(10, (step//4999)), 1e-5)},
                        {"params": player.policy.parameters()},
                        {"params": player.critic.parameters()},
                    ],
                    lr=1e-5,
                )

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps
ob1 = env.reset()

# Number of episodes/games to play
episodes = 100000

# Define the player
player_id = 1
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
player = Agent(3, 32, 128, 2, "vae_19.pkl").to(device)
if args.pretrain:
    optimizer = torch.optim.Adam(
        [
            {"params": player.policy.parameters()},
            {"params": player.critic.parameters()},
        ],
        lr=1e-5,
    )
else:
    optimizer = torch.optim.Adam(
        [
            {"params": player.cnn_vae.encoder.parameters()},
            {"params": player.policy.parameters()},
            {"params": player.critic.parameters()},
        ],
        lr=1e-5,
    )

train(env, player, episodes, optimizer, device)
