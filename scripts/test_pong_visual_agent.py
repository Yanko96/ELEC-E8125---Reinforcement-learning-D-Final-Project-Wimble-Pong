"""
This is an example on how to use the two player Wimblepong environment with one
agent and the SimpleAI
"""
import argparse
import os
import pickle
import warnings
from random import randint

import cv2
import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as T
import wimblepong
from agent.agent import Agent
from PIL import Image, ImageDraw, ImageFont

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode")
parser.add_argument("--housekeeping", action="store_true", help="Plot, player and ball positions and velocities at the end of each episode")
parser.add_argument("--fps", type=int, help="FPS for rendering", default=30)
parser.add_argument("--scale", type=int, help="Scale of the rendered game", default=1)
parser.add_argument("--model_path", type=str, help="Path to model that will be tested", default=None)
parser.add_argument("--video_dir", type=str, help="Dir to store the video", default=None)
args = parser.parse_args()

# Make the environment
env = gym.make("WimblepongVisualSimpleAI-v0")
env.unwrapped.scale = args.scale
env.unwrapped.fps = args.fps

# Number of episodes/games to play
episodes = 10

# Define the player
player_id = 1
# Set up the player here. We used the SimpleAI that does not take actions for now
# player = wimblepong.SimpleAi(env, player_id)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
player = Agent(3, 32, 128, 2, None, args.model_path).to(device)

# Housekeeping
states = []
win1 = 0
ob1 = env.reset()

TRANSFORM_IMG = T.Compose([
    T.ToPILImage(),
    T.Resize(128),
    T.ToTensor()
    ])

fourcc = cv2.VideoWriter_fourcc('I', '4', '2', '0')
if args.video_dir:
    out = cv2.VideoWriter(os.path.join(args.video_dir, "replay.avi"), fourcc, args.fps, (200, 235))
    args.headless = False
    

for i in range(0,episodes):
    done = False
    while not done:
        # action1 is zero because in this example no agent is playing as player 0
        # action1 = 0 #player.get_action()
        action, _, _ = player.choose_action(TRANSFORM_IMG(ob1.copy()).unsqueeze(0).to(device))
        ob1, rew1, done, info = env.step(action+1)

        if args.housekeeping:
            states.append(ob1)
        # Count the wins
        if rew1 == 10:
            win1 += 1
        if not args.headless:
            env.render()
            frame = cv2.cvtColor(np.uint8(env.screen), cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (env.SCREEN_RESOLUTION[1]*env.scale,
                                env.SCREEN_RESOLUTION[0]*env.scale),
                                interpolation=cv2.INTER_NEAREST)
            out.write(frame)
        if done:
            observation = env.reset()
            plt.close()  # Hides game window
            if args.housekeeping:
                plt.plot(states)
                plt.legend(["Player", "Opponent", "Ball X", "Ball Y", "Ball vx", "Ball vy"])
                plt.show()
                states.clear()
            print("episode {} over. Broken WR: {:.3f}".format(i, win1/(i+1)))
            if i % 5 == 4:
                env.switch_sides()
if args.video_dir:
    out.release()   

