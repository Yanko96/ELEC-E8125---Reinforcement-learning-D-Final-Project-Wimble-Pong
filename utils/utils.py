import torch
import numpy as np
import torch.nn.functional as F

def compute_keep_playing_reward(length):
    return max(0, int(length-30)/2)

def discounted_returns(rewards, gamma=0.9):
    ## Init R
    R = 0
    returns = list()
    for reward in reversed(rewards):
        R = reward + gamma * R
        returns.insert(0, R)

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

def roll_out(agent, env, device, headless=True, keep_playing_reward=False):
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
    TRANSFORM_IMG = transforms.Compose([
        # transforms.ToPILImage(),
        # transforms.Resize(128),
        transforms.ToTensor()
        ])

    while not is_done:
        if not args.headless:
            env.render()
        states.append(state)
        action, value, policy_prob = agent.choose_action(TRANSFORM_IMG(state.copy()).unsqueeze(0).to(device))
        next_state, reward, done, info = env.step(action+1)
        if not done and keep_playing_reward:
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
