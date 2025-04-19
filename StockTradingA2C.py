import gym
import gym_anytrading
from gym_anytrading.envs.stocks_env import StocksEnv
from gym_anytrading.envs.trading_env import Positions
from gym_anytrading.datasets import STOCKS_GOOGL
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(s):
    if isinstance(s, dict):
        return np.concatenate([flatten_state(s[k]) for k in sorted(s.keys())])
    if isinstance(s, Positions):
        if hasattr(s, 'buy_positions') and hasattr(s, 'sell_positions'):
            return np.array([1.0]) if (len(s.buy_positions) or len(s.sell_positions)) else np.array([0.0])
        return np.array([1.0]) if bool(s) else np.array([0.0])
    if isinstance(s, (list, tuple)):
        return np.concatenate([flatten_state(x) for x in s])
    if isinstance(s, np.ndarray):
        return s.flatten()
    return np.array(s).flatten()

def process_state_with_padding(state, target_dim):
    flat = flatten_state(state).astype(np.float32)
    if flat.size < target_dim:
        return np.pad(flat, (0, target_dim - flat.size), 'constant')
    else:
        return flat[:target_dim]

window_size = 10
frame_bound = (window_size, len(STOCKS_GOOGL)-1)
env = StocksEnv(df=STOCKS_GOOGL, window_size=window_size, frame_bound=frame_bound)

initial = flatten_state(env.reset()).astype(np.float32)
input_dim = initial.size
action_dim = env.action_space.n

def process_state(obs):
    return process_state_with_padding(obs, input_dim)

class ActorCritic(nn.Module):
    def __init__(self, in_dim, n_actions, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.actor = nn.Linear(hidden, n_actions)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.actor(x), self.critic(x)
    
num_episodes    = 1000
gamma           = 0.99
lr              = 3e-4
value_coef      = 0.5
entropy_coef    = 0.01

model = ActorCritic(input_dim, action_dim).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
all_rewards = []

for ep in tqdm(range(num_episodes)):
    obs = env.reset()
    state = process_state(obs)
    done = False
    ep_reward = 0

    log_probs = []
    values    = []
    rewards   = []
    entropies = []
    while not done:
        st = torch.FloatTensor(state).to(device)
        logits, value = model(st)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy  = dist.entropy()

        obs, reward, done, truncated, _ = env.step(action.item())
        done_bool = done or truncated
        next_state = process_state(obs)

        log_probs.append(log_prob)
        values.append(value.squeeze())
        entropies.append(entropy)
        rewards.append(reward)

        state = next_state
        ep_reward += reward

        if done_bool:
            break

    all_rewards.append(ep_reward)

    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.FloatTensor(returns).to(device)
    values  = torch.stack(values)
    log_probs = torch.stack(log_probs)
    entropies = torch.stack(entropies)

    advantages = returns - values
    actor_loss  = -(log_probs * advantages.detach()).mean() - entropy_coef * entropies.mean()
    critic_loss = advantages.pow(2).mean()
    loss = actor_loss + value_coef * critic_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if ep % 25 == 0:
        print(f"Episode {ep:4d}  Reward {ep_reward:.2f}")

plt.figure(figsize=(10,4))
plt.plot(all_rewards)
plt.title("A2C Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

obs = env.reset()
state = process_state(obs)
done = False
truncated = False

while not (done or truncated):
    st = torch.FloatTensor(state).to(device)
    logits, _ = model(st)
    action = torch.argmax(logits).item()
    obs, reward, done, truncated, _ = env.step(action)
    state = process_state(obs)
    env.render()

plt.show()
