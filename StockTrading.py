import gym
import gym_anytrading
from gym_anytrading.envs.stocks_env import StocksEnv
from gym_anytrading.envs.trading_env import Positions
from gym_anytrading.datasets import STOCKS_GOOGL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def flatten_state(s):
    if isinstance(s, dict):
        flattened = []
        for key in sorted(s.keys()):
            flattened.append(flatten_state(s[key]))
        return np.concatenate(flattened)
    elif isinstance(s, Positions):
        if hasattr(s, 'buy_positions') and hasattr(s, 'sell_positions'):
            indicator = 1.0 if (len(s.buy_positions) > 0 or len(s.sell_positions) > 0) else 0.0
            return np.array([indicator])
        else:
            try:
                return np.array([1.0]) if bool(s) else np.array([0.0])
            except Exception:
                return np.array([0.0])
    elif isinstance(s, np.ndarray):
        return s.flatten()
    elif isinstance(s, (list, tuple)):
        flattened = [flatten_state(item) for item in s]
        return np.concatenate(flattened)
    else:
        return np.array(s).flatten()

def process_state_with_padding(state, target_dim):
    flat = flatten_state(state).astype(np.float32)
    current_dim = flat.shape[0]
    if current_dim < target_dim:
        pad_width = target_dim - current_dim
        flat = np.concatenate([flat, np.zeros(pad_width, dtype=np.float32)])
    elif current_dim > target_dim:
        flat = flat[:target_dim]
    return flat

window_size = 10
frame_bound = (window_size, len(STOCKS_GOOGL))
env = StocksEnv(df=STOCKS_GOOGL, window_size=window_size, frame_bound=frame_bound)

initial_state = flatten_state(env.reset()).astype(np.float32)
target_dim = initial_state.shape[0]

def process_state(state):
    return process_state_with_padding(state, target_dim)

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_dim=64):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.critic = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        logits = self.actor(x)
        value = self.critic(x)
        return logits, value

    def act(self, state):
        state_tensor = torch.FloatTensor(process_state(state)).to(device)
        logits, value = self.forward(state_tensor)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action).item(), value.item()

    def evaluate(self, states, actions):
        logits, values = self.forward(states)
        probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action_logprobs = dist.log_prob(actions)
        entropy = dist.entropy()
        return action_logprobs, torch.squeeze(values), entropy

input_dim = target_dim
action_dim = env.action_space.n

model = ActorCritic(input_dim, action_dim).to(device)

learning_rate = 3e-4
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
clip_epsilon = 0.2
ppo_epochs = 4
batch_size = 64
gamma = 0.99
gae_lambda = 0.95

def collect_trajectories(env, model, timesteps_per_batch):
    states, actions, rewards, dones, logprobs, values = [], [], [], [], [], []
    state = process_state(env.reset())
    total_steps = 0

    while total_steps < timesteps_per_batch:
        action, logprob, value = model.act(state)
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = process_state(next_state)
        
        states.append(state)
        actions.append(action)
        rewards.append(reward)
        dones.append(done)
        logprobs.append(logprob)
        values.append(value)
        
        state = next_state
        total_steps += 1
        
        if done or truncated:
            state = process_state(env.reset())
    return states, actions, rewards, dones, logprobs, values

def compute_returns_advantages(rewards, dones, values, gamma=0.99, lam=0.95):
    rewards = np.array(rewards)
    dones = np.array(dones, dtype=np.float32)
    values = np.array(values)
    
    returns = np.zeros_like(rewards)
    advantages = np.zeros_like(rewards)
    
    gae = 0.0
    for t in reversed(range(len(rewards))):
        if t == len(rewards) - 1:
            next_value = 0
            next_non_terminal = 1.0 - dones[t]
        else:
            next_value = values[t+1]
            next_non_terminal = 1.0 - dones[t+1]
        delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
        gae = delta + gamma * lam * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = advantages[t] + values[t]
    return returns, advantages

num_iterations = 1000
timesteps_per_batch = 2048
all_rewards = []

for iteration in tqdm(range(num_iterations)):
    states, actions, rewards, dones, logprobs, values = collect_trajectories(env, model, timesteps_per_batch)
    returns, advantages = compute_returns_advantages(rewards, dones, values, gamma, gae_lambda)
    
    states_tensor = torch.FloatTensor(np.array(states)).to(device)
    actions_tensor = torch.LongTensor(np.array(actions)).to(device)
    old_logprobs_tensor = torch.FloatTensor(np.array(logprobs)).to(device)
    returns_tensor = torch.FloatTensor(np.array(returns)).to(device)
    advantages_tensor = torch.FloatTensor(np.array(advantages)).to(device)
    advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)
    
    dataset_size = states_tensor.size(0)
    
    for epoch in range(ppo_epochs):
        permutation = torch.randperm(dataset_size)
        for i in range(0, dataset_size, batch_size):
            indices = permutation[i: i + batch_size]
            sampled_states = states_tensor[indices]
            sampled_actions = actions_tensor[indices]
            sampled_old_logprobs = old_logprobs_tensor[indices]
            sampled_returns = returns_tensor[indices]
            sampled_advantages = advantages_tensor[indices]
            
            new_logprobs, state_values, entropy = model.evaluate(sampled_states, sampled_actions)
            ratios = torch.exp(new_logprobs - sampled_old_logprobs)
            surr1 = ratios * sampled_advantages
            surr2 = torch.clamp(ratios, 1 - clip_epsilon, 1 + clip_epsilon) * sampled_advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(state_values, sampled_returns)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    if iteration % 10 == 0:
        total_reward = 0
        num_eval_episodes = 5
        for _ in range(num_eval_episodes):
            state = process_state(env.reset())
            done = False
            truncated = False
            episode_reward = 0
            while not (done or truncated):
                action, _, _ = model.act(state)
                state, reward, done, truncated, _ = env.step(action)
                state = process_state(state)
                episode_reward += reward
            total_reward += episode_reward
        avg_reward = total_reward / num_eval_episodes
        print(f"Iteration {iteration}, Average Evaluation Reward: {avg_reward:.2f}")

state = process_state(env.reset())
done = False
truncated = False
while not (done or truncated):
    action, _, _ = model.act(state)
    state, reward, done, truncated, _ = env.step(action)
    state = process_state(state)
    env.render()
plt.show()
