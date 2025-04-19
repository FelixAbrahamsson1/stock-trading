import gym
import gym_anytrading
from gym_anytrading.envs.stocks_env import StocksEnv
from gym_anytrading.envs.trading_env import Positions
from gym_anytrading.datasets import STOCKS_GOOGL
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
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
frame_bound = (window_size, len(STOCKS_GOOGL) - 1)
env = StocksEnv(df=STOCKS_GOOGL, window_size=window_size, frame_bound=frame_bound)

initial_obs = flatten_state(env.reset()).astype(np.float32)
target_dim = initial_obs.shape[0]

def process_state(state):
    return process_state_with_padding(state, target_dim)

class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super(QNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return (np.array(state),
                np.array(action),
                np.array(reward, dtype=np.float32),
                np.array(next_state),
                np.array(done, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


num_episodes = 1000
max_timesteps = 1000
batch_size = 64
gamma = 0.99
learning_rate = 1e-3
target_update_interval = 10
buffer_capacity = 50000
epsilon_start = 1.0
epsilon_final = 0.1
epsilon_decay = 300

replay_buffer = ReplayBuffer(buffer_capacity)
input_dim = target_dim
output_dim = env.action_space.n

policy_net = QNetwork(input_dim, output_dim).to(device)
target_net = QNetwork(input_dim, output_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

def epsilon_by_frame(frame_idx):
    return epsilon_final + (epsilon_start - epsilon_final) * np.exp(-frame_idx / epsilon_decay)

def dqn_update():
    if len(replay_buffer) < batch_size:
        return 0
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # Convert lists to tensors using np.array for efficiency.
    states = torch.FloatTensor(np.array(states)).to(device)
    actions = torch.LongTensor(np.array(actions)).unsqueeze(1).to(device)
    rewards = torch.FloatTensor(np.array(rewards)).to(device)
    next_states = torch.FloatTensor(np.array(next_states)).to(device)
    dones = torch.FloatTensor(np.array(dones)).to(device)
    
    q_values = policy_net(states).gather(1, actions).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + gamma * next_q_values * (1 - dones)
    
    loss = nn.MSELoss()(q_values, expected_q_values.detach())
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

all_rewards = []
losses = []
frame_idx = 0

for episode in tqdm(range(num_episodes)):
    state = process_state(env.reset())
    episode_reward = 0
    done = False
    timestep = 0
    while not done and timestep < max_timesteps:
        epsilon = epsilon_by_frame(frame_idx)
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            action = q_values.argmax().item()
        
        next_state, reward, done, truncated, _ = env.step(action)
        next_state = process_state(next_state)
        done_bool = done or truncated
        
        replay_buffer.push(state, action, reward, next_state, done_bool)
        state = next_state
        episode_reward += reward
        frame_idx += 1
        timestep += 1
        
        loss_val = dqn_update()
        if loss_val != 0:
            losses.append(loss_val)
        
        if done_bool:
            break

    all_rewards.append(episode_reward)
    
    if episode % target_update_interval == 0:
        target_net.load_state_dict(policy_net.state_dict())
    
    if episode % 10 == 0:
        avg_reward = np.mean(all_rewards[-10:])
        avg_loss = np.mean(losses[-10:]) if losses else 0
        print(f"Episode {episode}, Avg Reward: {avg_reward:.2f}, Epsilon: {epsilon:.2f}, Loss: {avg_loss:.4f}")

plt.figure(figsize=(10,4))
plt.plot(all_rewards)
plt.title("DQN Training Rewards")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.show()

num_eval = 10
eval_rewards = []
for _ in range(num_eval):
    state = process_state(env.reset())
    episode_reward = 0
    done = False
    while not done:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        q_values = policy_net(state_tensor)
        action = q_values.argmax().item()
        next_state, reward, done, truncated, _ = env.step(action)
        state = process_state(next_state)
        episode_reward += reward
        if done or truncated:
            break
    eval_rewards.append(episode_reward)
    
print("Evaluation Average Reward:", np.mean(eval_rewards))

state = process_state(env.reset())
done = False
truncated = False
while not (done or truncated):
    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
    q_values = policy_net(state_tensor)
    action = q_values.argmax().item()
    state, reward, done, truncated, _ = env.step(action)
    state = process_state(state)
    env.render()
plt.show()
