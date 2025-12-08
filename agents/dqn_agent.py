import argparse
import math
import os
import random
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import deque

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GameEnv


class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.position = 0
        self.max_priority = 1.0

    def push(self, state, action, reward, next_state, done):
        transition = (state.detach().cpu().clone(), action, reward, next_state.detach().cpu().clone(), done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.priorities[self.position] = self.max_priority
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size, beta):
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        actual_size = len(self.buffer)
        if actual_size == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:actual_size]
        scaled_priorities = priorities ** self.alpha
        probs = scaled_priorities / scaled_priorities.sum()
        replace = actual_size < batch_size
        indices = np.random.choice(actual_size, batch_size, p=probs, replace=replace)
        samples = [self.buffer[i] for i in indices]
        states = torch.stack([s[0] for s in samples])
        actions = torch.tensor([s[1] for s in samples], dtype=torch.long)
        rewards = torch.tensor([s[2] for s in samples], dtype=torch.float32)
        next_states = torch.stack([s[3] for s in samples])
        dones = torch.tensor([s[4] for s in samples], dtype=torch.float32)
        weights = (actual_size * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        weights = torch.tensor(weights, dtype=torch.float32)
        return states, actions, rewards, next_states, dones, weights, indices

    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            value = float(priority)
            self.priorities[idx] = value
            if value > self.max_priority:
                self.max_priority = value

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            last_dim = size
        self.feature = nn.Sequential(*layers)
        self.value = nn.Linear(last_dim, 1)
        self.advantage = nn.Linear(last_dim, output_dim)

    def forward(self, x):
        features = self.feature(x)
        values = self.value(features)
        advantages = self.advantage(features)
        q_values = values + advantages - advantages.mean(dim=1, keepdim=True)
        return q_values


class DQNAgent:
    def __init__(self, state_dim, action_dim, hidden_sizes, lr, gamma, batch_size, buffer_size, target_update_interval, device, max_grad_norm, tau, per_alpha, per_beta_start, per_beta_frames, priority_eps):
        self.device = device
        self.action_dim = action_dim
        self.q_net = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_net = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, per_alpha)
        self.target_update_interval = target_update_interval
        self.update_counter = 0
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.beta_start = per_beta_start
        self.beta_frames = per_beta_frames
        self.priority_eps = priority_eps

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            state_tensor = state.to(self.device).unsqueeze(0)
            q_values = self.q_net(state_tensor)
            return int(torch.argmax(q_values, dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def beta_by_step(self, step):
        if self.beta_frames <= 0:
            return 1.0
        return min(1.0, self.beta_start + (step / self.beta_frames) * (1.0 - self.beta_start))

    def update(self, step):
        if len(self.replay_buffer) < self.batch_size:
            return None
        beta = self.beta_by_step(step)
        states, actions, rewards, next_states, dones, weights, indices = self.replay_buffer.sample(self.batch_size, beta)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        weights = weights.to(self.device)
        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.q_net(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_net(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.gamma * next_q_values * (1 - dones)
        td_errors = targets - q_values
        loss = (weights * td_errors.pow(2)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        new_priorities = (td_errors.detach().abs() + self.priority_eps).cpu().numpy()
        self.replay_buffer.update_priorities(indices, new_priorities)
        self.update_counter += 1
        if self.tau < 1.0:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        elif self.update_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def parse_hidden_sizes(values):
    if not values:
        return [128, 128]
    return [int(v) for v in values.split(",")]


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    env = GameEnv()
    env.reset()
    state = env.get_obs()
    state_dim = state.shape[0]
    action_dim = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    agent = DQNAgent(
        state_dim,
        action_dim,
        hidden_sizes,
        args.lr,
        args.gamma,
        args.batch_size,
        args.buffer_size,
        args.target_update,
        device,
        args.max_grad_norm,
        args.tau,
        args.per_alpha,
        args.per_beta_start,
        args.per_beta_frames,
        args.priority_eps,
    )
    global_step = 0
    reward_history = deque(maxlen=args.log_interval)
    for episode in range(args.episodes):
        env.reset()
        state = env.get_obs()
        episode_reward = 0.0
        edge_exposure = 0.0
        for step in range(args.max_steps):
            epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-global_step / args.eps_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            exposure_decay = args.edge_exposure_decay * edge_exposure
            edge_exposure = max(0.0, edge_exposure - exposure_decay)
            min_border = min(next_state[0].item(), 1 - next_state[0].item(), next_state[1].item(), 1 - next_state[1].item())
            if min_border < args.edge_threshold:
                edge_exposure = min(args.edge_exposure_cap, edge_exposure + args.edge_exposure_increment * (1 - min_border / args.edge_threshold))
            shaped_reward = shape_reward(float(reward), state, next_state, edge_exposure, args)
            agent.store(state, action, shaped_reward, next_state, float(done))
            if global_step >= args.learning_starts and global_step % args.train_freq == 0:
                for _ in range(args.gradient_steps):
                    agent.update(global_step)
            state = next_state
            episode_reward += shaped_reward
            global_step += 1
            if done:
                break
        reward_history.append(episode_reward)
        if (episode + 1) % args.log_interval == 0:
            avg_reward = sum(reward_history) / len(reward_history)
            print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    torch.save(agent.q_net.state_dict(), args.output)


def shape_reward(reward_value, prev_state, next_state, edge_exposure, args):
    reward_value += args.reward_alive
    dx_next = next_state[0].item() - 0.5
    dy_next = next_state[1].item() - 0.5
    dist_next = math.sqrt(dx_next * dx_next + dy_next * dy_next)
    if args.reward_center_penalty > 0:
        reward_value -= args.reward_center_penalty * dist_next
    dx_prev = prev_state[0].item() - 0.5
    dy_prev = prev_state[1].item() - 0.5
    dist_prev = math.sqrt(dx_prev * dx_prev + dy_prev * dy_prev)
    if args.center_move_bonus != 0:
        reward_value += args.center_move_bonus * (dist_prev - dist_next)
    if args.edge_penalty > 0 and args.edge_threshold > 0:
        x = next_state[0].item()
        y = next_state[1].item()
        min_border = min(x, 1 - x, y, 1 - y)
        if min_border < args.edge_threshold:
            penalty = (args.edge_threshold - min_border) / args.edge_threshold
            penalty = penalty ** args.edge_penalty_power
            reward_value -= args.edge_penalty * penalty
            if args.outward_move_penalty > 0 and dist_next > dist_prev and dist_prev > args.outward_move_threshold:
                reward_value -= args.outward_move_penalty * (dist_next - dist_prev)
    if args.edge_exposure_penalty > 0:
        reward_value -= args.edge_exposure_penalty * edge_exposure
    return reward_value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--buffer-size", type=int, default=100000)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay", type=float, default=100000.0)
    parser.add_argument("--target-update", type=int, default=2000)
    parser.add_argument("--hidden-sizes", type=str, default="512,512,256")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output", type=str, default="agents/dqn_policy.pt")
    parser.add_argument("--learning-starts", type=int, default=5000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--reward-alive", type=float, default=0.01)
    parser.add_argument("--reward-center-penalty", type=float, default=0.5)
    parser.add_argument("--center-move-bonus", type=float, default=0.5)
    parser.add_argument("--edge-threshold", type=float, default=0.15)
    parser.add_argument("--edge-penalty", type=float, default=1.0)
    parser.add_argument("--edge-penalty-power", type=float, default=2.0)
    parser.add_argument("--outward-move-penalty", type=float, default=1.0)
    parser.add_argument("--outward-move-threshold", type=float, default=0.3)
    parser.add_argument("--edge-exposure-penalty", type=float, default=0.2)
    parser.add_argument("--edge-exposure-increment", type=float, default=0.1)
    parser.add_argument("--edge-exposure-decay", type=float, default=0.02)
    parser.add_argument("--edge-exposure-cap", type=float, default=2.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--per-alpha", type=float, default=0.6)
    parser.add_argument("--per-beta-start", type=float, default=0.4)
    parser.add_argument("--per-beta-frames", type=int, default=200000)
    parser.add_argument("--priority-eps", type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
