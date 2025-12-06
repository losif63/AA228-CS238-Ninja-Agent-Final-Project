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


class SequenceReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.episodes = deque()
        self.current_episode = None
        self.size = 0

    def start_episode(self):
        self.current_episode = []
        self.episodes.append(self.current_episode)
        self._trim()

    def _trim(self):
        while self.size > self.capacity and self.episodes:
            removed = self.episodes.popleft()
            self.size -= len(removed)

    def push(self, state, action, reward, next_state, done):
        if self.current_episode is None:
            self.start_episode()
        transition = (
            state.detach().cpu().clone(),
            action,
            reward,
            next_state.detach().cpu().clone(),
            done,
        )
        self.current_episode.append(transition)
        self.size += 1
        self._trim()
        if done:
            self.current_episode = None

    def sample(self, batch_size, sequence_length):
        eligible = [ep for ep in self.episodes if len(ep) >= sequence_length]
        if not eligible:
            raise ValueError("Not enough sequences")
        batch = []
        for _ in range(batch_size):
            episode = random.choice(eligible)
            start = random.randint(0, len(episode) - sequence_length)
            seq = episode[start : start + sequence_length]
            batch.append(seq)
        states = torch.stack([torch.stack([step[0] for step in seq]) for seq in batch])
        actions = torch.tensor([[step[1] for step in seq] for seq in batch], dtype=torch.long)
        rewards = torch.tensor([[step[2] for step in seq] for seq in batch], dtype=torch.float32)
        next_states = torch.stack([torch.stack([step[3] for step in seq]) for seq in batch])
        dones = torch.tensor([[step[4] for step in seq] for seq in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class RecurrentQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes, rnn_hidden, rnn_layers, dropout):
        super().__init__()
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            last_dim = size
        self.feature = nn.Sequential(*layers) if layers else nn.Identity()
        self.rnn = nn.GRU(last_dim, rnn_hidden, num_layers=rnn_layers, batch_first=True, dropout=dropout if rnn_layers > 1 else 0.0)
        self.value_head = nn.Linear(rnn_hidden, 1)
        self.adv_head = nn.Linear(rnn_hidden, action_dim)

    def forward(self, x, hidden=None):
        batch, seq_len, dim = x.shape
        x = x.view(batch * seq_len, dim)
        x = self.feature(x)
        x = x.view(batch, seq_len, -1)
        out, hidden = self.rnn(x, hidden)
        value = self.value_head(out)
        advantage = self.adv_head(out)
        q_values = value + advantage - advantage.mean(dim=2, keepdim=True)
        return q_values, hidden


class DRQNAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        hidden_sizes,
        rnn_hidden,
        rnn_layers,
        dropout,
        lr,
        gamma,
        batch_size,
        buffer_size,
        target_update_interval,
        device,
        sequence_length,
        burn_in,
        max_grad_norm,
        tau,
    ):
        self.device = device
        self.action_dim = action_dim
        self.train_length = max(1, sequence_length)
        self.burn_in = max(0, burn_in)
        self.sample_length = self.train_length + self.burn_in
        self.q_net = RecurrentQNetwork(state_dim, action_dim, hidden_sizes, rnn_hidden, rnn_layers, dropout).to(device)
        self.target_net = RecurrentQNetwork(state_dim, action_dim, hidden_sizes, rnn_hidden, rnn_layers, dropout).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = SequenceReplayBuffer(buffer_size)
        self.target_update_interval = target_update_interval
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.update_counter = 0
        self.hidden_state = None

    def reset_hidden(self):
        self.hidden_state = None

    def select_action(self, state, epsilon):
        state_tensor = state.to(self.device).unsqueeze(0).unsqueeze(0)
        q_values, hidden = self.q_net(state_tensor, self.hidden_state)
        if hidden is not None:
            self.hidden_state = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
        else:
            self.hidden_state = None
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        return int(torch.argmax(q_values[:, -1, :], dim=1).item())

    def store(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def update(self):
        if len(self.replay_buffer) < self.sample_length:
            return None
        try:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size, self.sample_length)
        except ValueError:
            return None
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)
        q_values, _ = self.q_net(states)
        next_eval, _ = self.q_net(next_states)
        next_actions = next_eval.argmax(dim=2, keepdim=True)
        next_target, _ = self.target_net(next_states)
        next_q = next_target.gather(2, next_actions).squeeze(-1)
        if self.burn_in > 0:
            q_values = q_values[:, self.burn_in :, :]
            next_q = next_q[:, self.burn_in :]
            actions = actions[:, self.burn_in :]
            rewards = rewards[:, self.burn_in :]
            dones = dones[:, self.burn_in :]
        q_taken = q_values.gather(2, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            targets = torch.zeros_like(q_taken)
            future = next_q[:, -1]
            for t in range(q_taken.size(1) - 1, -1, -1):
                future = rewards[:, t] + self.gamma * (1 - dones[:, t]) * future
                targets[:, t] = future
        loss = F.smooth_l1_loss(q_taken, targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.update_counter += 1
        if self.tau < 1.0:
            for target_param, param in zip(self.target_net.parameters(), self.q_net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
        elif self.update_counter % self.target_update_interval == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()


def parse_hidden_sizes(values):
    if not values:
        return [256, 256]
    return [int(v) for v in values.split(",")]


def shape_reward(reward_value, next_state, args):
    reward_value += args.reward_alive
    dx = next_state[0].item() - 0.5
    dy = next_state[1].item() - 0.5
    dist = math.sqrt(dx * dx + dy * dy)
    if args.center_penalty > 0:
        reward_value -= args.center_penalty * dist
    if args.edge_penalty > 0 and args.edge_threshold > 0:
        min_border = min(next_state[0].item(), 1 - next_state[0].item(), next_state[1].item(), 1 - next_state[1].item())
        if min_border < args.edge_threshold:
            reward_value -= args.edge_penalty * (args.edge_threshold - min_border) / args.edge_threshold
    return reward_value


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
    agent = DRQNAgent(
        state_dim,
        action_dim,
        hidden_sizes,
        args.rnn_hidden,
        args.rnn_layers,
        args.dropout,
        args.lr,
        args.gamma,
        args.batch_size,
        args.buffer_size,
        args.target_update,
        device,
        args.sequence_length,
        args.burn_in,
        args.max_grad_norm,
        args.tau,
    )
    global_step = 0
    reward_history = deque(maxlen=args.log_interval)
    for episode in range(args.episodes):
        env.reset()
        agent.reset_hidden()
        agent.replay_buffer.start_episode()
        state = env.get_obs()
        episode_reward = 0.0
        for step in range(args.max_steps):
            epsilon = args.eps_end + (args.eps_start - args.eps_end) * math.exp(-global_step / args.eps_decay)
            action = agent.select_action(state, epsilon)
            next_state, reward, done = env.step(action)
            shaped_reward = shape_reward(float(reward), next_state, args)
            agent.store(state, action, shaped_reward, next_state, float(done))
            if global_step >= args.learning_starts and global_step % args.train_freq == 0:
                for _ in range(args.gradient_steps):
                    agent.update()
            state = next_state
            episode_reward += shaped_reward
            global_step += 1
            if done:
                agent.reset_hidden()
                break
        reward_history.append(episode_reward)
        if (episode + 1) % args.log_interval == 0:
            avg_reward = sum(reward_history) / len(reward_history)
            print(f"Episode {episode + 1} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f}")
    torch.save(agent.q_net.state_dict(), args.output)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1500)
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--buffer-size", type=int, default=300000)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.05)
    parser.add_argument("--eps-decay", type=float, default=180000.0)
    parser.add_argument("--target-update", type=int, default=1500)
    parser.add_argument("--hidden-sizes", type=str, default="256,256")
    parser.add_argument("--rnn-hidden", type=int, default=384)
    parser.add_argument("--rnn-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output", type=str, default="agents/dqn_rnn_policy.pt")
    parser.add_argument("--learning-starts", type=int, default=8000)
    parser.add_argument("--train-freq", type=int, default=4)
    parser.add_argument("--gradient-steps", type=int, default=3)
    parser.add_argument("--reward-alive", type=float, default=0.02)
    parser.add_argument("--center-penalty", type=float, default=0.5)
    parser.add_argument("--edge-penalty", type=float, default=1.0)
    parser.add_argument("--edge-threshold", type=float, default=0.2)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.01)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
