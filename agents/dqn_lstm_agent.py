import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from collections import deque
import math
import os
import random
import argparse
import sys


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GameEnv
import src.config as cfg


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
        eligible = [episode for episode in self.episodes if len(episode) >= sequence_length]
        if not eligible:
            raise ValueError("Not enough sequences")
        sequences = []
        for _ in range(batch_size):
            episode = random.choice(eligible)
            start = random.randint(0, len(episode) - sequence_length)
            seq = episode[start : start + sequence_length]
            sequences.append(seq)
        states = torch.stack([torch.stack([step[0] for step in seq]) for seq in sequences])
        actions = torch.tensor([[step[1] for step in seq] for seq in sequences], dtype=torch.long)
        rewards = torch.tensor([[step[2] for step in seq] for seq in sequences], dtype=torch.float32)
        next_states = torch.stack([torch.stack([step[3] for step in seq]) for seq in sequences])
        dones = torch.tensor([[step[4] for step in seq] for seq in sequences], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return self.size


class RecurrentQNetwork(nn.Module):
    def __init__(self, input_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout):
        super().__init__()
        layers = []
        last_dim = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_dim, size))
            layers.append(nn.LayerNorm(size))
            layers.append(nn.SiLU())
            if mlp_dropout > 0:
                layers.append(nn.Dropout(mlp_dropout))
            last_dim = size
        self.feature = nn.Sequential(*layers) if layers else nn.Identity()
        lstm_dropout_val = lstm_dropout if lstm_layers > 1 else 0.0
        self.lstm = nn.LSTM(last_dim, lstm_hidden, num_layers=lstm_layers, dropout=lstm_dropout_val, batch_first=True)
        self.value_head = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.SiLU(), nn.Linear(lstm_hidden, 1))
        self.adv_head = nn.Sequential(nn.Linear(lstm_hidden, lstm_hidden), nn.SiLU(), nn.Linear(lstm_hidden, action_dim))

    def forward(self, x, hidden=None):
        batch, seq_len, dim = x.shape
        x = x.view(batch * seq_len, dim)
        x = self.feature(x)
        x = x.view(batch, seq_len, -1)
        out, hidden = self.lstm(x, hidden)
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
        lstm_hidden,
        lstm_layers,
        mlp_dropout,
        lstm_dropout,
        lr,
        gamma,
        batch_size,
        buffer_size,
        target_update_interval,
        device,
        sequence_length,
        burn_in_steps,
        max_grad_norm,
        tau,
    ):
        self.device = device
        self.action_dim = action_dim
        self.train_length = max(1, sequence_length)
        self.burn_in = max(0, burn_in_steps)
        self.sample_length = self.train_length + self.burn_in
        self.q_net = RecurrentQNetwork(state_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout).to(device)
        self.target_net = RecurrentQNetwork(state_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.replay_buffer = SequenceReplayBuffer(buffer_size)
        self.target_update_interval = target_update_interval
        self.update_counter = 0
        self.max_grad_norm = max_grad_norm
        self.tau = tau
        self.hidden_state = None

    def reset_hidden(self):
        self.hidden_state = None

    def select_action(self, state, epsilon):
        state_tensor = state.to(self.device).unsqueeze(0).unsqueeze(0)
        q_values, hidden = self.q_net(state_tensor, self.hidden_state)
        if hidden is not None:
            self.hidden_state = tuple(h.detach() for h in hidden)
        else:
            self.hidden_state = None
        if random.random() < epsilon:
            action = random.randrange(self.action_dim)
        else:
            action = int(torch.argmax(q_values, dim=2).item())
        if torch.any(torch.isnan(q_values)):
            action = random.randrange(self.action_dim)
        return action

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
        return [256]
    return [int(v) for v in values.split(",")]


def shape_reward(reward_value, next_state, args):
    reward_value += args.reward_alive
    if args.reward_center_penalty > 0:
        dx = next_state[0].item() - 0.5
        dy = next_state[1].item() - 0.5
        reward_value -= args.reward_center_penalty * math.sqrt(dx * dx + dy * dy)
    return reward_value


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    cfg.VISION_RADIUS = args.vision_radius
    env = GameEnv()
    env.reset()
    state = env.get_obs()
    state_dim = state.shape[0]
    action_dim = 5
    device = torch.device("mps" if torch.cuda.is_available() else "cpu")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    agent = DRQNAgent(
        state_dim,
        action_dim,
        hidden_sizes,
        args.lstm_hidden,
        args.lstm_layers,
        args.mlp_dropout,
        args.lstm_dropout,
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
    base, ext = os.path.splitext(args.output)
    if not ext:
        ext = ".pt"
    save_path = f"{base}_{args.vision_radius}{ext}"
    metadata = {
        "hidden_sizes": hidden_sizes,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "mlp_dropout": args.mlp_dropout,
        "lstm_dropout": args.lstm_dropout,
        "vision_radius": args.vision_radius,
        "sequence_length": args.sequence_length,
        "burn_in": args.burn_in,
    }
    torch.save({"model_state_dict": agent.q_net.state_dict(), "config": metadata}, save_path)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--max-steps", type=int, default=3500)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--buffer-size", type=int, default=250000)
    parser.add_argument("--gamma", type=float, default=0.992)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--eps-start", type=float, default=1.0)
    parser.add_argument("--eps-end", type=float, default=0.02)
    parser.add_argument("--eps-decay", type=float, default=150000.0)
    parser.add_argument("--target-update", type=int, default=1500)
    parser.add_argument("--hidden-sizes", type=str, default="384,256")
    parser.add_argument("--lstm-hidden", type=int, default=256)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--mlp-dropout", type=float, default=0.1)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--burn-in", type=int, default=8)
    parser.add_argument("--vision-radius", type=int, default=cfg.VISION_RADIUS)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--output", type=str, default="agents/dqn_lstm_policy.pt")
    parser.add_argument("--learning-starts", type=int, default=6000)
    parser.add_argument("--train-freq", type=int, default=6)
    parser.add_argument("--gradient-steps", type=int, default=2)
    parser.add_argument("--reward-alive", type=float, default=0.01)
    parser.add_argument("--reward-center-penalty", type=float, default=0.4)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
