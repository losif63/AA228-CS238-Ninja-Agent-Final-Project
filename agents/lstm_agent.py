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


def select_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
        self.action_dim = action_dim

    def forward(self, x, hidden=None):
        batch, seq_len, dim = x.shape
        x = x.view(batch * seq_len, dim)
        x = self.feature(x)
        x = x.view(batch, seq_len, -1)
        out, hidden = self.lstm(x, hidden)
        q_values = out[..., : self.action_dim]
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


def agent_center_distance(state):
    x = float(state[0].item())
    y = float(state[1].item())
    dx = x - 0.5
    dy = y - 0.5
    return math.sqrt(dx * dx + dy * dy)


def agent_edge_proximity(state):
    x = float(state[0].item())
    y = float(state[1].item())
    return min(x, 1.0 - x, y, 1.0 - y)


def min_arrow_distance(state):
    if state.shape[0] <= 2:
        return 1.0
    arrow_data = state[2:]
    slot_count = len(arrow_data) // 4
    min_dist = 1.0
    found = False
    for i in range(slot_count):
        dx = float(arrow_data[4 * i].item())
        dy = float(arrow_data[4 * i + 1].item())
        vx = float(arrow_data[4 * i + 2].item())
        vy = float(arrow_data[4 * i + 3].item())
        if abs(dx) < 1e-6 and abs(dy) < 1e-6 and abs(vx) < 1e-6 and abs(vy) < 1e-6:
            continue
        dist = math.sqrt(dx * dx + dy * dy)
        min_dist = min(min_dist, dist)
        found = True
    return min_dist if found else 1.0


def shape_reward(reward_value, prev_state, next_state, args):
    reward_value += args.reward_alive
    dist_prev = agent_center_distance(prev_state)
    dist_next = agent_center_distance(next_state)
    if args.reward_center_penalty > 0:
        reward_value -= args.reward_center_penalty * dist_next
    if args.center_move_bonus != 0:
        reward_value += args.center_move_bonus * (dist_prev - dist_next)
    if args.safe_zone_bonus > 0:
        reward_value += args.safe_zone_bonus * max(0.0, args.safe_zone_threshold - dist_next)
    if args.edge_penalty > 0 and args.edge_threshold > 0:
        proximity = agent_edge_proximity(next_state)
        if proximity < args.edge_threshold:
            penalty = (args.edge_threshold - proximity) / args.edge_threshold
            reward_value -= args.edge_penalty * penalty
    if args.arrow_distance_bonus != 0:
        min_dist = min_arrow_distance(next_state)
        threshold = max(1e-6, args.arrow_distance_threshold)
        if min_dist < threshold:
            penalty = (threshold - min_dist) / threshold
            reward_value -= args.arrow_distance_bonus * penalty
        else:
            reward_value += args.arrow_distance_bonus * min(1.0, min_dist)
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
    device = select_device()
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
            shaped_reward = shape_reward(float(reward), state, next_state, args)
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
        "reward_alive": args.reward_alive,
        "reward_center_penalty": args.reward_center_penalty,
        "center_move_bonus": args.center_move_bonus,
        "safe_zone_bonus": args.safe_zone_bonus,
        "safe_zone_threshold": args.safe_zone_threshold,
        "edge_penalty": args.edge_penalty,
        "edge_threshold": args.edge_threshold,
        "arrow_distance_bonus": args.arrow_distance_bonus,
        "arrow_distance_threshold": args.arrow_distance_threshold,
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
    parser.add_argument("--reward-alive", type=float, default=0.02)
    parser.add_argument("--reward-center-penalty", type=float, default=0.6)
    parser.add_argument("--center-move-bonus", type=float, default=0.5)
    parser.add_argument("--safe-zone-bonus", type=float, default=0.05)
    parser.add_argument("--safe-zone-threshold", type=float, default=0.1)
    parser.add_argument("--edge-penalty", type=float, default=0.8)
    parser.add_argument("--edge-threshold", type=float, default=0.15)
    parser.add_argument("--arrow-distance-bonus", type=float, default=0.3)
    parser.add_argument("--arrow-distance-threshold", type=float, default=0.2)
    parser.add_argument("--tau", type=float, default=0.005)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
