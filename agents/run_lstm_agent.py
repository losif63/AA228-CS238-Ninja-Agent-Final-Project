import argparse
import os
import sys
import torch
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.env import GameEnv
from agents.dqn_lstm_agent import RecurrentQNetwork, parse_hidden_sizes


def load_checkpoint(model_path, device):
    payload = torch.load(model_path, map_location=device)
    if isinstance(payload, dict) and "model_state_dict" in payload:
        return payload["model_state_dict"], payload.get("config", {})
    return payload, {}


def build_network(state_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout, device):
    return RecurrentQNetwork(state_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout).to(device)


def run_episode(env, net, device, render, max_steps):
    obs = env.get_obs().to(device)
    done = False
    steps = 0
    total_reward = 0.0
    hidden = None
    while not done and steps < max_steps:
        obs_batch = obs.unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            q_values, hidden = net(obs_batch, hidden)
            if hidden is not None:
                hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
            action = int(torch.argmax(q_values[:, -1, :], dim=1).item())
        next_obs, reward, done = env.step(action)
        total_reward += float(reward)
        steps += 1
        if render:
            env.render(view=True, step=steps)
        obs = next_obs.to(device)
        if done:
            hidden = None
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="agents/dqn_lstm_policy.pt")
    parser.add_argument("--hidden-sizes", type=str, default=None)
    parser.add_argument("--lstm-hidden", type=int, default=None)
    parser.add_argument("--lstm-layers", type=int, default=None)
    parser.add_argument("--mlp-dropout", type=float, default=None)
    parser.add_argument("--lstm-dropout", type=float, default=None)
    parser.add_argument("--vision-radius", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--render", "-r", action="store_true", default=False)
    args = parser.parse_args()
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    state_dict, metadata = load_checkpoint(args.model, device)
    hidden_sizes = None
    if args.hidden_sizes:
        hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    elif metadata.get("hidden_sizes") is not None:
        hidden_sizes = list(metadata["hidden_sizes"])
    else:
        hidden_sizes = parse_hidden_sizes("256,256")
    lstm_hidden = args.lstm_hidden if args.lstm_hidden is not None else metadata.get("lstm_hidden", 256)
    lstm_layers = args.lstm_layers if args.lstm_layers is not None else metadata.get("lstm_layers", 1)
    mlp_dropout = args.mlp_dropout if args.mlp_dropout is not None else metadata.get("mlp_dropout", 0.1)
    lstm_dropout = args.lstm_dropout if args.lstm_dropout is not None else metadata.get("lstm_dropout", 0.2)
    vision_radius = args.vision_radius if args.vision_radius is not None else metadata.get("vision_radius", cfg.VISION_RADIUS)
    cfg.VISION_RADIUS = vision_radius
    env = GameEnv()
    env.reset()
    state_dim = env.get_obs().shape[0]
    action_dim = 5
    net = build_network(state_dim, action_dim, hidden_sizes, lstm_hidden, lstm_layers, mlp_dropout, lstm_dropout, device)
    net.load_state_dict(state_dict)
    net.eval()
    reward, steps = run_episode(env, net, device, args.render, args.max_steps)
    print(f"Episode finished in {steps} steps | Reward {reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
