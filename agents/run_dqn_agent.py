import argparse
import os
import sys
import torch
import pygame

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.env import GameEnv
from agents.dqn_agent import QNetwork, parse_hidden_sizes


def load_policy(model_path, hidden_sizes, state_dim, action_dim, device):
    net = QNetwork(state_dim, action_dim, hidden_sizes).to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


def run_episode(env, net, device, render, max_steps):
    obs = env.get_obs().to(device)
    done = False
    total_reward = 0.0
    steps = 0
    while not done and steps < max_steps:
        with torch.no_grad():
            q_values = net(obs.unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        next_obs, reward, done = env.step(action)
        total_reward += float(reward)
        steps += 1
        if render:
            env.render(view=True, step=steps)
        obs = next_obs.to(device)
    return total_reward, steps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="agents/dqn_policy.pt")
    parser.add_argument("--hidden-sizes", type=str, default="512,512,256")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--render", "-r", action="store_true", default=False)
    args = parser.parse_args()
    pygame.init()
    env = GameEnv()
    env.reset()
    state_dim = env.get_obs().shape[0]
    action_dim = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hidden_sizes = parse_hidden_sizes(args.hidden_sizes)
    net = load_policy(args.model, hidden_sizes, state_dim, action_dim, device)
    reward, steps = run_episode(env, net, device, args.render, args.max_steps)
    print(f"Episode finished in {steps} steps | Reward {reward:.2f}")
    env.close()


if __name__ == "__main__":
    main()
