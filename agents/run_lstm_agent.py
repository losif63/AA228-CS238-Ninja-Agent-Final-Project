
from src.model import Q_LSTM
import random
from src.env import GameEnv
import pygame
import argparse
import torch

def select_action(q_net, obs, hidden):
    q_values, hidden = q_net(obs, hidden)
    with torch.no_grad():
        action = torch.argmax(q_values).item()
    return action, hidden

def main(args):
    # Initialize pygame first
    pygame.init()
    
    env = GameEnv()
    env.reset()

    q_net = Q_LSTM()
    q_net.load_state_dict(torch.load("q_lstm_network_225.pt"))
    q_net.eval()

    total_reward = 0.0
    done = False
    step = 0
    hidden = None

    obs = env.get_obs().unsqueeze(0).unsqueeze(0)

    while not done:
        action, hidden = select_action(q_net, obs, hidden)
        # Go one step

        next_obs, reward, done = env.step(action)
        total_reward += reward
        step += 1 
        # Render

        if args.render:
            env.render(view=True, step=step)

        obs = next_obs.unsqueeze(0).unsqueeze(0)

    print("Finished.")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False, help='Whether to render the NN agent or not')
    args = parser.parse_args()
    main(args)
