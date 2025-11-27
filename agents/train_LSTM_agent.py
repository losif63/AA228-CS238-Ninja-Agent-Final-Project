# Author: Fixed by ChatGPT (for Juhyun Jung)
# Fully corrected Q-LSTM Q-learning code

from src.model import Q, Q_LSTM
from src.env import GameEnv
import pygame
import argparse
import random
import torch


def select_action(q_net, seq_input, hidden, epsilon):
    q_values, hidden = q_net(seq_input, hidden)
    if random.random() < epsilon:
        action = random.randint(0, 4)
    else:
        with torch.no_grad():
            action = torch.argmax(q_values).item()
    return q_values, action, hidden


def main(args):

    pygame.init()

    env = GameEnv()

    q_net = Q_LSTM()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=5e-4, weight_decay=1e-5)
    gamma = 0.90
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.999

    num_episodes = 1000
    max_steps_per_episode = 3600

    ## new
    seq_len_max = 10

    print("\n=== TRAINING Q-LSTM ===")
    print(f"Training for {num_episodes} episodes...\n")

    for episode in range(num_episodes):

        env.reset()
        total_reard = 0.0
        step = 0 
        done = False 
        obs = env.get_obs()  

        obs_seq = [obs.clone().detach()]

        #sequence + recurrent state init
        obs_seq = [obs.clone().detach()]
        hidden = None   # reset hidden 

        while not done and step < max_steps_per_episode:

            # ============================================
            # 1) DETACH hidden at start of each step
            #    (Otherwise gradient graph accumulates forever)
            # ============================================
            if hidden is not None:
                hidden = (hidden[0].detach(), hidden[1].detach())

            # ============================================
            # 2) Prepare sequence input
            # ============================================
            if len(obs_seq) > seq_len_max:
                obs_seq.pop(0)

            seq_input = torch.stack(obs_seq)   # [L, 82]

            # ============================================
            # 3) Select action
            # ============================================
            optimizer.zero_grad()
            q_values, action, hidden = select_action(q_net, seq_input, hidden, epsilon)

            q_sa = q_values[action]

            # ============================================
            # 4) Step in environment
            # ============================================
            next_obs, reward, done, info = env.step(action)
            total_reward += reward
            step += 1

            # append next_obs (clone/detach mandatory)
            obs_seq.append(next_obs.clone().detach())

            if args.render:
                env.render(view=True)

            # ============================================
            # 5) Compute target (IMPORTANT FIX!)
            #    -- next_seq_input uses obs_seq AFTER append
            #    -- hidden=None (never reuse recurrent state)
            # ============================================
            with torch.no_grad():
                if done:
                    target = reward
                else:
                    next_seq_input = torch.stack(obs_seq[-seq_len_max:])
                    next_q, _ = q_net(next_seq_input, None)  # IMPORTANT FIX!!
                    target = reward + gamma * torch.max(next_q)

            # ============================================
            # 6) Backpropagate
            # ============================================
            loss = (target - q_sa) ** 2
            loss.backward()
            optimizer.step()

        # ============================================
        # 7) Epsilon decay
        # ============================================
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode+1}/{num_episodes} | "
              f"Reward={total_reward:.2f} | Steps={step} | Epsilon={epsilon:.3f}")

    print("\nTraining finished.")
    env.close()
    torch.save(q_net.state_dict(), "q_network_LSTM_version.pt")
    print("Model saved to q_network_LSTM_version.pt")
    print("==============================================")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", "-r", action="store_true", default=False)
    args = parser.parse_args()
    main(args)
