
from src.model import Q_LSTM
import random
from src.env import GameEnv
import pygame
import argparse
import torch
from collections import deque

SEED = 42

class ReplayBuffer():
    def __init__(self):
        self.obs = deque()
        self.actions = deque()
        self.rewards = deque()
        self.next_obs = deque()
        self.max_capacity = 20000
        return

    def __len__(self):
        return len(self.obs)

    def push(self, o, a, r, next_o):
        if len(self.obs) >= self.max_capacity:
            self.obs.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.next_obs.popleft()
        
        self.obs.append(o)
        self.actions.append(a)
        self.rewards.append(r)
        self.next_obs.append(next_o)
        return
    
    def sample(self, batch_size, seq_len=120):

        obs_batch = []
        actions_batch = []
        rewards_batch = []
        next_obs_batch = []

        max_start = len(self.obs) - seq_len-1
        idxs = random.sample(range(max_start), batch_size)
        
        for start_idx in idxs:
            end_idx = start_idx + seq_len

            obs_seq = torch.stack([self.obs[i] for i in range(start_idx, end_idx)])
            actions_seq = torch.tensor([self.actions[i] for i in range(start_idx, end_idx)], dtype=torch.long)
            rewards_seq = torch.tensor([self.rewards[i] for i in range(start_idx, end_idx)], dtype=torch.float32)
            next_obs_seq = torch.stack([self.next_obs[i] for i in range(start_idx, end_idx)])
            
            obs_batch.append(obs_seq)
            actions_batch.append(actions_seq)
            rewards_batch.append(rewards_seq)
            next_obs_batch.append(next_obs_seq)
            
        obs_batch = torch.stack(obs_batch)            # batch, seq_len, 82
        actions_batch = torch.stack(actions_batch)    # batch, seq_len
        rewards_batch = torch.stack(rewards_batch)    # batch, seq_len
        next_obs_batch = torch.stack(next_obs_batch)  # batch, seq_len, 82

        # obs_batch = torch.stack([self.obs[i] for i in idxs])
        # actions_batch = torch.tensor([self.actions[i] for i in idxs], dtype=torch.long)
        # rewards_batch = torch.tensor([self.rewards[i] for i in idxs], dtype=torch.float32)
        # next_obs_batch = torch.stack([self.next_obs[i] for i in idxs])
        return obs_batch, actions_batch, rewards_batch, next_obs_batch
        

def select_action(q_net, obs, epsilon, hidden):
    q_values, hidden = q_net(obs, hidden)
    if random.random() < epsilon:
        return q_values, random.randint(0, 4), hidden
    
    with torch.no_grad():
        return q_values, torch.argmax(q_values).item(), hidden


def main(args):
    pygame.init()
    
    env = GameEnv()
    env.reset()

    q_net = Q_LSTM()
    optimizer = torch.optim.Adam(q_net.parameters(), lr=5e-4, weight_decay=1e-5)
    buffer = ReplayBuffer()
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = 0.9993

    num_episodes = 3000
    max_steps_per_episode = 3600

    warmup_steps = 1000
    batch_size = 16
    
    seq_len = 120 




    print(f"Q-Learning for {num_episodes} episodes...")
    
    for episode in range(num_episodes):
        env.reset()
        total_reward = 0.0
        done = False
        step = 0
        hidden = None


        obs_lst = []
        obs = env.get_obs()#.unsqueeze(0).unsqueeze(0)
        # print(f'Initial obs.shape: {obs.shape}')
        obs_lst.append(obs)


        while not done and step < max_steps_per_episode:

            # inital point
            if len(obs_lst) < seq_len:
                pad =[obs_lst[0]] * (seq_len - len(obs_lst))
                seq_list = pad + obs_lst
            else:
                seq_list = obs_lst[-seq_len:]
            
            obs_seq = torch.stack(seq_list).unsqueeze(0)  # (1, seq_len, feature_dim)
            # print(f'obs_seq.shape: {obs_seq.shape}')

            with torch.no_grad():
                q_values, hidden = q_net(obs_seq, hidden)
            # select_action
            if random.random() < epsilon:
                action = random.randint(0, 4)
            else:
                action = torch.argmax(q_values[:, -1, :]).item()
            

            next_obs, reward, done = env.step(action)
            total_reward += reward
            step += 1

            # Render
            if args.render:
                env.render(view=True)
            
            obs_lst.append(next_obs)
            # print(f'next_obs.shape: {next_obs.shape}')
            # next seq

            if len(obs_lst) < seq_len:
                pad =[obs_lst[0]] * (seq_len - len(obs_lst))
                next_seq_list = pad + obs_lst
            else:
                next_seq_list = obs_lst[-seq_len:]
            next_obs_seq = torch.stack(next_seq_list)  # (1, seq_len, feature_dim)

            # buffer push
            buffer.push(torch.stack(seq_list), action, reward, next_obs_seq)

            if len(buffer) >= warmup_steps:
                optimizer.zero_grad()

                # TD Learning
                # Experience Replay
                obs_b, act_b, rew_b, next_obs_b = buffer.sample(batch_size, seq_len=seq_len)

                q_values, _ = q_net(obs_b)
                q_sa = q_values.gather(2, act_b.unsqueeze(2)).squeeze(2)  # (batch_size, seq_len)

                 # Target: r + gamma * max_a' Q(s', a')
                with torch.no_grad():
                    q_next, _ = q_net(next_obs_b)
                    max_q_next = q_next.max(dim=1).values  # (batch_size, seq_len)
                    targets = rew_b + gamma * max_q_next
                
                loss = torch.mean((targets - q_sa) ** 2)
                loss.backward()
                optimizer.step()

        obs = next_obs
        epsilon = max(epsilon_min, epsilon * epsilon_decay)


        #     # Select action with epsilon-greedy method
        #     q_values, action, hidden = select_action(q_net, obs, epsilon, hidden)

        #     # Go one step
        #     next_obs, reward, done = env.step(action)
        #     total_reward += reward
        #     step += 1

        #     # Render
        #     if args.render:
        #         env.render(view=True)

        #     next_obs = next_obs.unsqueeze(0).unsqueeze(0)

        #     buffer.push(obs.squeeze(0), action, reward, next_obs.squeeze(0))
        #     obs = next_obs

        #     if len(buffer) >= warmup_steps:
        #         optimizer.zero_grad()

        #         # TD Learning
        #         # Experience Replay
        #         obs_b, act_b, rew_b, next_obs_b = buffer.sample(batch_size)
        #         q_values, _ = q_net(obs_b)
        #         q_sa = q_values.gather(1, act_b.unsqueeze(1)).squeeze(1)
        #          # Target: r + gamma * max_a' Q(s', a')
        #         with torch.no_grad():
        #             q_next, _ = q_net(next_obs_b)
        #             max_q_next = q_next.max(dim=1).values
        #             targets = rew_b + gamma * max_q_next
                
        #         loss = torch.mean((targets - q_sa) ** 2)
        #         loss.backward()
        #         optimizer.step()
        # # Decay epsilon for epsilon-greedy
        # epsilon = max(epsilon_min, epsilon * epsilon_decay)
    
        print(f"Episode {episode+1}/{num_episodes} | "
              f"Steps: {step} | "
              f"Total reward: {total_reward:.2f} | "
              f"Epsilon: {epsilon:.3f}")
    
    print("\nTraining finished.")
    env.close()

    torch.save(q_net.state_dict(), f"q_lstm_network_150.pt")


if __name__ == "__main__":
    random.seed(SEED)
    torch.manual_seed(SEED)
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', '-r', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
