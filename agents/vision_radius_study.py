import argparse
import math
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import src.config as cfg
from src.env import GameEnv
from src.objects import Arrow
from src.model import Q as FeedforwardQ
from agents.dqn_lstm_agent import RecurrentQNetwork as LSTMNetwork, parse_hidden_sizes as parse_lstm_sizes


def load_feedforward_agent(model_path, device):
    net = FeedforwardQ().to(device)
    net.load_state_dict(torch.load(model_path, map_location=device))
    net.eval()
    return net


class LSTMRunner:
    def __init__(self, net, device):
        self.net = net
        self.device = device

    def run_episode(self, radius, max_steps, render):
        cfg.VISION_RADIUS = radius
        env = GameEnv()
        env.reset()
        obs = env.get_obs()
        hidden = None
        steps = 0
        done = False
        while not done and steps < max_steps:
            inp = obs.to(self.device).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                q_values, hidden = self.net(inp, hidden)
                if hidden is not None:
                    hidden = tuple(h.detach() for h in hidden) if isinstance(hidden, tuple) else hidden.detach()
                action = int(torch.argmax(q_values[:, -1, :], dim=1).item())
            obs, _, done = env.step(action)
            steps += 1
            if render:
                env.render(view=True, step=steps)
            if done:
                hidden = None
        env.close()
        return steps


def build_state_space(vision_radius):
    state_space = {(0, 0, 0, 0): 0}
    idx = 1
    for x in range(-vision_radius, vision_radius + 1):
        for y in range(-vision_radius, vision_radius + 1):
            if x * x + y * y > vision_radius * vision_radius:
                continue
            for speed in range(cfg.ARROW_SPEED_MIN, cfg.ARROW_SPEED_MAX + 1):
                for angle in range(0, 360, 10):
                    state_space[(speed, angle, x, y)] = idx
                    idx += 1
    return state_space


def mdp_select_action(env, obs, vision_radius, qstar, state_space):
    agent_pos = env.agent.get_position()
    terminal_idx = state_space[(0, 0, 0, 0)]
    q_dim = qstar.shape[0]
    if not obs:
        return int(np.argmax(qstar[terminal_idx % q_dim, :]))
    q_values = []
    for arrow in obs:
        arr_x, arr_y = arrow.get_position()
        arr_x = round(arr_x - agent_pos[0])
        arr_y = round(arr_y - agent_pos[1])
        dist = math.sqrt(arr_x ** 2 + arr_y ** 2)
        if dist > vision_radius or dist < cfg.AGENT_RADIUS + cfg.ARROW_RADIUS:
            cur_state = (0, 0, 0, 0)
        else:
            rounded_angle = round(arrow.angle / 10) * 10
            if rounded_angle < 0:
                rounded_angle += 360
            rounded_angle = rounded_angle % 360
            arr_x = max(-vision_radius, min(vision_radius, arr_x))
            arr_y = max(-vision_radius, min(vision_radius, arr_y))
            cur_state = (arrow.speed, rounded_angle, arr_x, arr_y)
        cur_idx = state_space.get(cur_state, terminal_idx)
        q_values.append(qstar[cur_idx % q_dim, :])
    if not q_values:
        return int(np.argmax(qstar[terminal_idx % q_dim, :]))
    q_values = np.stack(q_values, axis=0)
    q_min = np.min(q_values, axis=0)
    return int(np.argmax(q_min))


def run_mdp_episode(radius, max_steps, render, qstar, state_space_cache):
    cfg.VISION_RADIUS = radius
    env = GameEnv()
    env.reset()
    state_space = state_space_cache.get(radius)
    if state_space is None:
        state_space = build_state_space(radius)
        state_space_cache[radius] = state_space
    obs = env.get_obs2()
    steps = 0
    done = False
    while not done and steps < max_steps:
        action = mdp_select_action(env, obs, radius, qstar, state_space)
        obs, _, done = env.step2(action)
        steps += 1
        if render:
            env.render(view=True, step=steps)
    env.close()
    return steps


def run_feedforward_episode(net, radius, max_steps, render, device):
    cfg.VISION_RADIUS = radius
    env = GameEnv()
    env.reset()
    obs = env.get_obs().to(device)
    steps = 0
    done = False
    while not done and steps < max_steps:
        with torch.no_grad():
            q_values = net(obs.unsqueeze(0))
            action = int(torch.argmax(q_values, dim=1).item())
        obs, _, done = env.step(action)
        obs = obs.to(device)
        steps += 1
        if render:
            env.render(view=True, step=steps)
    env.close()
    return steps


def reseed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)


def summarize(values):
    arr = np.array(values, dtype=np.float32)
    return (
        float(arr.mean()),
        float(np.median(arr)),
        float(arr.std(ddof=0)),
        int(arr.max()),
        int(arr.min()),
    )


def parse_radii(radii_str):
    entries = [r.strip() for r in radii_str.split(",") if r.strip()]
    return [int(r) for r in entries]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vision-radii", type=str, default="150,300,450")
    parser.add_argument("--runs-per-radius", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=6000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--mdp-qstar", type=str, default=None)
    parser.add_argument("--mdp-qstar-template", type=str, default="mdp_qstar_{radius}.npy")
    parser.add_argument("--nn-model", type=str, default=None)
    parser.add_argument("--nn-model-template", type=str, default="q_network_{radius}.pt")
    parser.add_argument("--lstm-model", type=str, default=None)
    parser.add_argument("--lstm-model-template", type=str, default="agents/dqn_lstm_policy_{radius}.pt")
    parser.add_argument("--lstm-hidden-sizes", type=str, default="512,512,256")
    parser.add_argument("--lstm-hidden", type=int, default=512)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-mlp-dropout", type=float, default=0.1)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    args = parser.parse_args()

    radii = parse_radii(args.vision_radii)
    state_dim = GameEnv().get_obs().shape[0]
    action_dim = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    agents = {}
    nn_runners = {}

    def build_feedforward_runner(model_path):
        if not model_path or not os.path.exists(model_path):
            return None, f"model {model_path} not found."
        try:
            net = load_feedforward_agent(model_path, device)
            return net, None
        except Exception as exc:
            return None, str(exc)

    if args.nn_model_template:
        for radius in radii:
            model_path = args.nn_model_template.format(radius=radius)
            net, err = build_feedforward_runner(model_path)
            if net is None:
                print(f"Skipping run_nn_agent for radius {radius}: {err}")
                continue
            nn_runners[radius] = net
    elif args.nn_model:
        net, err = build_feedforward_runner(args.nn_model)
        if net is None:
            print(f"Skipping run_nn_agent: {err}")
        else:
            for radius in radii:
                nn_runners[radius] = net
    else:
        print("Skipping run_nn_agent: no model path provided.")

    lstm_runners = {}
    default_lstm_cfg = {
        "hidden_sizes": args.lstm_hidden_sizes,
        "lstm_hidden": args.lstm_hidden,
        "lstm_layers": args.lstm_layers,
        "mlp_dropout": args.lstm_mlp_dropout,
        "lstm_dropout": args.lstm_dropout,
    }

    def build_lstm_runner(model_path):
        try:
            payload = torch.load(model_path, map_location=device)
        except FileNotFoundError:
            return None, f"model {model_path} not found."
        except Exception as exc:
            return None, str(exc)

        state_dict = payload
        metadata = {}
        if isinstance(payload, dict) and "model_state_dict" in payload:
            state_dict = payload["model_state_dict"]
            metadata = payload.get("config", {})
        hidden_sizes = metadata.get("hidden_sizes")
        if hidden_sizes is None:
            hidden_cfg = default_lstm_cfg["hidden_sizes"]
            hidden_sizes = parse_lstm_sizes(hidden_cfg) if isinstance(hidden_cfg, str) else list(hidden_cfg)
        else:
            hidden_sizes = [int(v) for v in hidden_sizes]
        lstm_hidden = metadata.get("lstm_hidden", default_lstm_cfg["lstm_hidden"])
        lstm_layers = metadata.get("lstm_layers", default_lstm_cfg["lstm_layers"])
        mlp_dropout = metadata.get("mlp_dropout", default_lstm_cfg["mlp_dropout"])
        lstm_dropout = metadata.get("lstm_dropout", default_lstm_cfg["lstm_dropout"])
        net = LSTMNetwork(
            state_dim,
            action_dim,
            hidden_sizes,
            int(lstm_hidden),
            int(lstm_layers),
            float(mlp_dropout),
            float(lstm_dropout),
        ).to(device)
        try:
            net.load_state_dict(state_dict)
        except RuntimeError as exc:
            return None, str(exc)
        net.eval()
        return LSTMRunner(net, device), None

    if args.lstm_model_template:
        for radius in radii:
            model_path = args.lstm_model_template.format(radius=radius)
            runner, err = build_lstm_runner(model_path)
            if runner is None:
                print(f"Skipping run_lstm_agent for radius {radius}: {err}")
                continue
            lstm_runners[radius] = runner
    elif args.lstm_model:
        runner, err = build_lstm_runner(args.lstm_model)
        if runner is None:
            print(f"Skipping run_lstm_agent: {err}")
        else:
            for radius in radii:
                lstm_runners[radius] = runner
    else:
        print("Skipping run_lstm_agent: no model path provided.")

    if nn_runners:
        agents["run_nn_agent"] = nn_runners
    if lstm_runners:
        agents["run_lstm_agent"] = lstm_runners

    qstar_cache = {}
    state_space_cache = {}

    results = defaultdict(lambda: defaultdict(list))
    eval_agents = list(agents.keys()) + ["run_mdp_agent"]
    sample_env = GameEnv()
    sample_env.close()

    for agent_idx, agent_name in enumerate(eval_agents):
        for radius in radii:
            for run_idx in range(args.runs_per_radius):
                run_seed = args.seed + agent_idx * 100000 + radius * 100 + run_idx
                reseed(run_seed)
                if agent_name == "run_mdp_agent":
                    if args.mdp_qstar_template:
                        qstar_path = args.mdp_qstar_template.format(radius=radius)
                    else:
                        qstar_path = args.mdp_qstar
                    if qstar_path is None or not os.path.exists(qstar_path):
                        print(f"[run_mdp_agent] skipping radius {radius}: q* file {qstar_path} not found.")
                        continue
                    if qstar_path not in qstar_cache:
                        qstar_cache[qstar_path] = np.load(qstar_path)
                    steps = run_mdp_episode(radius, args.max_steps, args.render, qstar_cache[qstar_path], state_space_cache)
                elif agent_name == "run_lstm_agent":
                    runner = agents[agent_name].get(radius)
                    if runner is None:
                        print(f"[run_lstm_agent] skipping radius {radius}: no matching model.")
                        continue
                    steps = runner.run_episode(radius, args.max_steps, args.render)
                elif agent_name == "run_nn_agent":
                    net = agents[agent_name].get(radius)
                    if net is None:
                        print(f"[run_nn_agent] skipping radius {radius}: no matching model.")
                        continue
                    steps = run_feedforward_episode(net, radius, args.max_steps, args.render, device)
                else:
                    steps = run_feedforward_episode(agents[agent_name], radius, args.max_steps, args.render, device)
                results[agent_name][radius].append(steps)
                print(f"[{agent_name}] radius {radius} run {run_idx + 1}/{args.runs_per_radius}: {steps} steps")

    print("\n=== Vision Radius Study Summary ===")
    for agent_name in eval_agents:
        if agent_name != "run_mdp_agent" and agent_name not in agents:
            continue
        print(f"\nAgent: {agent_name}")
        for radius in radii:
            values = results[agent_name][radius]
            if not values:
                print(f"  Vision {radius}: no data")
                continue
            mean_val, median_val, std_val, longest, shortest = summarize(values)
            print(
                f"  Vision {radius:3d} -> mean {mean_val:.2f} | median {median_val:.2f} | std {std_val:.2f} | longest {longest} | shortest {shortest}"
            )


if __name__ == "__main__":
    main()
