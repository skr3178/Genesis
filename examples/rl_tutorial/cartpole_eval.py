"""
Evaluation Script for Trained Cartpole Policy

This script loads a trained policy and runs it in the environment with visualization.

Usage:
    python cartpole_eval.py -e cartpole-exp --ckpt 300

The script will:
1. Load the saved configuration and trained model
2. Create a single environment with visualization
3. Run the policy and render the simulation
"""

import argparse
import os
import pickle
from importlib import metadata

import torch

# Check for correct rsl-rl-lib version
try:
    try:
        if metadata.version("rsl-rl"):
            raise ImportError
    except metadata.PackageNotFoundError:
        if metadata.version("rsl-rl-lib") != "2.2.4":
            raise ImportError
except (metadata.PackageNotFoundError, ImportError) as e:
    raise ImportError("Please uninstall 'rsl_rl' and install 'rsl-rl-lib==2.2.4'.") from e

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from cartpole_env import CartpoleEnv


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained cartpole policy")
    parser.add_argument("-e", "--exp_name", type=str, default="cartpole-rl",
                        help="Experiment name (same as training)")
    parser.add_argument("--ckpt", type=int, default=300,
                        help="Checkpoint iteration to load")
    args = parser.parse_args()
    
    # ============================================================
    # Initialize Genesis (CPU is fine for single-env evaluation)
    # ============================================================
    gs.init(backend=gs.cpu)
    
    # ============================================================
    # Load Configurations
    # ============================================================
    log_dir = f"logs/{args.exp_name}"
    
    # Load saved configs from training
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    
    # Disable rewards for evaluation (we just want to watch)
    reward_cfg["reward_scales"] = {}
    
    # ============================================================
    # Create Environment (single env with viewer)
    # ============================================================
    env = CartpoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=True,  # Enable visualization
    )
    
    # ============================================================
    # Load Trained Policy
    # ============================================================
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    # Load the checkpoint
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    print(f"Loaded checkpoint: {resume_path}")
    
    # Get inference policy (deterministic, no exploration noise)
    policy = runner.get_inference_policy(device=gs.device)
    
    # ============================================================
    # Run Evaluation Loop
    # ============================================================
    print("Running evaluation... Press Ctrl+C to stop")
    print("=" * 50)
    
    obs, _ = env.reset()
    episode_reward = 0
    episode_len = 0
    num_episodes = 0
    
    with torch.no_grad():
        while True:
            # Get action from policy
            actions = policy(obs)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            # Track episode stats
            episode_reward += rewards.item()
            episode_len += 1
            
            # Check if episode ended
            if dones.item():
                num_episodes += 1
                print(f"Episode {num_episodes}: length={episode_len}, reward={episode_reward:.2f}")
                episode_reward = 0
                episode_len = 0


if __name__ == "__main__":
    main()
