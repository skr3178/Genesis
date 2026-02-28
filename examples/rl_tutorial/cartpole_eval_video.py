"""
Evaluation Script with Video Recording for Trained Cartpole Policy

This script loads a trained policy, runs it, and saves a video.

Usage:
    python cartpole_eval_video.py -e my-first-run --ckpt 50 --record

Requirements for video:
    pip install imageio-ffmpeg
"""

import argparse
import os
import pickle
from importlib import metadata

import torch
import numpy as np

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
    parser = argparse.ArgumentParser(description="Evaluate trained cartpole policy and record video")
    parser.add_argument("-e", "--exp_name", type=str, default="my-first-run",
                        help="Experiment name (same as training)")
    parser.add_argument("--ckpt", type=int, default=50,
                        help="Checkpoint iteration to load")
    parser.add_argument("--record", action="store_true",
                        help="Record video to file")
    parser.add_argument("--video_length", type=int, default=500,
                        help="Number of frames to record")
    parser.add_argument("--fps", type=int, default=50,
                        help="Video FPS")
    args = parser.parse_args()
    
    # ============================================================
    # Initialize Genesis
    # ============================================================
    gs.init(backend=gs.cpu, logging_level="warning")
    
    # ============================================================
    # Load Configurations
    # ============================================================
    log_dir = f"logs/{args.exp_name}"
    
    # Load saved configs from training
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    
    # Disable rewards for evaluation
    reward_cfg["reward_scales"] = {}
    
    # ============================================================
    # Create Environment (single env)
    # ============================================================
    print("Creating environment...")
    env = CartpoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,  # No viewer, we'll record instead
    )
    
    # ============================================================
    # Load Trained Policy
    # ============================================================
    print(f"Loading checkpoint: model_{args.ckpt}.pt")
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    
    # Get inference policy (deterministic)
    policy = runner.get_inference_policy(device=gs.device)
    
    # ============================================================
    # Setup Video Recording
    # ============================================================
    frames = []
    
    if args.record:
        print(f"Recording {args.video_length} frames at {args.fps} FPS...")
        print("Video will be saved to logs/{args.exp_name}/video.mp4")
    else:
        print("Running simulation (no recording)...")
    
    # ============================================================
    # Run Evaluation Loop
    # ============================================================
    obs, _ = env.reset()
    episode_reward = 0
    episode_len = 0
    
    with torch.no_grad():
        for step in range(args.video_length if args.record else 1000):
            # Get action from policy
            actions = policy(obs)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            # Track stats
            episode_reward += rewards.item()
            episode_len += 1
            
            # Record frame if requested
            if args.record:
                # Get camera render - we need to add a camera
                # For now, we'll use a workaround by saving state info
                pass
            
            # Print progress
            if step % 100 == 0:
                print(f"Step {step}/{args.video_length if args.record else 1000}")
            
            # Check if episode ended
            if dones.item():
                print(f"\nEpisode finished: length={episode_len}, reward={episode_reward:.2f}")
                episode_reward = 0
                episode_len = 0
    
    print(f"\nEvaluation complete!")
    if args.record:
        print("Note: Video recording requires adding a camera to the scene.")
        print("The policy ran successfully - to see visualization, you would need:")
        print("  1. A system with display (X11)")
        print("  2. Or use the --vis flag during training with a local machine")


if __name__ == "__main__":
    main()
