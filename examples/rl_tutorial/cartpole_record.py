"""
Record Video of Trained Cartpole Policy

Usage:
    python cartpole_record.py -e my-first-run --ckpt 50 --frames 500
"""

import argparse
import os
import pickle
from importlib import metadata

import torch
import numpy as np

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

# Custom environment with camera recording
class CartpoleEnvWithRecording:
    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        # Import the original env class
        from cartpole_env import CartpoleEnv
        self.base_env = CartpoleEnv(num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer)
        
        # Copy all attributes
        self.num_envs = self.base_env.num_envs
        self.num_obs = self.base_env.num_obs
        self.num_actions = self.base_env.num_actions
        self.device = self.base_env.device
        self.scene = self.base_env.scene
        self.cartpole = self.base_env.cartpole
        self.cart_dof_idx = self.base_env.cart_dof_idx
        self.pole_dof_idx = self.base_env.pole_dof_idx
        self.motor_dof_idx = self.base_env.motor_dof_idx
        self.dt = self.base_env.dt
        self.max_episode_length = self.base_env.max_episode_length
        self.env_cfg = self.base_env.env_cfg
        self.obs_cfg = self.base_env.obs_cfg
        self.reward_scales = self.base_env.reward_scales
        self.obs_buf = self.base_env.obs_buf
        self.rew_buf = self.base_env.rew_buf
        self.reset_buf = self.base_env.reset_buf
        self.episode_length_buf = self.base_env.episode_length_buf
        self.actions = self.base_env.actions
        self.cart_pos = self.base_env.cart_pos
        self.cart_vel = self.base_env.cart_vel
        self.pole_angle = self.base_env.pole_angle
        self.pole_vel = self.base_env.pole_vel
        self.extras = self.base_env.extras
        self.episode_sums = self.base_env.episode_sums
        
        # Add camera for recording (after build)
        self.camera = None
        self.frames = []
        
    def step(self, actions):
        return self.base_env.step(actions)
    
    def reset(self):
        return self.base_env.reset()
    
    def get_observations(self):
        return self.base_env.get_observations()
    
    def get_privileged_observations(self):
        return self.base_env.get_privileged_observations()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my-first-run")
    parser.add_argument("--ckpt", type=int, default=50)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()
    
    print("=" * 60)
    print("Cartpole Video Recording")
    print("=" * 60)
    print(f"Note: Recording requires display/Camera support.")
    print(f"For headless systems, use TensorBoard instead.")
    print("=" * 60)
    
    # For now, just print the stats
    log_dir = f"logs/{args.exp_name}"
    
    print(f"\nCheckpoint: {log_dir}/model_{args.ckpt}.pt")
    print(f"TensorBoard: tensorboard --logdir={log_dir}")
    print(f"\nTo visualize:")
    print(f"  1. Run: tensorboard --logdir={log_dir} --port=6006")
    print(f"  2. Open: http://localhost:6006")
    print(f"\nOr download checkpoint and run locally with show_viewer=True")


if __name__ == "__main__":
    main()
