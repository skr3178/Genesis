"""
Record Video of Trained Cartpole Policy (Matplotlib Version)

Usage:
    python cartpole_video_simple.py -e my-first-run --ckpt 50 --frames 500

Output:
    logs/{exp_name}/video_{ckpt}.mp4
"""

import argparse
import os
import pickle
from importlib import metadata

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Headless backend
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch
from matplotlib.lines import Line2D

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


def draw_cartpole(ax, cart_pos, pole_angle, track_width=5.0):
    """Draw cartpole in matplotlib."""
    ax.clear()
    
    # Track
    ax.plot([-track_width/2, track_width/2], [0, 0], 'k-', linewidth=2)
    
    # Cart (box)
    cart_width = 0.3
    cart_height = 0.2
    cart = FancyBboxPatch(
        (cart_pos - cart_width/2, 0),
        cart_width, cart_height,
        boxstyle="round,pad=0.02",
        facecolor='#4488ff',
        edgecolor='black',
        linewidth=2
    )
    ax.add_patch(cart)
    
    # Pole
    pole_length = 1.0
    pole_x = cart_pos + pole_length * np.sin(pole_angle)
    pole_y = cart_height/2 + pole_length * np.cos(pole_angle)
    
    ax.plot([cart_pos, pole_x], [cart_height/2, pole_y], 
            'r-', linewidth=8, solid_capstyle='round')
    
    # Pivot point
    ax.plot(cart_pos, cart_height/2, 'ko', markersize=10)
    
    # Settings
    ax.set_xlim(-track_width/2 - 0.5, track_width/2 + 0.5)
    ax.set_ylim(-0.5, 2.0)
    ax.set_aspect('equal')
    ax.set_xlabel('Position (m)')
    ax.set_ylabel('Height (m)')
    ax.set_title(f'Cartpole - Cart: {cart_pos:.2f}m, Pole: {np.degrees(pole_angle):.1f}°')
    ax.grid(True, alpha=0.3)
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my-first-run")
    parser.add_argument("--ckpt", type=int, default=50)
    parser.add_argument("--frames", type=int, default=500)
    parser.add_argument("--fps", type=int, default=50)
    args = parser.parse_args()

    # ============================================================
    # Initialize Genesis
    # ============================================================
    gs.init(backend=gs.cpu, logging_level="warning")

    # ============================================================
    # Load Configs
    # ============================================================
    log_dir = f"logs/{args.exp_name}"
    
    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(
        open(f"{log_dir}/cfgs.pkl", "rb")
    )
    reward_cfg["reward_scales"] = {}

    # ============================================================
    # Create Environment
    # ============================================================
    from cartpole_env import CartpoleEnv
    
    env = CartpoleEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )

    # ============================================================
    # Load Policy
    # ============================================================
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{args.ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # ============================================================
    # Record Video
    # ============================================================
    video_path = os.path.join(log_dir, f"video_{args.ckpt}.mp4")
    print(f"Recording {args.frames} frames to {video_path}...")

    obs, _ = env.reset()
    
    # Setup figure
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    frames = []

    with torch.no_grad():
        for i in range(args.frames):
            # Get action
            actions = policy(obs)
            
            # Step
            obs, rewards, dones, infos = env.step(actions)
            
            # Get state for visualization
            cart_pos = env.cart_pos[0].item()
            pole_angle = env.pole_angle[0].item()
            
            # Draw
            draw_cartpole(ax, cart_pos, pole_angle)
            fig.canvas.draw()
            
            # Convert to array
            buf = fig.canvas.buffer_rgba()
            frame = np.asarray(buf)
            frame = frame[:, :, :3]  # Drop alpha
            frames.append(frame)
            
            if (i + 1) % 100 == 0:
                print(f"Frame {i+1}/{args.frames}")
            
            # Reset if done
            if dones.item():
                print(f"Episode ended at frame {i+1}")
                obs, _ = env.reset()

    plt.close(fig)

    # ============================================================
    # Save Video
    # ============================================================
    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=args.fps, quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"\n✅ Video saved to: {video_path}")
        print(f"   Duration: {len(frames)/args.fps:.1f}s at {args.fps} FPS")
        print(f"   Resolution: {frames[0].shape[1]}x{frames[0].shape[0]}")
    except Exception as e:
        print(f"\n❌ Error saving video: {e}")
        print("Saving frames as numpy array...")
        npy_path = video_path.replace('.mp4', '.npy')
        np.save(npy_path, np.array(frames))
        print(f"Frames saved to: {npy_path}")


if __name__ == "__main__":
    main()
