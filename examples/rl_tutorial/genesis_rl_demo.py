"""
Genesis RL Demo - All-in-One: Train + Record Video

Demonstrates Genesis physics engine + PPO reinforcement learning.

Key features vs the original tutorial:
  - LARGE initial perturbations: pole starts at ~17° (0.3 rad), cart offset 0.5m
  - Matplotlib-based video recording (no EGL/display dependency)
  - Single script: trains from scratch then immediately records the video
  - Color-coded visualization with HUD (angle, position, episode info)

Usage:
    cd /media/skr/storage/Garment/Genesis/examples/rl_tutorial
    conda run -n bcat python genesis_rl_demo.py

Optional flags:
    --iters 200          Number of PPO iterations  (default 200)
    --envs  512          Parallel environments      (default 512)
    --frames 600         Video frames               (default 600)
    --fps    50          Video FPS                  (default 50)
    --exp    demo-run    Experiment name            (default genesis-rl-demo)
    --eval_only          Skip training, just make video from existing run
"""

import argparse
import os
import math
import pickle
import shutil
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless – no X11 needed
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.gridspec as gridspec

import torch


# ────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────

def get_cfgs():
    env_cfg = {
        "num_actions": 1,
        "dt": 0.0025,             # 400 Hz - same as original, stable with rigid solver
        "episode_length_s": 10.0,
        "kp": 50.0,
        "kd": 5.0,
        "action_scale": 0.5,
        "clip_actions": 10.0,
        "termination_angle": 0.8,  # ~46° - wide so policy can recover from 17°
        "cart_limit": 2.5,
        # LARGE perturbations so the video clearly shows the pole off-balance
        "init_perturbation": {
            "cart_pos":   0.5,   # ±0.5 m
            "pole_angle": 0.3,   # ±0.3 rad (~±17°)
            "cart_vel":   0.2,   # ±0.2 m/s
            "pole_vel":   0.3,   # ±0.3 rad/s
        },
    }
    obs_cfg = {
        "num_obs": 4,
        "obs_scales": {
            "cart_pos":   1.0,
            "cart_vel":   0.5,
            "pole_angle": 1.0,
            "pole_vel":   0.5,
        },
    }
    reward_cfg = {
        "reward_scales": {
            "pole_upright": 10.0,
            "pole_balance": 0.1,
            "cart_center":  0.5,
            "survival":     1.0,
        },
    }
    return env_cfg, obs_cfg, reward_cfg


def get_train_cfg(exp_name, max_iters):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
        },
        "init_member_classes": {},
        "policy": {
            "activation": "elu",
            "actor_hidden_dims": [128, 128],
            "critic_hidden_dims": [128, 128],
            "init_noise_std": 1.0,
            "class_name": "ActorCritic",
        },
        "runner": {
            "checkpoint": -1,
            "experiment_name": exp_name,
            "load_run": -1,
            "log_interval": 10,
            "max_iterations": max_iters,
            "record_interval": -1,
            "resume": False,
            "resume_path": None,
            "run_name": "",
        },
        "runner_class_name": "OnPolicyRunner",
        "num_steps_per_env": 50,
        "save_interval": 50,
        "empirical_normalization": None,
        "seed": 42,
    }


# ────────────────────────────────────────────────────────────────────
# Training
# ────────────────────────────────────────────────────────────────────

def train(args):
    from importlib import metadata
    try:
        try:
            if metadata.version("rsl-rl"):
                raise ImportError
        except metadata.PackageNotFoundError:
            if metadata.version("rsl-rl-lib") != "2.2.4":
                raise ImportError
    except (metadata.PackageNotFoundError, ImportError):
        raise ImportError(
            "Install rsl-rl-lib==2.2.4:\n"
            "  pip uninstall rsl_rl rsl-rl-lib\n"
            "  pip install rsl-rl-lib==2.2.4"
        )

    from rsl_rl.runners import OnPolicyRunner
    import genesis as gs
    from cartpole_env_v2 import CartpoleEnvV2

    log_dir = f"logs/{args.exp}"
    env_cfg, obs_cfg, reward_cfg = get_cfgs()
    train_cfg = get_train_cfg(args.exp, args.iters)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir, exist_ok=True)

    pickle.dump([env_cfg, obs_cfg, reward_cfg, train_cfg], open(f"{log_dir}/cfgs.pkl", "wb"))

    gs.init(backend=gs.cpu, precision="32", logging_level="warning", seed=train_cfg["seed"])

    print(f"\n[Genesis RL Demo] Creating {args.envs} parallel environments...")
    env = CartpoleEnvV2(
        num_envs=args.envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    print(f"[Genesis RL Demo] Training PPO for {args.iters} iterations...")
    print(f"  Init pole angle range : ±{math.degrees(env_cfg['init_perturbation']['pole_angle']):.0f}°")
    print(f"  Init cart pos range   : ±{env_cfg['init_perturbation']['cart_pos']:.1f} m")
    print(f"  Parallel envs         : {args.envs}")
    print(f"  Logs → {log_dir}\n")

    runner.learn(num_learning_iterations=args.iters, init_at_random_ep_len=False)

    print(f"\n[Genesis RL Demo] Training complete. Best checkpoint: model_{args.iters}.pt")
    return log_dir


# ────────────────────────────────────────────────────────────────────
# Matplotlib visualisation helpers
# ────────────────────────────────────────────────────────────────────

TRACK_HALF = 2.8
CART_W, CART_H = 0.32, 0.18
POLE_LEN = 1.0
POLE_W = 0.07
WHEEL_R = 0.07

# Gradient colours for pole angle
def pole_color(angle_rad):
    """Green → yellow → red as |angle| grows from 0 → 0.8 rad."""
    t = min(abs(angle_rad) / 0.8, 1.0)
    r = t
    g = 1.0 - 0.6 * t
    b = 0.0
    return (r, g, b)


def draw_frame(ax_cart, ax_angle, ax_pos,
               cart_pos, pole_angle, cart_vel, pole_vel,
               episode, step, ep_reward, ep_len):
    # ── main cartpole view ────────────────────────────────────────
    ax_cart.clear()
    ax_cart.set_facecolor("#1a1a2e")
    ax_cart.set_xlim(-TRACK_HALF - 0.3, TRACK_HALF + 0.3)
    ax_cart.set_ylim(-0.35, 2.2)
    ax_cart.set_aspect("equal")
    ax_cart.axis("off")

    # Track with end stops
    ax_cart.plot([-TRACK_HALF, TRACK_HALF], [0, 0], color="#aaaaaa", lw=3, zorder=1)
    for x in [-TRACK_HALF, TRACK_HALF]:
        ax_cart.plot([x, x], [0, 0.12], color="#ff4444", lw=4, zorder=2)

    # Shadow
    ax_cart.add_patch(plt.Ellipse(
        (cart_pos, -0.02), 0.55, 0.06,
        color="black", alpha=0.3, zorder=2
    ))

    # Cart body
    cart_col = "#4488ff"
    ax_cart.add_patch(FancyBboxPatch(
        (cart_pos - CART_W / 2, 0),
        CART_W, CART_H,
        boxstyle="round,pad=0.02",
        facecolor=cart_col, edgecolor="white", linewidth=1.5, zorder=4
    ))

    # Wheels
    for dx in [-CART_W * 0.32, CART_W * 0.32]:
        ax_cart.add_patch(Circle(
            (cart_pos + dx, -WHEEL_R * 0.5), WHEEL_R,
            color="#dddddd", zorder=3
        ))

    # Pole
    pivot_x = cart_pos
    pivot_y = CART_H
    pole_tip_x = pivot_x + POLE_LEN * math.sin(pole_angle)
    pole_tip_y = pivot_y + POLE_LEN * math.cos(pole_angle)

    pcol = pole_color(pole_angle)
    ax_cart.plot(
        [pivot_x, pole_tip_x], [pivot_y, pole_tip_y],
        color=pcol, lw=8, solid_capstyle="round", zorder=5
    )
    # Ball at tip
    ax_cart.add_patch(Circle((pole_tip_x, pole_tip_y), 0.06, color=pcol, zorder=6))
    # Pivot pin
    ax_cart.add_patch(Circle((pivot_x, pivot_y), 0.05, color="#ffdd44", zorder=7))

    # HUD
    angle_deg = math.degrees(pole_angle)
    ax_cart.text(
        -TRACK_HALF, 2.1,
        f"Episode {episode}   Step {step}   Reward {ep_reward:.1f}",
        color="white", fontsize=9, va="top"
    )
    ax_cart.text(
        TRACK_HALF, 2.1,
        f"Pole {angle_deg:+.1f}°   Cart {cart_pos:+.2f} m",
        color="white", fontsize=9, va="top", ha="right"
    )

    # ── angle history ─────────────────────────────────────────────
    # (stored as running list in ax_angle.history)
    if not hasattr(ax_angle, "history"):
        ax_angle.history = []
    ax_angle.history.append(math.degrees(pole_angle))
    if len(ax_angle.history) > 200:
        ax_angle.history.pop(0)

    ax_angle.clear()
    ax_angle.set_facecolor("#12122a")
    xs = range(len(ax_angle.history))
    ax_angle.plot(xs, ax_angle.history, color="#ffaa44", lw=1.5)
    ax_angle.axhline(0, color="white", lw=0.7, ls="--", alpha=0.4)
    ax_angle.axhline( math.degrees(0.3), color="red", lw=0.7, ls=":", alpha=0.5)
    ax_angle.axhline(-math.degrees(0.3), color="red", lw=0.7, ls=":", alpha=0.5)
    ax_angle.set_ylabel("Pole angle (°)", color="white", fontsize=8)
    ax_angle.tick_params(colors="white", labelsize=7)
    ax_angle.set_ylim(-50, 50)
    ax_angle.set_xlim(0, 200)
    for spine in ax_angle.spines.values():
        spine.set_edgecolor("#444466")

    # ── cart position history ─────────────────────────────────────
    if not hasattr(ax_pos, "history"):
        ax_pos.history = []
    ax_pos.history.append(cart_pos)
    if len(ax_pos.history) > 200:
        ax_pos.history.pop(0)

    ax_pos.clear()
    ax_pos.set_facecolor("#12122a")
    ax_pos.plot(range(len(ax_pos.history)), ax_pos.history, color="#44ddaa", lw=1.5)
    ax_pos.axhline(0, color="white", lw=0.7, ls="--", alpha=0.4)
    ax_pos.set_ylabel("Cart pos (m)", color="white", fontsize=8)
    ax_pos.tick_params(colors="white", labelsize=7)
    ax_pos.set_ylim(-3, 3)
    ax_pos.set_xlim(0, 200)
    for spine in ax_pos.spines.values():
        spine.set_edgecolor("#444466")


# ────────────────────────────────────────────────────────────────────
# Video recording
# ────────────────────────────────────────────────────────────────────

def record_video(log_dir, ckpt, num_frames, fps):
    from importlib import metadata
    try:
        try:
            if metadata.version("rsl-rl"):
                raise ImportError
        except metadata.PackageNotFoundError:
            if metadata.version("rsl-rl-lib") != "2.2.4":
                raise ImportError
    except (metadata.PackageNotFoundError, ImportError):
        raise ImportError("Install rsl-rl-lib==2.2.4")

    from rsl_rl.runners import OnPolicyRunner
    import genesis as gs
    from cartpole_env_v2 import CartpoleEnvV2

    env_cfg, obs_cfg, reward_cfg, train_cfg = pickle.load(open(f"{log_dir}/cfgs.pkl", "rb"))
    reward_cfg["reward_scales"] = {}   # no reward needed during eval

    gs.init(backend=gs.cpu, logging_level="warning")

    env = CartpoleEnvV2(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        show_viewer=False,
    )

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    resume_path = os.path.join(log_dir, f"model_{ckpt}.pt")
    runner.load(resume_path)
    policy = runner.get_inference_policy(device=gs.device)

    # ── figure layout ─────────────────────────────────────────────
    fig = plt.figure(figsize=(12, 7), facecolor="#0d0d1a")
    gs_layout = gridspec.GridSpec(2, 2, figure=fig,
                                  left=0.04, right=0.97,
                                  top=0.93, bottom=0.07,
                                  hspace=0.38, wspace=0.25)
    ax_cart  = fig.add_subplot(gs_layout[:, 0])   # left half, full height
    ax_angle = fig.add_subplot(gs_layout[0, 1])   # top right
    ax_pos   = fig.add_subplot(gs_layout[1, 1])   # bottom right

    fig.suptitle("Genesis + PPO  |  CartPole Balancing  |  Large Initial Perturbation",
                 color="white", fontsize=11, y=0.97)

    # ── run eval with FIXED large initial perturbation ────────────
    # Override: set the first episode to start at exactly 0.25 rad ≈ 14°
    obs, _ = env.reset()

    # Manually push the pole to a specific angle to guarantee the video
    # starts with a clearly visible tilt
    import genesis as _gs
    qpos = torch.zeros((1, env.cartpole.n_qs), device=_gs.device)
    qpos[0, 7] = 0.4      # cart 0.4 m off-center
    qpos[0, 8] = 0.25     # pole 0.25 rad (~14°) – clearly visible but recoverable
    env.cartpole.set_qpos(qpos, envs_idx=torch.tensor([0], device=_gs.device), zero_velocity=True)
    env._update_state()
    env._update_observation()
    obs = env.obs_buf.clone()

    frames = []
    episode = 1
    ep_reward = 0.0
    ep_len = 0

    print(f"\n[Genesis RL Demo] Recording {num_frames} frames ...")
    print(f"  Pole starts at ~14° (0.25 rad) - clearly tilted at video start")

    with torch.no_grad():
        for i in range(num_frames):
            actions = policy(obs)
            obs, rewards, dones, _ = env.step(actions)

            ep_reward += rewards.item()
            ep_len += 1

            cart_pos_val  = env.cart_pos[0].item()
            pole_angle_val = env.pole_angle[0].item()
            cart_vel_val  = env.cart_vel[0].item()
            pole_vel_val  = env.pole_vel[0].item()

            draw_frame(ax_cart, ax_angle, ax_pos,
                       cart_pos_val, pole_angle_val, cart_vel_val, pole_vel_val,
                       episode, ep_len, ep_reward, ep_len)

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[:, :, :3].copy())

            if dones.item():
                episode += 1
                ep_reward = 0.0
                ep_len = 0

            if (i + 1) % 100 == 0:
                print(f"  frame {i+1}/{num_frames}  ep {episode}  pole {math.degrees(pole_angle_val):+.1f}°")

    plt.close(fig)

    # ── save video ────────────────────────────────────────────────
    video_path = os.path.join(log_dir, f"cartpole_perturbed_{ckpt}.mp4")
    try:
        import imageio
        writer = imageio.get_writer(video_path, fps=fps, quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"\n  Video saved → {video_path}")
        print(f"  Duration: {len(frames)/fps:.1f}s  |  {frames[0].shape[1]}×{frames[0].shape[0]} px")
    except ImportError:
        npy = video_path.replace(".mp4", ".npy")
        np.save(npy, np.array(frames))
        print(f"\n  imageio not installed. Frames saved → {npy}")
        print("  Install imageio to get an mp4: pip install imageio imageio-ffmpeg")


# ────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Genesis RL Demo")
    parser.add_argument("--exp",       type=str, default="genesis-rl-demo")
    parser.add_argument("--iters",     type=int, default=200)
    parser.add_argument("--envs",      type=int, default=512)
    parser.add_argument("--frames",    type=int, default=600)
    parser.add_argument("--fps",       type=int, default=50)
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, record video from existing checkpoint")
    args = parser.parse_args()

    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    if not args.eval_only:
        log_dir = train(args)
    else:
        log_dir = f"logs/{args.exp}"
        if not os.path.exists(f"{log_dir}/cfgs.pkl"):
            print(f"No saved run found at {log_dir}. Run without --eval_only first.")
            sys.exit(1)
        print(f"[Genesis RL Demo] Skipping training, using {log_dir}")

    # Find latest checkpoint
    ckpt = args.iters
    if not os.path.exists(f"{log_dir}/model_{ckpt}.pt"):
        pts = [f for f in os.listdir(log_dir) if f.startswith("model_") and f.endswith(".pt")]
        if not pts:
            print("No checkpoint found! Training may have failed.")
            sys.exit(1)
        ckpt = max(int(p.replace("model_", "").replace(".pt", "")) for p in pts)
        print(f"[Genesis RL Demo] Using checkpoint model_{ckpt}.pt")

    record_video(log_dir, ckpt, args.frames, args.fps)

    print("\n[Genesis RL Demo] Done.")
    print(f"  Video  → logs/{args.exp}/cartpole_perturbed_{ckpt}.mp4")
    print(f"  Ckpts  → logs/{args.exp}/model_*.pt")


if __name__ == "__main__":
    main()
