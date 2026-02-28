"""
Record Video of Trained Cartpole Policy

Usage:
    python cartpole_video.py -e my-first-run --ckpt 50 --frames 500 --fps 50

Output:
    logs/{exp_name}/video_{ckpt}.mp4
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="my-first-run")
    parser.add_argument("--ckpt", type=int, default=50)
    parser.add_argument("--frames", type=int, default=300)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("-B", "--num_envs", type=int, default=1)
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
    # Create Scene with Camera
    # ============================================================
    from cartpole_env import CartpoleEnv
    
    # Monkey-patch to add camera before build
    original_init = CartpoleEnv.__init__
    
    def init_with_camera(self, num_envs, env_cfg, obs_cfg, reward_cfg, show_viewer=False):
        # Call original init but don't build yet
        self.num_envs = num_envs
        self.num_obs = obs_cfg["num_obs"]
        self.num_actions = env_cfg["num_actions"]
        self.device = gs.device
        self.dt = env_cfg["dt"]
        self.max_episode_length = int(env_cfg["episode_length_s"] / self.dt)
        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_scales = reward_cfg["reward_scales"]

        # Create scene
        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(enable_collision=False, enable_joint_limit=True),
            viewer_options=gs.options.ViewerOptions(),
            vis_options=gs.options.VisOptions(rendered_envs_idx=[0]),
            show_viewer=False,
        )
        
        # Add ground
        self.scene.add_entity(gs.morphs.Plane())
        
        # Add cartpole
        self.cartpole = self.scene.add_entity(
            gs.morphs.URDF(file="urdf/cartpole/urdf/cartpole.urdf"),
        )
        
        # Add camera for recording (BEFORE build)
        self.camera = self.scene.add_camera(
            res=(640, 480),
            pos=(2.5, 0.0, 1.5),  # Side view
            lookat=(0.0, 0.0, 0.5),
            fov=45,
            GUI=False,
        )
        
        # Build
        # Patch visualizer to skip EGL
        original_vis_build = self.scene._visualizer.build
        def patched_vis_build():
            self.scene._visualizer._context.build(self.scene)
            if self.scene._visualizer._viewer is not None:
                self.scene._visualizer._viewer.build(self.scene)
            # Skip rasterizer
            if self.scene._visualizer._raytracer is not None:
                self.scene._visualizer._raytracer.build(self.scene)
            for camera in self.scene._visualizer._cameras:
                camera.build()
            if self.scene._visualizer._batch_renderer is not None:
                self.scene._visualizer._batch_renderer.build()
            self.scene._visualizer._is_built = True
            self.scene._visualizer.reset()
        self.scene._visualizer.build = patched_vis_build
        
        self.scene.build(n_envs=num_envs)
        
        # Setup joints
        self.cart_dof_idx = 6
        self.pole_dof_idx = 7
        self.motor_dof_idx = [self.cart_dof_idx]
        
        self.cartpole.set_dofs_kp([self.env_cfg["kp"]], [self.cart_dof_idx])
        self.cartpole.set_dofs_kv([self.env_cfg["kd"]], [self.cart_dof_idx])
        
        # Buffers
        self.obs_buf = torch.zeros((num_envs, self.num_obs), dtype=gs.tc_float, device=gs.device)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.actions = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.cart_pos = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.cart_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_angle = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.pole_vel = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.extras = {"observations": {}}
        self.episode_sums = {}
        for name in self.reward_scales.keys():
            self.episode_sums[name] = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
    
    CartpoleEnv.__init__ = init_with_camera

    # Create environment
    env = CartpoleEnv(
        num_envs=args.num_envs,
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
    frames = []

    with torch.no_grad():
        for i in range(args.frames):
            # Get action
            actions = policy(obs)
            
            # Step
            obs, rewards, dones, infos = env.step(actions)
            
            # Render frame
            rgb, _, _, _ = env.camera.render(rgb=True, depth=False, segmentation=False, normal=False)
            frame = rgb[0].cpu().numpy()  # (H, W, 3)
            frames.append(frame)
            
            if (i + 1) % 50 == 0:
                print(f"Frame {i+1}/{args.frames}")

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
    except ImportError:
        print("\n❌ imageio not installed. Saving frames as numpy array...")
        np.save(video_path.replace('.mp4', '.npy'), np.array(frames))
        print(f"Frames saved to: {video_path.replace('.mp4', '.npy')}")
        print("Install imageio to save as video: pip install imageio imageio-ffmpeg")


if __name__ == "__main__":
    main()
