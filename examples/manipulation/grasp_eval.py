import argparse
import re
import pickle
from importlib import metadata
from pathlib import Path

import torch

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

from grasp_env import GraspEnv
from behavior_cloning import BehaviorCloning


def load_rl_policy(env, train_cfg, log_dir):
    """Load reinforcement learning policy."""
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"model_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    runner.load(last_ckpt)
    print(f"Loaded RL checkpoint from {last_ckpt}")

    return runner.get_inference_policy(device=gs.device)


def load_bc_policy(env, bc_cfg, log_dir):
    """Load behavior cloning policy."""
    # Create behavior cloning instance
    bc_runner = BehaviorCloning(env, bc_cfg, None, device=gs.device)

    # Find the latest checkpoint
    checkpoint_files = [f for f in log_dir.iterdir() if re.match(r"checkpoint_\d+\.pt", f.name)]
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")

    try:
        *_, last_ckpt = sorted(checkpoint_files)
    except ValueError as e:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}") from e
    print(f"Loaded BC checkpoint from {last_ckpt}")
    bc_runner.load(last_ckpt)

    return bc_runner._policy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp_name", type=str, default="grasp")
    parser.add_argument(
        "--stage",
        type=str,
        default="rl",
        choices=["rl", "bc"],
        help="Model type: 'rl' for reinforcement learning, 'bc' for behavior cloning",
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="Record stereo images as video during evaluation",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory to save videos (default: logs/{exp_name}_{stage}/recordings/)",
    )
    args = parser.parse_args()

    # Set PyTorch default dtype to float32 for better performance
    torch.set_default_dtype(torch.float32)

    gs.init()

    log_dir = Path("logs") / f"{args.exp_name + '_' + args.stage}"

    # Load configurations
    if args.stage == "rl":
        # For RL, load the standard configs
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(log_dir / "cfgs.pkl", "rb"))
    else:
        # For BC, we need to load the configs and create BC config
        env_cfg, reward_cfg, robot_cfg, rl_train_cfg, bc_train_cfg = pickle.load(open(log_dir / "cfgs.pkl", "rb"))

    # set the max FPS for visualization
    env_cfg["max_visualize_FPS"] = 60
    # set the box collision
    env_cfg["box_collision"] = True
    # set the box fixed
    env_cfg["box_fixed"] = False
    # set the number of envs for evaluation
    env_cfg["num_envs"] = 10
    # for video recording
    env_cfg["visualize_camera"] = args.record

    env = GraspEnv(
        env_cfg=env_cfg,
        reward_cfg=reward_cfg,
        robot_cfg=robot_cfg,
        show_viewer=not args.record,  # Disable viewer when recording on headless machines
    )

    # Load the appropriate policy based on model type
    if args.stage == "rl":
        policy = load_rl_policy(env, rl_train_cfg, log_dir)
    else:
        policy = load_bc_policy(env, bc_train_cfg, log_dir)
        policy.eval()

    obs, _ = env.reset()

    max_sim_step = int(env_cfg["episode_length_s"] * env_cfg["max_visualize_FPS"])
    
    # When recording, ensure at least 11 seconds of video to capture full grasp and lift
    if args.record:
        min_steps_for_11s = int(11 * env_cfg["max_visualize_FPS"])  # 660 steps at 60 FPS
        max_sim_step = max(max_sim_step, min_steps_for_11s)

    with torch.no_grad():
        if args.record:
            print("Recording video...")
            env.vis_cam.start_recording()
            env.left_cam.start_recording()
            env.right_cam.start_recording()
        for step in range(max_sim_step):
            if args.stage == "rl":
                actions = policy(obs)
            else:
                # Get stereo grayscale images and ensure float32
                rgb_obs = env.get_stereo_rgb_images(normalize=True).float()
                ee_pose = env.robot.ee_pose.float()

                actions = policy(rgb_obs, ee_pose)

            # Collect frame for video recording
            if args.record:
                env.vis_cam.render()  # render the visualization camera

            obs, rews, dones, infos = env.step(actions)
        env.grasp_and_lift_demo(record=args.record, vis_cam=env.vis_cam if args.record else None)
        if args.record:
            print("Stopping video recording...")
            # Determine video save directory
            if args.video_dir:
                video_dir = Path(args.video_dir)
            else:
                video_dir = log_dir / "recordings"
            video_dir.mkdir(parents=True, exist_ok=True)
            
            # Save videos with stage-appropriate names
            vis_video_path = video_dir / f"{args.stage}_vis_cam.mp4"
            left_video_path = video_dir / f"{args.stage}_left_cam.mp4"
            right_video_path = video_dir / f"{args.stage}_right_cam.mp4"
            
            env.vis_cam.stop_recording(save_to_filename=str(vis_video_path), fps=env_cfg["max_visualize_FPS"])
            env.left_cam.stop_recording(save_to_filename=str(left_video_path), fps=env_cfg["max_visualize_FPS"])
            env.right_cam.stop_recording(save_to_filename=str(right_video_path), fps=env_cfg["max_visualize_FPS"])
            print(f"Videos saved to: {video_dir}")


if __name__ == "__main__":
    main()

"""
# evaluation
# For reinforcement learning model:
python examples/manipulation/grasp_eval.py --stage=rl

# For behavior cloning model:
python examples/manipulation/grasp_eval.py --stage=bc

# With video recording:
python examples/manipulation/grasp_eval.py --stage=bc --record
"""
