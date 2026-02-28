import math
import numpy as np
import genesis as gs
import torch


def run_simulation(friction, damping, filename, num_steps=500):
    """Run FEM simulation with specified friction and damping"""
    
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            substeps=10,
            gravity=(0, 0, -9.8),
        ),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(2.0, 0, 1.0),  # Wider view
            camera_lookat=(0.5, 0.3, 0.0),
            camera_fov=50,
        ),
        fem_options=gs.options.FEMOptions(
            dt=1e-4,
            damping=damping,
            use_implicit_solver=False,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
        ),
    )
    
    # Ground with specified friction
    scene.add_entity(
        morph=gs.morphs.Plane(),
        material=gs.materials.Rigid(coup_friction=friction),
    )
    
    # FEM Muscle worm
    worm = scene.add_entity(
        morph=gs.morphs.Mesh(
            file='meshes/worm/worm.obj',
            pos=(0.3, 0.3, 0.05),
            scale=0.1,
            euler=(90, 0, 0),
        ),
        material=gs.materials.FEM.Muscle(
            E=5e5,
            nu=0.45,
            rho=10000.0,
            model='stable_neohookean',
            n_groups=4,
        ),
    )
    
    cam = scene.add_camera(
        pos=(2.0, 0, 1.0),
        lookat=(0.5, 0.3, 0.0),
        fov=50,
    )
    
    scene.build()
    
    # Set up muscle groups
    pos = worm.get_state().pos[0, worm.get_el2v()].mean(dim=1)
    n_units = worm.n_elements
    pos_max, pos_min = pos.max(dim=0).values, pos.min(dim=0).values
    pos_range = pos_max - pos_min
    
    lu_thresh, fh_thresh = 0.3, 0.6
    muscle_group = torch.zeros((n_units,), dtype=gs.tc_int, device=gs.device)
    mask_upper = pos[:, 2] > (pos_min[2] + pos_range[2] * lu_thresh)
    mask_fore = pos[:, 1] < (pos_min[1] + pos_range[1] * fh_thresh)
    muscle_group[mask_upper & mask_fore] = 0
    muscle_group[mask_upper & ~mask_fore] = 1
    muscle_group[~mask_upper & mask_fore] = 2
    muscle_group[~mask_upper & ~mask_fore] = 3
    
    muscle_direction = torch.tensor([[0.0, 1.0, 0.0]] * n_units, dtype=gs.tc_float, device=gs.device)
    worm.set_muscle(muscle_group=muscle_group, muscle_direction=muscle_direction)
    
    # Record
    video_interval = 5
    cam.start_recording()
    scene.reset()
    
    for i in range(num_steps):
        actu = (0.0, 0.0, 0.0, 1.0 * (0.5 + math.sin(0.005 * math.pi * i)))
        worm.set_actuation(actu)
        scene.step()
        
        if i % video_interval == 0:
            cam.render()
    
    cam.stop_recording(save_to_filename=filename, fps=30)
    print(f"Saved: {filename}")
    print(f"  Friction={friction}, Damping={damping}")


# Initialize once
gs.init(seed=0, precision='32', logging_level='warning')

# Test 1: Low friction, low damping (should slide)
print("\n=== Test 1: Low friction (0.1), Low damping (10) ===")
run_simulation(friction=0.1, damping=10.0, filename='fem_low_friction_low_damping.mp4')

# Test 2: Low friction, high damping (stays put)
print("\n=== Test 2: Low friction (0.1), High damping (100) ===")
run_simulation(friction=0.1, damping=100.0, filename='fem_low_friction_high_damping.mp4')

# Test 3: High friction, low damping (locomotion works)
print("\n=== Test 3: High friction (5.0), Low damping (10) ===")
run_simulation(friction=5.0, damping=10.0, filename='fem_high_friction_low_damping.mp4')

print("\n=== All tests complete ===")
