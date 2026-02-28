import math
import numpy as np
import genesis as gs
import torch


########################## init ##########################
gs.init(seed=0, precision='32', logging_level='info')

########################## create a scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        substeps=10,
        gravity=(0, 0, -9.8),
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(1.5, 0, 0.8),
        camera_lookat=(0.0, 0.0, 0.0),
        camera_fov=40,
    ),
    pbd_options=gs.options.PBDOptions(
        dt=1e-3,  # PBD can use larger timesteps
        gravity=(0, 0, -9.8),
        particle_size=0.01,
        max_stretch_solver_iterations=10,
        max_bending_solver_iterations=5,
        lower_bound=(-1.0, -1.0, -0.2),
        upper_bound=(1.0, 1.0, 1.0),
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
    ),
)

########################## entities ##########################
# Low friction ground plane
scene.add_entity(
    morph=gs.morphs.Plane(),
    material=gs.materials.Rigid(
        coup_friction=0.1,  # LOW FRICTION
    ),
)

# PBD Elastic worm (NO muscle support - passive only)
# PBD uses particles with constraints, not muscle actuation
worm = scene.add_entity(
    morph=gs.morphs.Mesh(
        file='meshes/worm/worm.obj',
        pos=(0.3, 0.3, 0.05),
        scale=0.1,
        euler=(90, 0, 0),
    ),
    material=gs.materials.PBD.Elastic(
        rho=10000.0,
        static_friction=0.1,  # Low friction
        kinetic_friction=0.1,
        stretch_compliance=1e-5,  # Stiff material
        volume_compliance=1e-5,
    ),
)

########################## add camera for recording ##########################
cam = scene.add_camera(
    pos=(1.5, 0, 0.8),
    lookat=(0.3, 0.3, 0.0),
    fov=40,
)

########################## build ##########################
scene.build()

########################## run and record video ##########################
video_interval = 5
cam.start_recording()

scene.reset()

# PBD doesn't support set_actuation - no muscle actuation available
# Just let the worm fall and settle
for i in range(500):
    scene.step()
    
    if i % video_interval == 0:
        cam.render()

########################## save video ##########################
cam.stop_recording(save_to_filename='advanced_worm_pbd.mp4', fps=30)
print("Video saved to: advanced_worm_pbd.mp4")
print("\nNote: PBD does NOT support muscle actuation!")
print("This demo shows PBD stability with low friction, but no locomotion.")
