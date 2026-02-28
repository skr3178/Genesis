import genesis as gs
import numpy as np

# Use GPU for simulation
gs.init(backend=gs.gpu)

# No viewer - we'll render frames manually
scene = gs.Scene(show_viewer=False)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file='xml/franka_emika_panda/panda.xml'),
)

# Add camera for offscreen rendering
cam = scene.add_camera(
    res=(640, 480),
    pos=(2.0, 1.5, 2.0),  # Camera position
    lookat=(0, 0, 0.5),   # Look at point
)

scene.build()

# Save video
cam.start_recording()

for i in range(1000):
    scene.step()
    
    # Render every frame (or every N frames)
    cam.render()
    
    if i % 100 == 0:
        print(f"Step {i}/1000")

cam.stop_recording(save_to_filename='franka_simulation.mp4', fps=60)
print("Video saved to franka_simulation.mp4")
