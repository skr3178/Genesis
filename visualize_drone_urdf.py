#!/usr/bin/env python3
"""
Viser URDF Visualizer for Genesis Drone Models
Usage: python visualize_drone_urdf.py [--urdf cf2x|cf2p|racer]

Reference: https://viser.studio/main/examples/demos/urdf_visualizer/
"""

import argparse
import os
import time
from pathlib import Path

import viser
import viser.transforms as tf
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Visualize Genesis drone URDF files using Viser")
    parser.add_argument(
        "--urdf",
        type=str,
        default="cf2x",
        choices=["cf2x", "cf2p", "racer"],
        help="Drone URDF model to visualize"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port for viser server"
    )
    args = parser.parse_args()

    # Path to URDF files
    urdf_dir = Path("/media/skr/storage/Garment/Genesis/genesis/assets/urdf/drones")
    urdf_path = urdf_dir / f"{args.urdf}.urdf"
    
    if not urdf_path.exists():
        print(f"URDF file not found: {urdf_path}")
        return

    print(f"Loading URDF: {urdf_path}")
    
    # Import dependencies
    try:
        import yourdfpy
        import trimesh
    except ImportError as e:
        print(f"Missing dependency: {e}")
        print("Please install: pip install yourdfpy trimesh")
        return
    
    # Change to URDF directory so relative mesh paths resolve correctly
    original_dir = os.getcwd()
    os.chdir(urdf_dir)
    
    try:
        # Load URDF with yourdfpy
        robot = yourdfpy.URDF.load(
            urdf_path.name,
            build_scene_graph=True,
            build_collision_scene_graph=True,
            load_meshes=True,
        )
    finally:
        # Always change back to original directory
        os.chdir(original_dir)
    
    # Create viser server
    server = viser.ViserServer(port=args.port)
    
    print(f"Viser server started at: http://localhost:{args.port}")
    
    # Add grid and coordinate frame
    server.scene.add_grid(name="/grid", width=2.0, height=2.0, position=(0.0, 0.0, 0.0))
    server.scene.add_frame(name="/world_frame", axes_length=0.2, axes_radius=0.005)
    
    # Store mesh handles
    mesh_handles = {}
    
    def load_meshes():
        """Load all meshes from the URDF."""
        nonlocal mesh_handles
        
        for link_name in robot.link_map.keys():
            link = robot.link_map[link_name]
            
            if not link.visuals:
                continue
                
            for i, visual in enumerate(link.visuals):
                visual_name = f"/robot/{link_name}/visual_{i}"
                
                try:
                    # Get visual origin
                    origin = visual.origin
                    if origin is None:
                        origin = np.eye(4)
                    
                    # Handle different geometry types
                    if visual.geometry.mesh is not None:
                        mesh_filename = visual.geometry.mesh.filename
                        # Remove leading ./ if present
                        if mesh_filename.startswith('./'):
                            mesh_filename = mesh_filename[2:]
                        mesh_path = urdf_dir / mesh_filename
                        
                        if mesh_path.exists():
                            # Load mesh with trimesh
                            mesh = trimesh.load(mesh_path, force='mesh')
                            
                            # Get vertices and faces
                            vertices = mesh.vertices
                            faces = mesh.faces
                            
                            # Apply visual origin transform
                            vertices_h = np.hstack([vertices, np.ones((len(vertices), 1))])
                            vertices_transformed = (origin @ vertices_h.T).T[:, :3]
                            
                            handle = server.scene.add_mesh_simple(
                                name=visual_name,
                                vertices=vertices_transformed,
                                faces=faces,
                                color=(0.7, 0.7, 0.7),
                            )
                            mesh_handles[visual_name] = handle
                            
                    elif visual.geometry.box is not None:
                        size = visual.geometry.box.size
                        position = origin[:3, 3]
                        rotation = tf.SO3.from_matrix(origin[:3, :3]).wxyz
                        
                        handle = server.scene.add_box(
                            name=visual_name,
                            size=size,
                            position=position,
                            wxyz=rotation,
                            color=(0.5, 0.5, 0.5),
                        )
                        mesh_handles[visual_name] = handle
                        
                    elif visual.geometry.cylinder is not None:
                        radius = visual.geometry.cylinder.radius
                        length = visual.geometry.cylinder.length
                        position = origin[:3, 3]
                        rotation = tf.SO3.from_matrix(origin[:3, :3]).wxyz
                        
                        handle = server.scene.add_cylinder(
                            name=visual_name,
                            radius=radius,
                            length=length,
                            position=position,
                            wxyz=rotation,
                            color=(0.6, 0.6, 0.6),
                        )
                        mesh_handles[visual_name] = handle
                        
                except Exception as e:
                    print(f"Failed to load visual for {link_name}: {e}")
    
    def update_poses():
        """Update mesh poses based on current configuration."""
        for link_name in robot.link_map.keys():
            try:
                # Get global transform for this link
                pose = robot.get_transform(link_name)
                
                # Update all visuals for this link
                for visual_name in list(mesh_handles.keys()):
                    if f"/robot/{link_name}/" in visual_name:
                        handle = mesh_handles[visual_name]
                        handle.position = pose[:3, 3]
                        handle.wxyz = tf.SO3.from_matrix(pose[:3, :3]).wxyz
                        
            except Exception as e:
                print(f"Error updating {link_name}: {e}")
    
    # Initial load
    load_meshes()
    update_poses()
    
    # GUI controls
    with server.gui.add_folder("Drone Info"):
        server.gui.add_text(
            "Model",
            initial_value=args.urdf.upper(),
            disabled=True,
        )
        server.gui.add_text(
            "Links",
            initial_value=str(len(robot.link_map)),
            disabled=True,
        )
        server.gui.add_text(
            "Joints",
            initial_value=str(len(robot.joint_names)),
            disabled=True,
        )
    
    # Link visibility toggles
    link_visibilities = {}
    with server.gui.add_folder("Links"):
        for link_name in robot.link_map.keys():
            link_visibilities[link_name] = server.gui.add_checkbox(
                link_name,
                initial_value=True,
            )
    
    # Update visibility callback
    def update_visibility():
        for link_name, checkbox in link_visibilities.items():
            visible = checkbox.value
            for visual_name in mesh_handles:
                if f"/robot/{link_name}/" in visual_name:
                    mesh_handles[visual_name].visible = visible
    
    # Attach callback to each checkbox
    for checkbox in link_visibilities.values():
        checkbox.on_update(lambda _: update_visibility())
    
    # Print info
    print(f"\n{'='*50}")
    print(f"Robot: {args.urdf.upper()}")
    print(f"Links ({len(robot.link_map)}): {list(robot.link_map.keys())}")
    print(f"Joints ({len(robot.joint_names)}): {robot.joint_names}")
    print(f"\nOpen browser: http://localhost:{args.port}")
    print(f"{'='*50}")
    
    # Keep running
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.stop()


if __name__ == "__main__":
    main()
