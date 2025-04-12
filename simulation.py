"""
Simulation module for differential vehicle simulation.
This module handles the 3D simulation and visualization of the vehicle.
"""

import numpy as np
from direct.showbase.ShowBase import ShowBase
from panda3d.core import Vec3, Point3, LVector3, BitMask32
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import Texture, TextureStage
from panda3d.core import LineSegs, NodePath
from direct.task import Task
from vehicle_model import DifferentialVehicle

class DifferentialVehicleSimulation:
    """
    3D simulation of the differential vehicle, handling visualization
    and visual updates based on the physical model.
    """
    
    def __init__(self, render_node, vehicle_model):
        """
        Initialize the 3D simulation environment.
        
        Args:
            render_node: The render node to attach objects to
            vehicle_model (DifferentialVehicle): The physical model of the vehicle
        """
        self.render = render_node
        self.vehicle = vehicle_model
        
        # Set up the environment
        self._setup_environment()
        
        # Create the vehicle visualization
        self._create_vehicle()
        
        # Set up path visualization
        self.path_lines = LineSegs()
        self.path_lines.setThickness(2.0)
        self.path_lines.setColor(1.0, 0.8, 0.2, 1.0)
        self.path_node = self.render.attachNewNode(self.path_lines.create())
        self.path_node.setZ(0.01)  # Slightly above the floor
        
        # Path waypoints for visualization
        self.waypoints = []
        
        # Last position for drawing path
        self.last_position = (self.vehicle.x, self.vehicle.y)
    
    def _setup_environment(self):
        """Set up the 3D environment including terrain, lights, etc."""
        # Create a flat ground plane
        self.ground = self.render.attachNewNode("Ground")
        
        # Simple box for the floor
        size = 20
        self.floor = self.render.attachNewNode("Floor")
        
        # Create a grid for reference
        grid_lines = LineSegs()
        grid_lines.setThickness(1.0)
        grid_lines.setColor(0.5, 0.5, 0.5, 1.0)
        
        # Draw grid lines
        grid_size = 20
        grid_step = 1
        for i in range(-grid_size, grid_size + 1, grid_step):
            grid_lines.moveTo(-grid_size, i, 0.01)
            grid_lines.drawTo(grid_size, i, 0.01)
            grid_lines.moveTo(i, -grid_size, 0.01)
            grid_lines.drawTo(i, grid_size, 0.01)
            
        # Highlight the axes
        grid_lines.setThickness(3.0)
        grid_lines.setColor(1.0, 0.0, 0.0, 1.0)  # X-axis in red
        grid_lines.moveTo(0, 0, 0.01)
        grid_lines.drawTo(grid_size, 0, 0.01)
        
        grid_lines.setColor(0.0, 1.0, 0.0, 1.0)  # Y-axis in green
        grid_lines.moveTo(0, 0, 0.01)
        grid_lines.drawTo(0, grid_size, 0.01)
        
        grid_node = self.render.attachNewNode(grid_lines.create())
        
        # Add a flat floor
        floor_color = (0.2, 0.2, 0.25, 1.0)
        floor_size = grid_size * 2
        floor_thickness = 0.1
        
        # Create a visual representation of the floor using a box
        floor_node = self.render.attachNewNode("FloorNode")
        floor_node.setPos(0, 0, -floor_thickness/2)
        # We would normally load a model here, but for simplicity, we'll use a marker
        
        # Setup lighting
        # Ambient light for basic visibility
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        
        # Main directional light (sun)
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1.0))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(30, -60, 0)  # Point the light downwards at an angle
        self.render.setLight(dlnp)
    
    def _create_vehicle(self):
        """Create the 3D visualization of the vehicle."""
        # Create a node for the vehicle
        self.vehicle_node = self.render.attachNewNode("VehicleNode")
        self.vehicle_node.setPos(0, 0, self.vehicle.wheel_radius)
        
        # Create vehicle body
        body_size = (0.15, 0.15, 0.05)  # Width, length, height
        self.body_node = self.vehicle_node.attachNewNode("BodyNode")
        
        # Create a simple shape for the body
        body_lines = LineSegs()
        body_lines.setThickness(2.0)
        body_lines.setColor(0.2, 0.6, 1.0, 1.0)
        
        # Draw a box for the body
        w, l, h = body_size
        body_lines.moveTo(-w/2, -l/2, 0)
        body_lines.drawTo(w/2, -l/2, 0)
        body_lines.drawTo(w/2, l/2, 0)
        body_lines.drawTo(-w/2, l/2, 0)
        body_lines.drawTo(-w/2, -l/2, 0)
        
        body_lines.moveTo(-w/2, -l/2, h)
        body_lines.drawTo(w/2, -l/2, h)
        body_lines.drawTo(w/2, l/2, h)
        body_lines.drawTo(-w/2, l/2, h)
        body_lines.drawTo(-w/2, -l/2, h)
        
        body_lines.moveTo(-w/2, -l/2, 0)
        body_lines.drawTo(-w/2, -l/2, h)
        body_lines.moveTo(w/2, -l/2, 0)
        body_lines.drawTo(w/2, -l/2, h)
        body_lines.moveTo(w/2, l/2, 0)
        body_lines.drawTo(w/2, l/2, h)
        body_lines.moveTo(-w/2, l/2, 0)
        body_lines.drawTo(-w/2, l/2, h)
        
        # Draw a direction indicator
        body_lines.setColor(1.0, 0.4, 0.4, 1.0)
        body_lines.moveTo(0, 0, h)
        body_lines.drawTo(0, l, h)
        
        self.body_node.attachNewNode(body_lines.create())
        
        # Create the wheels
        wheel_radius = self.vehicle.wheel_radius
        wheel_width = 0.02
        wheel_distance = self.vehicle.wheel_distance
        
        self.left_wheel = self.vehicle_node.attachNewNode("LeftWheel")
        self.right_wheel = self.vehicle_node.attachNewNode("RightWheel")
        
        # Position the wheels
        self.left_wheel.setPos(-wheel_distance/2, 0, 0)
        self.right_wheel.setPos(wheel_distance/2, 0, 0)
        
        # Create visual representation for wheels
        for wheel in [self.left_wheel, self.right_wheel]:
            wheel_lines = LineSegs()
            wheel_lines.setThickness(2.0)
            wheel_lines.setColor(0.3, 0.3, 0.3, 1.0)
            
            # Draw a circle for the wheel
            segments = 16
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                x = wheel_width / 2
                y = wheel_radius * np.cos(angle)
                z = wheel_radius * np.sin(angle)
                if i == 0:
                    wheel_lines.moveTo(x, y, z)
                else:
                    wheel_lines.drawTo(x, y, z)
            
            for i in range(segments + 1):
                angle = 2 * np.pi * i / segments
                x = -wheel_width / 2
                y = wheel_radius * np.cos(angle)
                z = wheel_radius * np.sin(angle)
                if i == 0:
                    wheel_lines.moveTo(x, y, z)
                else:
                    wheel_lines.drawTo(x, y, z)
            
            # Connect the circles
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                y = wheel_radius * np.cos(angle)
                z = wheel_radius * np.sin(angle)
                wheel_lines.moveTo(wheel_width/2, y, z)
                wheel_lines.drawTo(-wheel_width/2, y, z)
            
            wheel.attachNewNode(wheel_lines.create())
        
        # Update the vehicle position to match the model
        self.update_vehicle_position()
    
    def update_vehicle_position(self):
        """Update the visual position and orientation of the vehicle based on the model."""
        # Set position
        self.vehicle_node.setPos(self.vehicle.x, self.vehicle.y, self.vehicle.wheel_radius)
        
        # Set orientation (heading angle)
        self.vehicle_node.setH(-np.degrees(self.vehicle.theta))
        
        # Update wheel rotation based on motor positions
        left_angle = -np.degrees(self.vehicle.left_motor.get_position()) % 360
        right_angle = -np.degrees(self.vehicle.right_motor.get_position()) % 360
        
        self.left_wheel.setR(left_angle)
        self.right_wheel.setR(right_angle)
    
    def update(self, dt):
        """
        Update the simulation for one time step.
        
        Args:
            dt (float): Time step in seconds
        """
        # Update the vehicle model
        self.vehicle.update(dt)
        
        # Update the visual position
        self.update_vehicle_position()
        
        # Draw path
        current_pos = (self.vehicle.x, self.vehicle.y)
        if np.linalg.norm(np.array(current_pos) - np.array(self.last_position)) > 0.05:
            self.path_lines.moveTo(self.last_position[0], self.last_position[1], 0.02)
            self.path_lines.drawTo(current_pos[0], current_pos[1], 0.02)
            self.path_node.removeNode()
            self.path_node = self.render.attachNewNode(self.path_lines.create())
            self.path_node.setZ(0.02)  # Slightly above the floor
            self.last_position = current_pos
    
    def add_waypoint(self, x, y):
        """Add a waypoint to the path for visualization."""
        # Create a visual marker for the waypoint
        waypoint_lines = LineSegs()
        waypoint_lines.setThickness(3.0)
        waypoint_lines.setColor(1.0, 0.0, 1.0, 1.0)
        
        # Draw a cross at the waypoint
        size = 0.1
        waypoint_lines.moveTo(x - size, y, 0.03)
        waypoint_lines.drawTo(x + size, y, 0.03)
        waypoint_lines.moveTo(x, y - size, 0.03)
        waypoint_lines.drawTo(x, y + size, 0.03)
        
        # Create and add the waypoint node
        waypoint_node = self.render.attachNewNode(waypoint_lines.create())
        self.waypoints.append(waypoint_node)
    
    def clear_waypoints(self):
        """Clear all waypoints from the visualization."""
        for waypoint in self.waypoints:
            waypoint.removeNode()
        self.waypoints = []
    
    def clear_path(self):
        """Clear the drawn path."""
        self.path_lines = LineSegs()
        self.path_lines.setThickness(2.0)
        self.path_lines.setColor(1.0, 0.8, 0.2, 1.0)
        self.path_node.removeNode()
        self.path_node = self.render.attachNewNode(self.path_lines.create())
        self.last_position = (self.vehicle.x, self.vehicle.y)
