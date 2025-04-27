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
    def __init__(self, render_node, vehicle_model):
        self.render = render_node
        self.vehicle = vehicle_model
        self._setup_environment()
        self._create_vehicle()
        self.path_lines = LineSegs()
        self.path_lines.setThickness(2.0)
        self.path_lines.setColor(1.0, 0.8, 0.2, 1.0)
        self.path_node = self.render.attachNewNode(self.path_lines.create())
        self.path_node.setZ(0.01)
        self.waypoints = []
        self.last_position = (self.vehicle.x, self.vehicle.y)
    
    def _setup_environment(self):
        self.ground = self.render.attachNewNode("Ground")
        size = 20
        self.floor = self.render.attachNewNode("Floor")
        grid_lines = LineSegs()
        grid_lines.setThickness(1.0)
        grid_lines.setColor(0.5, 0.5, 0.5, 1.0)
        grid_size = 20
        grid_step = 1
        for i in range(-grid_size, grid_size + 1, grid_step):
            grid_lines.moveTo(-grid_size, i, 0.01)
            grid_lines.drawTo(grid_size, i, 0.01)
            grid_lines.moveTo(i, -grid_size, 0.01)
            grid_lines.drawTo(i, grid_size, 0.01)
        grid_lines.setThickness(3.0)
        grid_lines.setColor(1.0, 0.0, 0.0, 1.0)
        grid_lines.moveTo(0, 0, 0.01)
        grid_lines.drawTo(grid_size, 0, 0.01)
        grid_lines.setColor(0.0, 1.0, 0.0, 1.0)
        grid_lines.moveTo(0, 0, 0.01)
        grid_lines.drawTo(0, grid_size, 0.01)
        grid_node = self.render.attachNewNode(grid_lines.create())
        floor_color = (0.2, 0.2, 0.25, 1.0)
        floor_size = grid_size * 2
        floor_thickness = 0.1
        floor_node = self.render.attachNewNode("FloorNode")
        floor_node.setPos(0, 0, -floor_thickness/2)
        alight = AmbientLight('alight')
        alight.setColor((0.3, 0.3, 0.3, 1.0))
        alnp = self.render.attachNewNode(alight)
        self.render.setLight(alnp)
        dlight = DirectionalLight('dlight')
        dlight.setColor((0.8, 0.8, 0.8, 1.0))
        dlnp = self.render.attachNewNode(dlight)
        dlnp.setHpr(30, -60, 0)
        self.render.setLight(dlnp)
    
    def _create_vehicle(self):
        self.vehicle_node = self.render.attachNewNode("VehicleNode")
        self.vehicle_node.setPos(0, 0, self.vehicle.wheel_radius)
        body_size = (0.15, 0.15, 0.05)
        self.body_node = self.vehicle_node.attachNewNode("BodyNode")
        body_lines = LineSegs()
        body_lines.setThickness(2.0)
        body_lines.setColor(0.2, 0.6, 1.0, 1.0)
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
        body_lines.setColor(1.0, 0.4, 0.4, 1.0)
        body_lines.moveTo(0, 0, h)
        body_lines.drawTo(0, l, h)
        self.body_node.attachNewNode(body_lines.create())
        wheel_radius = self.vehicle.wheel_radius
        wheel_width = 0.02
        wheel_distance = self.vehicle.wheel_distance
        self.left_wheel = self.vehicle_node.attachNewNode("LeftWheel")
        self.right_wheel = self.vehicle_node.attachNewNode("RightWheel")
        self.left_wheel.setPos(-wheel_distance/2, 0, 0)
        self.right_wheel.setPos(wheel_distance/2, 0, 0)
        for wheel in [self.left_wheel, self.right_wheel]:
            wheel_lines = LineSegs()
            wheel_lines.setThickness(2.0)
            wheel_lines.setColor(0.3, 0.3, 0.3, 1.0)
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
            for i in range(segments):
                angle = 2 * np.pi * i / segments
                y = wheel_radius * np.cos(angle)
                z = wheel_radius * np.sin(angle)
                wheel_lines.moveTo(wheel_width/2, y, z)
                wheel_lines.drawTo(-wheel_width/2, y, z)
            wheel.attachNewNode(wheel_lines.create())
        self.update_vehicle_position()
    
    def update_vehicle_position(self):
        self.vehicle_node.setPos(self.vehicle.x, self.vehicle.y, self.vehicle.wheel_radius)
        self.vehicle_node.setH(-np.degrees(self.vehicle.theta))
        left_angle = -np.degrees(self.vehicle.left_motor.get_position()) % 360
        right_angle = -np.degrees(self.vehicle.right_motor.get_position()) % 360
        self.left_wheel.setR(left_angle)
        self.right_wheel.setR(right_angle)
    
    def update(self, dt):
        self.vehicle.update(dt)
        self.update_vehicle_position()
        current_pos = (self.vehicle.x, self.vehicle.y)
        if np.linalg.norm(np.array(current_pos) - np.array(self.last_position)) > 0.05:
            self.path_lines.moveTo(self.last_position[0], self.last_position[1], 0.02)
            self.path_lines.drawTo(current_pos[0], current_pos[1], 0.02)
            self.path_node.removeNode()
            self.path_node = self.render.attachNewNode(self.path_lines.create())
            self.path_node.setZ(0.02)
            self.last_position = current_pos
    
    def add_waypoint(self, x, y):
        waypoint_lines = LineSegs()
        waypoint_lines.setThickness(3.0)
        waypoint_lines.setColor(1.0, 0.0, 1.0, 1.0)
        size = 0.1
        waypoint_lines.moveTo(x - size, y, 0.03)
        waypoint_lines.drawTo(x + size, y, 0.03)
        waypoint_lines.moveTo(x, y - size, 0.03)
        waypoint_lines.drawTo(x, y + size, 0.03)
        waypoint_node = self.render.attachNewNode(waypoint_lines.create())
        self.waypoints.append(waypoint_node)
    
    def clear_waypoints(self):
        for waypoint in self.waypoints:
            waypoint.removeNode()
        self.waypoints = []
    
    def clear_path(self):
        self.path_lines = LineSegs()
        self.path_lines.setThickness(2.0)
        self.path_lines.setColor(1.0, 0.8, 0.2, 1.0)
        self.path_node.removeNode()
        self.path_node = self.render.attachNewNode(self.path_lines.create())
        self.last_position = (self.vehicle.x, self.vehicle.y)
