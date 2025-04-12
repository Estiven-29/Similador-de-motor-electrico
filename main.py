#!/usr/bin/env python
"""
Tesla Differential Vehicle Simulation
Universidad Popular del Cesar - Ingeniería de Sistemas
Modelos y Simulación - Parcial 2 abril de 2025

Main application entry point for the 3D simulation of a two-wheeled differential vehicle.
"""

import sys
import numpy as np
from direct.showbase.ShowBase import ShowBase
from direct.gui.OnscreenText import OnscreenText
from direct.gui.DirectGui import DirectButton, DirectEntry, DirectFrame, DirectLabel
from panda3d.core import TextNode, PandaNode, NodePath, Vec3, Point3
from panda3d.core import WindowProperties

from simulation import DifferentialVehicleSimulation
from ui_controller import UIController
from motor_model import ElectricMotor
from vehicle_model import DifferentialVehicle
from path_planner import PathPlanner
from position_controller import PositionController

class DifferentialVehicleApp(ShowBase):
    """Main application class for the differential vehicle simulation."""
    
    def __init__(self):
        ShowBase.__init__(self)
        
        # Configure window
        self.setBackgroundColor(0.1, 0.1, 0.2)
        props = WindowProperties()
        props.setTitle("Tesla Differential Vehicle Simulation")
        props.setSize(1280, 720)
        self.win.requestProperties(props)
        
        # Setup camera
        self.disableMouse()
        self.camera.setPos(0, -10, 5)
        self.camera.lookAt(0, 0, 0)
        
        # Create the motor models (example values from real DC motors)
        left_motor = ElectricMotor(
            name="Left Motor",
            voltage_constant=0.03,    # V/(rad/s)
            torque_constant=0.03,     # Nm/A
            resistance=1.0,           # Ohms
            inertia=0.001,            # kg*m^2
            friction=0.01             # Nm/(rad/s)
        )
        
        right_motor = ElectricMotor(
            name="Right Motor",
            voltage_constant=0.03,    # V/(rad/s)
            torque_constant=0.03,     # Nm/A
            resistance=1.0,           # Ohms
            inertia=0.001,            # kg*m^2
            friction=0.01             # Nm/(rad/s)
        )
        
        # Create the vehicle model
        self.vehicle = DifferentialVehicle(
            wheel_radius=0.05,        # 5 cm
            wheel_distance=0.2,       # 20 cm
            robot_mass=1.0,           # 1 kg
            wheel_mass=0.1,           # 0.1 kg each
            left_motor=left_motor,
            right_motor=right_motor
        )
        
        # Create the path planner
        self.path_planner = PathPlanner()
        
        # Create the position controller
        self.position_controller = PositionController(self.vehicle)
        
        # Create the simulation
        self.simulation = DifferentialVehicleSimulation(self.render, self.vehicle)
        
        # Create the UI
        self.ui = UIController(self, self.vehicle, self.simulation, self.path_planner, self.position_controller)
        
        # Setup update task
        self.taskMgr.add(self.update, "update")
        
        # Add instructions
        self.instructions = OnscreenText(
            text="Tesla Differential Vehicle Simulation\n"
                 "Use the UI panels to control the vehicle",
            style=1, fg=(1, 1, 1, 1), pos=(-0.95, 0.95), align=TextNode.ALeft, scale=.05)
        
        print("Tesla Differential Vehicle Simulation initialized")
        
    def update(self, task):
        """Main update loop"""
        dt = globalClock.getDt()
        
        # Update the simulation
        self.simulation.update(dt)
        
        # Update the UI
        self.ui.update(dt)
        
        return task.cont

if __name__ == "__main__":
    app = DifferentialVehicleApp()
    app.run()
