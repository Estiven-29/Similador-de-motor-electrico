"""
UI Controller module for differential vehicle simulation.
This module handles the user interface for controlling the vehicle.
"""

import numpy as np
from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, DirectEntry
from direct.gui.DirectGui import DirectSlider, DirectOptionMenu
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Vec3, Point3

class UIController:
    """
    User interface controller for the differential vehicle simulation.
    Provides controls for:
    - Manual voltage control
    - Position and orientation setting
    - Waypoint management
    - Control algorithm selection
    """
    
    def __init__(self, app, vehicle, simulation, path_planner, position_controller):
        """
        Initialize the UI controller.
        
        Args:
            app: The main application
            vehicle: The vehicle model
            simulation: The simulation environment
            path_planner: The path planner module
            position_controller: The position controller module
        """
        self.app = app
        self.vehicle = vehicle
        self.simulation = simulation
        self.path_planner = path_planner
        self.position_controller = position_controller
        
        # State variables
        self.control_mode = "manual"  # "manual", "waypoint", "position"
        self.left_voltage = 0.0
        self.right_voltage = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        
        # Create the UI elements
        self._create_ui()
    
    def _create_ui(self):
        """Create the UI elements."""
        # Main control panel frame
        self.main_frame = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0.8),
            frameSize=(-0.8, 0.8, -0.4, 0.4),
            pos=(0, 0, -0.5)
        )
        
        # Create tabs for different control modes
        self._create_tabs()
        
        # Create the manual control panel
        self._create_manual_control()
        
        # Create the waypoint control panel
        self._create_waypoint_control()
        
        # Create the position control panel
        self._create_position_control()
        
        # Create the telemetry panel
        self._create_telemetry()
        
        # Show the default panel
        self._show_panel("manual")
    
    def _create_tabs(self):
        """Create the tabs for different control modes."""
        tab_width = 0.25
        tab_height = 0.08
        tab_y = 0.4
        
        # Manual control tab
        self.manual_tab = DirectButton(
            text="Manual",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(-0.5, 0, tab_y),
            command=self._show_panel,
            extraArgs=["manual"],
            parent=self.main_frame
        )
        
        # Waypoint control tab
        self.waypoint_tab = DirectButton(
            text="Waypoints",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(0, 0, tab_y),
            command=self._show_panel,
            extraArgs=["waypoint"],
            parent=self.main_frame
        )
        
        # Position control tab
        self.position_tab = DirectButton(
            text="Position",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(0.5, 0, tab_y),
            command=self._show_panel,
            extraArgs=["position"],
            parent=self.main_frame
        )
    
    def _create_manual_control(self):
        """Create the manual control panel."""
        self.manual_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0),
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )
        
        # Left motor voltage slider
        DirectLabel(
            text="Left Motor Voltage",
            scale=0.05,
            pos=(-0.5, 0, 0.25),
            parent=self.manual_panel
        )
        
        self.left_voltage_slider = DirectSlider(
            range=(-12, 12),
            value=0,
            pageSize=1,
            orientation="horizontal",
            command=self._set_left_voltage,
            pos=(-0.5, 0, 0.15),
            scale=0.5,
            parent=self.manual_panel
        )
        
        self.left_voltage_label = DirectLabel(
            text="0.0 V",
            scale=0.05,
            pos=(-0.5, 0, 0.05),
            parent=self.manual_panel
        )
        
        # Right motor voltage slider
        DirectLabel(
            text="Right Motor Voltage",
            scale=0.05,
            pos=(0.5, 0, 0.25),
            parent=self.manual_panel
        )
        
        self.right_voltage_slider = DirectSlider(
            range=(-12, 12),
            value=0,
            pageSize=1,
            orientation="horizontal",
            command=self._set_right_voltage,
            pos=(0.5, 0, 0.15),
            scale=0.5,
            parent=self.manual_panel
        )
        
        self.right_voltage_label = DirectLabel(
            text="0.0 V",
            scale=0.05,
            pos=(0.5, 0, 0.05),
            parent=self.manual_panel
        )
        
        # Quick command buttons
        buttons_y = -0.1
        button_width = 0.2
        
        # Forward button
        DirectButton(
            text="Forward",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(0, 0, buttons_y + 0.15),
            command=self._cmd_forward,
            parent=self.manual_panel
        )
        
        # Left button
        DirectButton(
            text="Left",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(-0.3, 0, buttons_y),
            command=self._cmd_left,
            parent=self.manual_panel
        )
        
        # Stop button
        DirectButton(
            text="Stop",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(0, 0, buttons_y),
            command=self._cmd_stop,
            parent=self.manual_panel
        )
        
        # Right button
        DirectButton(
            text="Right",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(0.3, 0, buttons_y),
            command=self._cmd_right,
            parent=self.manual_panel
        )
        
        # Backward button
        DirectButton(
            text="Backward",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(0, 0, buttons_y - 0.15),
            command=self._cmd_backward,
            parent=self.manual_panel
        )
        
        # Reset position button
        DirectButton(
            text="Reset Position",
            scale=0.05,
            frameSize=(-button_width, button_width, -0.05, 0.05),
            pos=(0, 0, -0.3),
            command=self._reset_position,
            parent=self.manual_panel
        )
    
    def _create_waypoint_control(self):
        """Create the waypoint control panel."""
        self.waypoint_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0),
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )
        
        # Add waypoint section
        DirectLabel(
            text="Add Waypoint",
            scale=0.05,
            pos=(0, 0, 0.25),
            parent=self.waypoint_panel
        )
        
        # X position entry
        DirectLabel(
            text="X:",
            scale=0.05,
            pos=(-0.6, 0, 0.15),
            parent=self.waypoint_panel
        )
        
        self.waypoint_x_entry = DirectEntry(
            scale=0.05,
            width=5,
            pos=(-0.5, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.waypoint_panel
        )
        
        # Y position entry
        DirectLabel(
            text="Y:",
            scale=0.05,
            pos=(-0.3, 0, 0.15),
            parent=self.waypoint_panel
        )
        
        self.waypoint_y_entry = DirectEntry(
            scale=0.05,
            width=5,
            pos=(-0.2, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.waypoint_panel
        )
        
        # Add waypoint button
        DirectButton(
            text="Add Waypoint",
            scale=0.05,
            frameSize=(-0.2, 0.2, -0.05, 0.05),
            pos=(0.1, 0, 0.15),
            command=self._add_waypoint,
            parent=self.waypoint_panel
        )
        
        # Add waypoint at click
        DirectLabel(
            text="Or click in the scene to add waypoints",
            scale=0.04,
            pos=(0, 0, 0.05),
            parent=self.waypoint_panel
        )
        
        # Start/stop following waypoints
        self.follow_button = DirectButton(
            text="Start Following",
            scale=0.05,
            frameSize=(-0.2, 0.2, -0.05, 0.05),
            pos=(-0.3, 0, -0.1),
            command=self._toggle_follow_waypoints,
            parent=self.waypoint_panel
        )
        
        # Clear waypoints button
        DirectButton(
            text="Clear Waypoints",
            scale=0.05,
            frameSize=(-0.2, 0.2, -0.05, 0.05),
            pos=(0.3, 0, -0.1),
            command=self._clear_waypoints,
            parent=self.waypoint_panel
        )
        
        # Waypoint planning algorithm selection
        DirectLabel(
            text="Planning Algorithm:",
            scale=0.05,
            pos=(-0.3, 0, -0.25),
            parent=self.waypoint_panel
        )
        
        self.planning_algo = DirectOptionMenu(
            scale=0.05,
            items=["Linear", "Smooth", "Pure Pursuit"],
            initialitem=0,
            command=self._set_planning_algorithm,
            pos=(0.3, 0, -0.25),
            parent=self.waypoint_panel
        )
        
        # Set up a click event for the screen to add waypoints
        self.app.accept("mouse1", self._on_screen_click)
    
    def _create_position_control(self):
        """Create the position control panel."""
        self.position_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0),
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )
        
        # Target position and orientation
        DirectLabel(
            text="Target Position and Orientation",
            scale=0.05,
            pos=(0, 0, 0.25),
            parent=self.position_panel
        )
        
        # X position entry
        DirectLabel(
            text="X:",
            scale=0.05,
            pos=(-0.6, 0, 0.15),
            parent=self.position_panel
        )
        
        self.target_x_entry = DirectEntry(
            scale=0.05,
            width=5,
            pos=(-0.5, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )
        
        # Y position entry
        DirectLabel(
            text="Y:",
            scale=0.05,
            pos=(-0.3, 0, 0.15),
            parent=self.position_panel
        )
        
        self.target_y_entry = DirectEntry(
            scale=0.05,
            width=5,
            pos=(-0.2, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )
        
        # Theta orientation entry
        DirectLabel(
            text="θ:",
            scale=0.05,
            pos=(0.0, 0, 0.15),
            parent=self.position_panel
        )
        
        self.target_theta_entry = DirectEntry(
            scale=0.05,
            width=5,
            pos=(0.1, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )
        
        # Set target button
        DirectButton(
            text="Set Target",
            scale=0.05,
            frameSize=(-0.15, 0.15, -0.05, 0.05),
            pos=(0.35, 0, 0.15),
            command=self._set_target_position,
            parent=self.position_panel
        )
        
        # Or click in the scene to set target position
        DirectLabel(
            text="Or click in the scene and press 'T' to set target",
            scale=0.04,
            pos=(0, 0, 0.05),
            parent=self.position_panel
        )
        
        # Start/stop position control
        self.position_control_button = DirectButton(
            text="Start Position Control",
            scale=0.05,
            frameSize=(-0.25, 0.25, -0.05, 0.05),
            pos=(-0.3, 0, -0.1),
            command=self._toggle_position_control,
            parent=self.position_panel
        )
        
        # Control algorithm selection
        DirectLabel(
            text="Control Algorithm:",
            scale=0.05,
            pos=(-0.3, 0, -0.25),
            parent=self.position_panel
        )
        
        self.control_algo = DirectOptionMenu(
            scale=0.05,
            items=["PID", "Pure Pursuit", "Feedback Linearization"],
            initialitem=0,
            command=self._set_control_algorithm,
            pos=(0.3, 0, -0.25),
            parent=self.position_panel
        )
        
        # Set up keyboard event to set target position
        self.app.accept("t", self._set_target_at_click)
    
    def _create_telemetry(self):
        """Create the telemetry display panel."""
        self.telemetry_frame = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0.8),
            frameSize=(-0.3, 0.3, -0.3, 0.3),
            pos=(0.85, 0, 0.4)
        )
        
        # Telemetry title
        DirectLabel(
            text="Vehicle Telemetry",
            scale=0.05,
            pos=(0, 0, 0.25),
            parent=self.telemetry_frame
        )
        
        # Position and orientation
        DirectLabel(
            text="Position:",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, 0.15),
            parent=self.telemetry_frame
        )
        
        self.telemetry_position = DirectLabel(
            text="X: 0.00  Y: 0.00",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, 0.10),
            parent=self.telemetry_frame
        )
        
        DirectLabel(
            text="Orientation:",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, 0.05),
            parent=self.telemetry_frame
        )
        
        self.telemetry_orientation = DirectLabel(
            text="θ: 0.00 degrees",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, 0.00),
            parent=self.telemetry_frame
        )
        
        # Velocities
        DirectLabel(
            text="Velocities:",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, -0.05),
            parent=self.telemetry_frame
        )
        
        self.telemetry_velocities = DirectLabel(
            text="Linear: 0.00 m/s\nAngular: 0.00 rad/s",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, -0.15),
            parent=self.telemetry_frame
        )
        
        # Motor info
        DirectLabel(
            text="Motors:",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, -0.20),
            parent=self.telemetry_frame
        )
        
        self.telemetry_motors = DirectLabel(
            text="Left: 0.00 V, 0.00 rad/s\nRight: 0.00 V, 0.00 rad/s",
            scale=0.04,
            align=TextNode.ALeft,
            pos=(-0.25, 0, -0.30),
            parent=self.telemetry_frame
        )
    
    def _show_panel(self, panel_name):
        """Show the selected panel and hide others."""
        self.control_mode = panel_name
        
        # Hide all panels
        self.manual_panel.hide()
        self.waypoint_panel.hide()
        self.position_panel.hide()
        
        # Show the selected panel
        if panel_name == "manual":
            self.manual_panel.show()
            # Reset control modes
            self._cmd_stop()
            self.path_planner.stop_following()
            self.position_controller.stop_control()
        elif panel_name == "waypoint":
            self.waypoint_panel.show()
            # Reset control modes
            self._cmd_stop()
            self.position_controller.stop_control()
        elif panel_name == "position":
            self.position_panel.show()
            # Reset control modes
            self._cmd_stop()
            self.path_planner.stop_following()
    
    def _set_left_voltage(self):
        """Set the left motor voltage from the slider."""
        self.left_voltage = self.left_voltage_slider["value"]
        self.left_voltage_label["text"] = f"{self.left_voltage:.1f} V"
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _set_right_voltage(self):
        """Set the right motor voltage from the slider."""
        self.right_voltage = self.right_voltage_slider["value"]
        self.right_voltage_label["text"] = f"{self.right_voltage:.1f} V"
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _cmd_forward(self):
        """Command to move forward."""
        self.left_voltage = 6.0
        self.right_voltage = 6.0
        self.left_voltage_slider.setValue(self.left_voltage)
        self.right_voltage_slider.setValue(self.right_voltage)
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _cmd_backward(self):
        """Command to move backward."""
        self.left_voltage = -6.0
        self.right_voltage = -6.0
        self.left_voltage_slider.setValue(self.left_voltage)
        self.right_voltage_slider.setValue(self.right_voltage)
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _cmd_left(self):
        """Command to turn left."""
        self.left_voltage = -3.0
        self.right_voltage = 3.0
        self.left_voltage_slider.setValue(self.left_voltage)
        self.right_voltage_slider.setValue(self.right_voltage)
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _cmd_right(self):
        """Command to turn right."""
        self.left_voltage = 3.0
        self.right_voltage = -3.0
        self.left_voltage_slider.setValue(self.left_voltage)
        self.right_voltage_slider.setValue(self.right_voltage)
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _cmd_stop(self):
        """Command to stop."""
        self.left_voltage = 0.0
        self.right_voltage = 0.0
        self.left_voltage_slider.setValue(self.left_voltage)
        self.right_voltage_slider.setValue(self.right_voltage)
        self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
    
    def _reset_position(self):
        """Reset the vehicle position to the origin."""
        self.vehicle.set_position(0.0, 0.0, 0.0)
        self.simulation.clear_path()
    
    def _add_waypoint(self):
        """Add a waypoint from the entry fields."""
        try:
            x = float(self.waypoint_x_entry.get())
            y = float(self.waypoint_y_entry.get())
            self.path_planner.add_waypoint(x, y)
            self.simulation.add_waypoint(x, y)
        except ValueError:
            print("Invalid waypoint coordinates")
    
    def _on_screen_click(self):
        """Handle screen click to add waypoint or set target position."""
        if not hasattr(self.app, "mouseWatcherNode") or not self.app.mouseWatcherNode.hasMouse():
            return
        
        # Get the mouse position
        mouse_pos = self.app.mouseWatcherNode.getMouse()
        
        # Create a ray from the camera
        pMouse = self.app.camLens.getProjectionMat().xform(Vec3(mouse_pos[0], 0, mouse_pos[1]))
        nearPoint = Point3(pMouse[0], -1, pMouse[2])
        farPoint = Point3(pMouse[0], 1, pMouse[2])
        
        nearPoint = self.app.render.getRelativePoint(self.app.cam, nearPoint)
        farPoint = self.app.render.getRelativePoint(self.app.cam, farPoint)
        
        # Project onto the z=0 plane (ground)
        if abs(farPoint.z - nearPoint.z) > 0.001:
            t = -nearPoint.z / (farPoint.z - nearPoint.z)
            x = nearPoint.x + t * (farPoint.x - nearPoint.x)
            y = nearPoint.y + t * (farPoint.y - nearPoint.y)
            
            # If in waypoint mode, add a waypoint
            if self.control_mode == "waypoint":
                self.path_planner.add_waypoint(x, y)
                self.simulation.add_waypoint(x, y)
                print(f"Added waypoint at ({x:.2f}, {y:.2f})")
            
            # Store click position for target setting
            self.click_pos = (x, y)
    
    def _set_target_at_click(self):
        """Set the target position at the last click position."""
        if self.control_mode == "position" and hasattr(self, "click_pos"):
            x, y = self.click_pos
            # Use current orientation as default
            theta = float(self.target_theta_entry.get())
            
            self.target_x_entry.enterText(f"{x:.2f}")
            self.target_y_entry.enterText(f"{y:.2f}")
            
            self._set_target_position()
    
    def _set_target_position(self):
        """Set the target position and orientation from the entry fields."""
        try:
            x = float(self.target_x_entry.get())
            y = float(self.target_y_entry.get())
            theta = float(self.target_theta_entry.get())
            
            # Convert from degrees to radians if necessary
            if abs(theta) > np.pi * 2:
                theta = np.radians(theta)
            
            self.target_x = x
            self.target_y = y
            self.target_theta = theta
            
            # Update the position controller
            self.position_controller.set_target(x, y, theta)
            
            # Visualize the target
            print(f"Set target position to ({x:.2f}, {y:.2f}, {np.degrees(theta):.2f}°)")
            
            # Add a distinctive marker (we would use simulation.add_target if it existed)
            self.simulation.add_waypoint(x, y)
        except ValueError:
            print("Invalid target position or orientation")
    
    def _toggle_follow_waypoints(self):
        """Toggle following waypoints."""
        if self.path_planner.is_following():
            self.path_planner.stop_following()
            self.follow_button["text"] = "Start Following"
        else:
            self.path_planner.start_following()
            self.follow_button["text"] = "Stop Following"
    
    def _clear_waypoints(self):
        """Clear all waypoints."""
        self.path_planner.clear_waypoints()
        self.simulation.clear_waypoints()
    
    def _set_planning_algorithm(self, algorithm):
        """Set the path planning algorithm."""
        self.path_planner.set_algorithm(algorithm)
    
    def _toggle_position_control(self):
        """Toggle position control."""
        if self.position_controller.is_controlling():
            self.position_controller.stop_control()
            self.position_control_button["text"] = "Start Position Control"
        else:
            self.position_controller.start_control()
            self.position_control_button["text"] = "Stop Position Control"
    
    def _set_control_algorithm(self, algorithm):
        """Set the position control algorithm."""
        self.position_controller.set_algorithm(algorithm)
    
    def update(self, dt):
        """
        Update the UI for one time step.
        
        Args:
            dt (float): Time step in seconds
        """
        # Update telemetry display
        pos = self.vehicle.get_position()
        vel = self.vehicle.get_velocity()
        wheel_speeds = self.vehicle.get_wheel_speeds()
        
        self.telemetry_position["text"] = f"X: {pos[0]:.2f}  Y: {pos[1]:.2f}"
        self.telemetry_orientation["text"] = f"θ: {np.degrees(pos[2]):.2f} degrees"
        self.telemetry_velocities["text"] = f"Linear: {vel[0]:.2f} m/s\nAngular: {vel[1]:.2f} rad/s"
        self.telemetry_motors["text"] = f"Left: {self.left_voltage:.2f} V, {wheel_speeds[0]:.2f} rad/s\n" \
                                        f"Right: {self.right_voltage:.2f} V, {wheel_speeds[1]:.2f} rad/s"
        
        # Update controllers based on active mode
        if self.control_mode == "waypoint" and self.path_planner.is_following():
            # Path planner handles the vehicle control
            voltages = self.path_planner.update(dt, self.vehicle)
            if voltages:
                left_v, right_v = voltages
                self.left_voltage = left_v
                self.right_voltage = right_v
                self.left_voltage_slider.setValue(self.left_voltage)
                self.right_voltage_slider.setValue(self.right_voltage)
        
        elif self.control_mode == "position" and self.position_controller.is_controlling():
            # Position controller handles the vehicle control
            voltages = self.position_controller.update(dt)
            if voltages:
                left_v, right_v = voltages
                self.left_voltage = left_v
                self.right_voltage = right_v
                self.left_voltage_slider.setValue(self.left_voltage)
                self.right_voltage_slider.setValue(self.right_voltage)
