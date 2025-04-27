import numpy as np
from direct.gui.DirectGui import DirectButton, DirectFrame, DirectLabel, DirectEntry
from direct.gui.DirectGui import DirectSlider, DirectOptionMenu
from direct.gui.OnscreenText import OnscreenText
from panda3d.core import TextNode, Vec3, Point3

class UIController:
    def __init__(self, app, vehicle, simulation, path_planner, position_controller):
        self.app = app
        self.vehicle = vehicle
        self.simulation = simulation
        self.path_planner = path_planner
        self.position_controller = position_controller
        self.control_mode = "manual"
        self.left_voltage = 0.0
        self.right_voltage = 0.0
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        self.setup_keyboard_controls()
        self._create_ui()

    def _create_ui(self):
        self.main_frame = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0.8),
            frameSize=(-0.8, 0.8, -0.4, 0.4),
            pos=(0, 0, -0.5)
        )
        self._create_tab_buttons()
        self._create_manual_control()
        self._create_waypoint_control()
        self._create_position_control()
        self._create_model_visualization()
        self._create_telemetry()
        self._show_panel("manual")

    def _create_tab_buttons(self):
        tab_width = 0.2
        tab_height = 0.05
        tab_y = 0.4

        # Manual control tab
        DirectButton(
            text="Manual Control",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(-0.75, 0, tab_y),
            command=self._show_panel,
            extraArgs=["manual"],
            parent=self.main_frame
        )

        # Model visualization tab
        DirectButton(
            text="Model Visualization",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(-0.25, 0, tab_y),
            command=self._show_panel,
            extraArgs=["model"],
            parent=self.main_frame
        )

        # Waypoint control tab
        DirectButton(
            text="Waypoints",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(0.25, 0, tab_y),
            command=self._show_panel,
            extraArgs=["waypoint"],
            parent=self.main_frame
        )

        # Position control tab
        DirectButton(
            text="Position",
            scale=0.05,
            frameSize=(-tab_width, tab_width, -tab_height, tab_height),
            pos=(0.75, 0, tab_y),
            command=self._show_panel,
            extraArgs=["position"],
            parent=self.main_frame
        )


    def _create_manual_control(self):
        """Create the manual control panel."""
        # Crear primero el panel manual
        self.manual_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0), 
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )

        # Organizar los botones en cruz
        button_layout = {
            'forward': (0, 0.15),      # Arriba
            'left': (-0.3, 0),         # Izquierda  
            'stop': (0, 0),            # Centro
            'right': (0.3, 0),         # Derecha
            'backward': (0, -0.15),    # Abajo
            'reset': (0, -0.3)         # Abajo del todo
        }
        
        button_style = {
            'scale': 0.07,
            'frameSize': (-0.2, 0.2, -0.07, 0.07),
            'relief': 1,
            'pressEffect': 1
        }

        # Crear los botones usando el layout
        for name, pos in button_layout.items():
            button = DirectButton(
                text=name.capitalize(),
                pos=(pos[0], 0, pos[1]),
                command=getattr(self, f'_cmd_{name}'),
                parent=self.manual_panel,
                **button_style
            )

        # Agregar sliders de voltaje
        self.left_voltage_slider = DirectSlider(
            range=(-12, 12),
            value=0,
            pageSize=1,
            command=self._set_left_voltage,
            pos=(-0.5, 0, -0.2),
            scale=0.3,
            parent=self.manual_panel
        )

        self.right_voltage_slider = DirectSlider(
            range=(-12, 12),
            value=0,
            pageSize=1,
            command=self._set_right_voltage,
            pos=(0.5, 0, -0.2),
            scale=0.3,
            parent=self.manual_panel
        )

        # Etiquetas de voltaje
        self.left_voltage_label = DirectLabel(
            text="0.0 V",
            scale=0.05,
            pos=(-0.5, 0, -0.25),
            parent=self.manual_panel
        )

        self.right_voltage_label = DirectLabel(
            text="0.0 V", 
            scale=0.05,
            pos=(0.5, 0, -0.25),
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

        # Coordenadas entrada manual
        DirectLabel(
            text="Manual Waypoint Entry",
            scale=0.05,
            pos=(0, 0, 0.25),
            parent=self.waypoint_panel
        )

        # Panel izquierdo - Entrada manual
        left_panel = DirectFrame(
            frameColor=(0, 0, 0, 0),
            pos=(-0.35, 0, 0),
            parent=self.waypoint_panel
        )

        DirectLabel(
            text="X:", scale=0.05,
            pos=(-0.1, 0, 0.15),
            parent=left_panel
        )

        self.waypoint_x_entry = DirectEntry(
            scale=0.07,
            width=7,
            pos=(0.05, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=left_panel
        )

        DirectLabel(
            text="Y:", scale=0.05,
            pos=(-0.1, 0, 0.05),
            parent=left_panel
        )

        self.waypoint_y_entry = DirectEntry(
            scale=0.07,
            width=7,
            pos=(0.05, 0, 0.05),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=left_panel
        )

        # Panel derecho - Botones de control
        right_panel = DirectFrame(
            frameColor=(0, 0, 0, 0),
            pos=(0.35, 0, 0),
            parent=self.waypoint_panel
        )

        self.add_waypoint_button = DirectButton(
            text="Add Waypoint",
            scale=0.07,
            pos=(0, 0, 0.15),
            command=self._add_waypoint,
            parent=right_panel
        )

        self.follow_button = DirectButton(
            text="Start Following",
            scale=0.07,
            pos=(0, 0, 0),
            command=self._toggle_follow_waypoints,
            parent=right_panel
        )

        self.clear_waypoints_button = DirectButton(
            text="Clear Waypoints",
            scale=0.07,
            pos=(0, 0, -0.15),
            command=self._clear_waypoints,
            parent=right_panel
        )

        # Panel inferior - Algoritmo y mensajes
        bottom_panel = DirectFrame(
            frameColor=(0, 0, 0, 0),
            pos=(0, 0, -0.25),
            parent=self.waypoint_panel
        )

        DirectLabel(
            text="Planning Algorithm:",
            scale=0.05,
            pos=(-0.2, 0, 0),
            parent=bottom_panel
        )

        self.planning_algo = DirectOptionMenu(
            scale=0.05,
            items=["Linear", "Smooth", "Pure Pursuit"],
            initialitem=0,
            command=self._set_planning_algorithm,
            pos=(0.2, 0, 0),
            parent=bottom_panel
        )


        DirectLabel(
            text="Click in scene: Ctrl + Click to add waypoint",
            scale=0.04,
            pos=(0, 0, -0.1),
            parent=bottom_panel
        )

        self.follow_button = DirectButton(
            text="Start Following",
            scale=0.07,
            frameSize=(-0.2, 0.2, -0.07, 0.07),
            pos=(-0.3, 0, -0.1),
            relief=1,  # Raised relief
            command=self._toggle_follow_waypoints,
            parent=self.waypoint_panel,
            pressEffect=1
        )

        # Clear waypoints button
        self.clear_waypoints_button = DirectButton(
            text="Clear Waypoints",
            scale=0.07,
            frameSize=(-0.2, 0.2, -0.07, 0.07),
            pos=(0.3, 0, -0.1),
            relief=1,  # Raised relief
            command=self._clear_waypoints,
            parent=self.waypoint_panel,
            pressEffect=1
        )

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

        self.app.accept("mouse1", self._on_screen_click)

    def _create_position_control(self):
        """Create the position control panel."""
        self.position_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0),
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )

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
            scale=0.07,
            width=7,
            pos=(-0.5, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )

        DirectLabel(
            text="Y:",
            scale=0.05,
            pos=(-0.3, 0, 0.15),
            parent=self.position_panel
        )

        self.target_y_entry = DirectEntry(
            scale=0.07,
            width=7,
            pos=(-0.2, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )

        DirectLabel(
            text="Theta:",  # Cambiado de θ a Theta
            scale=0.05,
            pos=(0.0, 0, 0.15),
            parent=self.position_panel
        )

        self.target_theta_entry = DirectEntry(
            scale=0.07,
            width=7,
            pos=(0.1, 0, 0.15),
            initialText="0.0",
            numLines=1,
            focus=0,
            parent=self.position_panel
        )

        # Set target button
        button_style = {
            'scale': 0.05,
            'frameSize': (-0.4, 0.4, -0.1, 0.1),
            'relief': 1,
            'pressEffect': 1
        }
        self.set_target_button = DirectButton(
            text="Set Target",
            **button_style,
            pos=(0.39, 0, 0.15),
            command=self._set_target_position,
            parent=self.position_panel,
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
            **button_style,
            pos=(-0.3, 0, -0.1),
            command=self._toggle_position_control,
            parent=self.position_panel,
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

    def _create_model_visualization(self):
        """Create the model visualization panel."""
        self.model_panel = DirectFrame(
            frameColor=(0.2, 0.2, 0.2, 0),
            frameSize=(-0.75, 0.75, -0.35, 0.35),
            pos=(0, 0, 0),
            parent=self.main_frame
        )

        # Kinematic model button
        self.kinematic_button = DirectButton(
            text="Show Kinematic Model",
            scale=0.05,
            frameSize=(-0.4, 0.4, -0.1, 0.1),
            pos=(-0.3, 0, 0.2),
            relief=1,
            command=lambda: self.vehicle.plot_kinematic_model(),
            parent=self.model_panel,
            pressEffect=1
        )

        # Dynamic model button
        self.dynamic_button = DirectButton(
            text="Show Dynamic Model",
            scale=0.05,
            frameSize=(-0.4, 0.4, -0.1, 0.1),
            pos=(0.3, 0, 0.2),
            relief=1,
            command=lambda: self.vehicle.plot_dynamic_model(),
            parent=self.model_panel,
            pressEffect=1
        )

        # Kinematic Model Equations
        DirectLabel(
            text="Kinematic Model Equations:",
            scale=0.05,
            pos=(-0.3, 0, 0.1),
            parent=self.model_panel,
            text_align=TextNode.ALeft
        )

        DirectLabel(
            text=(
                "v = (vr + vl)/2         - Linear velocity\n"
                "w = (vr - vl)/L         - Angular velocity\n"
                "dx = v*cos(theta)       - X velocity\n"
                "dy = v*sin(theta)       - Y velocity\n"
                "dtheta = w              - Angular velocity"
            ),
            scale=0.04,
            pos=(-0.3, 0, -0.05),
            parent=self.model_panel,
            text_align=TextNode.ALeft
        )

        # Dynamic Model Equations
        DirectLabel(
            text="Dynamic Model Equations:",
            scale=0.05,
            pos=(0.3, 0, 0.1),
            parent=self.model_panel,
            text_align=TextNode.ALeft
        )

        DirectLabel(
            text=(
                "τm = Kt·i               - Motor torque\n"
                "i = (V - Ke·ω)/R        - Motor current\n"
                "J·ω̇ = τm - τf           - Angular dynamics\n"
                "m·v̇ = F - Fr            - Linear dynamics\n"
                "Fr = μ·m·g              - Rolling friction"
            ),
            scale=0.04,
            pos=(0.3, 0, -0.05),
            parent=self.model_panel,
            text_align=TextNode.ALeft
        )


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
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, 0.15),
            parent=self.telemetry_frame
        )

        self.telemetry_position = DirectLabel(
            text="X: 0.00  Y: 0.00",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, 0.10),
            parent=self.telemetry_frame
        )

        DirectLabel(
            text="Orientation:",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, 0.05),
            parent=self.telemetry_frame
        )

        self.telemetry_orientation = DirectLabel(
            text="Theta: 0.00 degrees",  # Cambiado de θ a Theta
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, 0.00),
            parent=self.telemetry_frame
        )

        # Velocities
        DirectLabel(
            text="Velocities:",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, -0.05),
            parent=self.telemetry_frame
        )

        self.telemetry_velocities = DirectLabel(
            text="Linear: 0.00 m/s\nAngular: 0.00 rad/s",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, -0.15),
            parent=self.telemetry_frame
        )

        # Motor info
        DirectLabel(
            text="Motors:",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, -0.20),
            parent=self.telemetry_frame
        )

        self.telemetry_motors = DirectLabel(
            text="Left: 0.00 V, 0.00 rad/s\nRight: 0.00 V, 0.00 rad/s",
            scale=0.04,
            text_align=TextNode.ALeft,
            pos=(-0.25, 0, -0.30),
            parent=self.telemetry_frame
        )

    def _show_panel(self, panel_name):
        """Show the specified panel and hide others."""
        self.manual_panel.hide()
        self.waypoint_panel.hide()
        self.position_panel.hide()
        self.model_panel.hide()

        # Actualizar el modo de control
        if panel_name == "manual":
            self.manual_panel.show()
            self.control_mode = "manual"
        elif panel_name == "waypoint":
            self.waypoint_panel.show()
            self.control_mode = "waypoint"
        elif panel_name == "position":
            self.position_panel.show()
            self.control_mode = "position"
        elif panel_name == "model":
            self.model_panel.show()

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

    def _cmd_reset(self):
        """Comando para reiniciar la posición del vehículo."""
        self.vehicle.reset()  
        self.simulation.clear_path()
        self._cmd_stop()  # Detener los motores
        self.path_planner.clear_waypoints()
        self.simulation.clear_waypoints()
        print("Vehicle position reset to origin")

    def setup_keyboard_controls(self):
        """Setup keyboard controls for the vehicle."""
        # Cambiar teclas de cambio de panel a Alt + número
        self.app.accept("alt-1", lambda: self._show_panel("manual"))     
        self.app.accept("alt-2", lambda: self._show_panel("waypoint"))   
        self.app.accept("alt-3", lambda: self._show_panel("position"))   
        self.app.accept("alt-4", lambda: self._show_panel("model"))      

        # Resto de controles igual
        self.app.accept("arrow_up", self._cmd_forward)        
        self.app.accept("arrow_down", self._cmd_backward)     
        self.app.accept("arrow_left", self._cmd_left)         
        self.app.accept("arrow_right", self._cmd_right)       
        self.app.accept("space", self._cmd_stop)
        
        # Soltar teclas detiene el movimiento
        self.app.accept("arrow_up-up", self._cmd_stop)
        self.app.accept("arrow_down-up", self._cmd_stop)
        self.app.accept("arrow_left-up", self._cmd_stop) 
        self.app.accept("arrow_right-up", self._cmd_stop)

        # Cambio entre paneles (números)
        self.app.accept("1", lambda: self._show_panel("manual"))      # Panel manual
        self.app.accept("2", lambda: self._show_panel("waypoint"))    # Panel waypoints
        self.app.accept("3", lambda: self._show_panel("position"))    # Panel posición
        self.app.accept("4", lambda: self._show_panel("model"))       # Panel modelo

        # Controles de waypoints
        self.app.accept("q", self._toggle_follow_waypoints)   # Alternar seguimiento
        self.app.accept("e", self._clear_waypoints)          # Borrar waypoints
        self.app.accept("mouse1", self._on_screen_click)     # Click para agregar waypoint
        
        # Controles de posición
        self.app.accept("t", self._set_target_at_click)      # Establecer objetivo en click
        self.app.accept("p", self._toggle_position_control)   # Alternar control de posición

        # Otros controles
        self.app.accept("r", self._reset_position)           # Reiniciar posición
        self.app.accept("m", self.vehicle.plot_kinematic_model)  # Mostrar modelo cinemático
        self.app.accept("n", self.vehicle.plot_dynamic_model)    # Mostrar modelo dinámico

    def _reset_position(self):
        """Reset the vehicle position to the origin."""
        self.vehicle.set_position(0.0, 0.0, 0.0)
        self.simulation.clear_path()

    def _add_waypoint(self):
        try:
            x = float(self.waypoint_x_entry.get())
            y = float(self.waypoint_y_entry.get())
            self.path_planner.add_waypoint(x, y)
            self.simulation.add_waypoint(x, y)
            print(f"Waypoint added: ({x:.2f}, {y:.2f})")
            
            # Iniciar seguimiento si hay waypoints
            if not self.path_planner.is_following():
                self.path_planner.start_following()
                self.follow_button["text"] = "Stop Following"
        except ValueError:
            print("Invalid waypoint coordinates")

    def _on_screen_click(self):
        """Handle screen click to add waypoint or set target position."""
        if not self.app.mouseWatcherNode.hasMouse():
            return

        # Solo procesar si Ctrl está presionado
        if not self.app.mouseWatcherNode.isButtonDown("control"):
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
        try:
            x = float(self.target_x_entry.get())
            y = float(self.target_y_entry.get())
            theta = float(self.target_theta_entry.get())
            
            if abs(theta) > np.pi * 2:
                theta = np.radians(theta)
                
            self.position_controller.set_target(x, y, theta)
            self.position_controller.start_control()
            self.position_control_button["text"] = "Stop Position Control"
            print(f"Target set: ({x:.2f}, {y:.2f}, {np.degrees(theta):.2f}°)")
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
        pos = self.vehicle.get_position()
        vel = self.vehicle.get_velocity()
        wheel_speeds = self.vehicle.get_wheel_speeds()

        # Actualizar telemetría
        self.telemetry_position["text"] = f"X: {pos[0]:.2f}  Y: {pos[1]:.2f}"
        self.telemetry_orientation["text"] = f"Theta: {np.degrees(pos[2]):.2f} degrees"  # Cambiado de θ a Theta
        self.telemetry_velocities["text"] = f"Linear: {vel[0]:.2f} m/s\nAngular: {vel[1]:.2f} rad/s"
        self.telemetry_motors["text"] = f"Left: {self.left_voltage:.2f} V, {wheel_speeds[0]:.2f} rad/s\n" \
                                        f"Right: {self.right_voltage:.2f} V, {wheel_speeds[1]:.2f} rad/s"

        # Actualizar control basado en el modo activo
        if self.control_mode == "waypoint" and self.path_planner.is_following():
            voltages = self.path_planner.update(dt, self.vehicle)
            if voltages:
                self.left_voltage, self.right_voltage = voltages
                self.left_voltage_slider.setValue(self.left_voltage)
                self.right_voltage_slider.setValue(self.right_voltage)
                # Aplicar voltajes al vehículo
                self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)

        elif self.control_mode == "position" and self.position_controller.is_controlling():
            voltages = self.position_controller.update(dt)
            if voltages:
                self.left_voltage, self.right_voltage = voltages
                self.left_voltage_slider.setValue(self.left_voltage)
                self.right_voltage_slider.setValue(self.right_voltage)
                # Aplicar voltajes al vehículo
                self.vehicle.set_motor_voltages(self.left_voltage, self.right_voltage)
