"""
Position Controller module for differential vehicle simulation.
This module implements control algorithms to drive the vehicle to a target position and orientation.
"""

import numpy as np

class PositionController:
    """
    Position controller for the differential drive vehicle.
    Implements control algorithms to drive the vehicle to a specified position and orientation.
    """
    
    def __init__(self, vehicle):
        """
        Initialize the position controller.
        
        Args:
            vehicle: The vehicle model
        """
        self.vehicle = vehicle
        
        # Target position and orientation
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        
        # Control algorithm and parameters
        self.algorithm = "PID"  # "PID", "Pure Pursuit", "Feedback Linearization"
        
        # PID controllers for position and orientation
        self.position_pid = PIDController(kp=5.0, ki=0.1, kd=1.0, max_output=10.0)
        self.orientation_pid = PIDController(kp=8.0, ki=0.1, kd=1.0, max_output=10.0)
        
        # State
        self.controlling = False
        
        # Control parameters
        self.distance_tolerance = 0.05  # meters
        self.angle_tolerance = np.radians(3.0)  # radians
    
    def set_target(self, x, y, theta):
        """
        Set the target position and orientation.
        
        Args:
            x (float): Target x position
            y (float): Target y position
            theta (float): Target orientation in radians
        """
        self.target_x = x
        self.target_y = y
        self.target_theta = theta
        
        # Reset controller states
        self.position_pid.reset()
        self.orientation_pid.reset()
        
        print(f"Set target to ({x:.2f}, {y:.2f}, {np.degrees(theta):.2f}°)")
    
    def start_control(self):
        """Start position control."""
        self.controlling = True
        print(f"Started position control using {self.algorithm} algorithm")
    
    def stop_control(self):
        """Stop position control."""
        self.controlling = False
        print("Stopped position control")
    
    def is_controlling(self):
        """Check if the controller is active."""
        return self.controlling
    
    def set_algorithm(self, algorithm):
        """
        Set the control algorithm.
        
        Args:
            algorithm (str): Algorithm name ('PID', 'Pure Pursuit', 'Feedback Linearization')
        """
        self.algorithm = algorithm
        print(f"Set position control algorithm to {algorithm}")
    
    def update(self, dt):
        """
        Update the position controller and calculate control signals.
        
        Args:
            dt (float): Time step in seconds
            
        Returns:
            tuple: (left_voltage, right_voltage) or None if not controlling
        """
        if not self.controlling:
            return None
        
        # Get vehicle position and orientation
        x, y, theta = self.vehicle.get_position()
        
        # Check if we've reached the target position and orientation
        distance_to_target = np.sqrt((self.target_x - x)**2 + (self.target_y - y)**2)
        angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
        
        if distance_to_target < self.distance_tolerance and abs(angle_error) < self.angle_tolerance:
            print("Reached target position and orientation")
            self.controlling = False
            return (0.0, 0.0)  # Stop the vehicle
        
        # Calculate control signals based on selected algorithm
        if self.algorithm == "PID":
            return self._pid_control(x, y, theta, dt)
        elif self.algorithm == "Pure Pursuit":
            return self._pure_pursuit_control(x, y, theta, dt)
        elif self.algorithm == "Feedback Linearization":
            return self._feedback_linearization_control(x, y, theta, dt)
        else:
            # Default to PID control
            return self._pid_control(x, y, theta, dt)
    
    def _pid_control(self, x, y, theta, dt):
        """
        PID control for position and orientation.
        Simplified approach: first orient toward the target, then move forward.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        # Calculate angle to target
        desired_angle = np.arctan2(self.target_y - y, self.target_x - x)
        
        # Calculate angle error (-pi to pi)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        
        # Calculate distance to target
        distance = np.sqrt((self.target_x - x)**2 + (self.target_y - y)**2)
        
        # Two-phase control: first orient, then move
        if abs(angle_error) > self.angle_tolerance and distance > self.distance_tolerance:
            # Phase 1: Orient toward the target first
            angular_speed = self.orientation_pid.update(angle_error, dt)
            linear_speed = 0.0
        else:
            # Phase 2: Move toward the target
            angular_speed = self.orientation_pid.update(angle_error, dt)
            linear_speed = self.position_pid.update(distance, dt)
            
            # Slow down when getting close to the target
            slow_distance = 0.5  # meters
            if distance < slow_distance:
                linear_speed *= distance / slow_distance
        
        # If close to target position but orientation is off, adjust orientation
        if distance < self.distance_tolerance:
            # Calculate angle error to target orientation
            final_angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
            angular_speed = self.orientation_pid.update(final_angle_error, dt)
            linear_speed = 0.0
        
        # Convert to wheel voltages
        left_voltage = linear_speed - angular_speed
        right_voltage = linear_speed + angular_speed
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)
    
    def _pure_pursuit_control(self, x, y, theta, dt):
        """
        Pure pursuit control to reach the target position.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        # Vector to target point in global frame
        dx = self.target_x - x
        dy = self.target_y - y
        
        # Rotate to vehicle frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        target_x_local = dx * cos_theta + dy * sin_theta
        target_y_local = -dx * sin_theta + dy * cos_theta
        
        # Calculate lookahead distance and angle
        lookahead_distance = np.sqrt(dx**2 + dy**2)
        lookahead_angle = np.arctan2(target_y_local, target_x_local)
        
        # Calculate curvature for pure pursuit
        if abs(target_y_local) < 1e-6 or lookahead_distance < 1e-6:
            curvature = 0
        else:
            curvature = 2 * target_y_local / (lookahead_distance**2)
        
        # Calculate wheel speeds based on curvature
        wheel_distance = self.vehicle.wheel_distance
        
        # Base speed depends on distance to target (slow down when close)
        max_speed = 8.0
        min_speed = 2.0
        slow_distance = 1.0  # meters
        
        if lookahead_distance < self.distance_tolerance:
            # Reached position target, adjust orientation
            return self._orientation_control(theta, dt)
        elif lookahead_distance < slow_distance:
            # Slow down as we approach the target
            base_speed = min_speed + (max_speed - min_speed) * (lookahead_distance / slow_distance)
        else:
            base_speed = max_speed
        
        # Adjust speeds based on curvature
        if abs(curvature) < 1e-6:
            # Straight line
            left_voltage = base_speed
            right_voltage = base_speed
        else:
            # Turning
            radius = 1.0 / abs(curvature)
            speed_ratio = (radius - wheel_distance/2) / (radius + wheel_distance/2)
            
            if curvature > 0:
                # Turn left
                left_voltage = base_speed * speed_ratio
                right_voltage = base_speed
            else:
                # Turn right
                left_voltage = base_speed
                right_voltage = base_speed * speed_ratio
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)
    
    def _feedback_linearization_control(self, x, y, theta, dt):
        """
        Feedback linearization control for position and orientation.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        # Calculate error in global frame
        dx = self.target_x - x
        dy = self.target_y - y
        
        # Rotate error to vehicle frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        error_x_local = dx * cos_theta + dy * sin_theta
        error_y_local = -dx * sin_theta + dy * cos_theta
        
        # Control gains
        k_rho = 3.0  # Position gain
        k_alpha = 8.0  # Heading gain
        k_beta = -1.5  # Orientation gain
        
        # Calculate polar coordinates of error
        rho = np.sqrt(error_x_local**2 + error_y_local**2)
        alpha = np.arctan2(error_y_local, error_x_local)
        beta = -theta - alpha
        
        # Calculate control inputs
        if rho < self.distance_tolerance:
            # Reached position target, adjust orientation
            return self._orientation_control(theta, dt)
        
        v = k_rho * rho  # Linear velocity
        w = k_alpha * alpha + k_beta * beta  # Angular velocity
        
        # Convert to wheel speeds
        wheel_radius = self.vehicle.wheel_radius
        wheel_distance = self.vehicle.wheel_distance
        
        left_speed = (v - w * wheel_distance / 2) / wheel_radius
        right_speed = (v + w * wheel_distance / 2) / wheel_radius
        
        # Convert speeds to voltages (simplified model)
        motor_k = 0.5  # Motor constant (V/(rad/s))
        left_voltage = left_speed * motor_k
        right_voltage = right_speed * motor_k
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)
    
    def _orientation_control(self, theta, dt):
        """
        Control to achieve the target orientation.
        
        Args:
            theta: Current vehicle orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        # Calculate angle error to target orientation
        angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
        
        # Use PID controller for orientation
        angular_speed = self.orientation_pid.update(angle_error, dt)
        
        # Convert to wheel voltages
        left_voltage = -angular_speed
        right_voltage = angular_speed
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)


class PIDController:
    """Simple PID controller."""
    
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=float('inf'), min_output=-float('inf')):
        """
        Initialize the PID controller.
        
        Args:
            kp (float): Proportional gain
            ki (float): Integral gain
            kd (float): Derivative gain
            max_output (float): Maximum output value
            min_output (float): Minimum output value
        """
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        
        self.prev_error = 0.0
        self.integral = 0.0
    
    def update(self, error, dt):
        """
        Update the PID controller and calculate the output.
        
        Args:
            error (float): Current error value
            dt (float): Time step in seconds
            
        Returns:
            float: Controller output
        """
        # Proportional term
        p_term = self.kp * error
        
        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral
        
        # Derivative term
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        
        # Calculate output
        output = p_term + i_term + d_term
        
        # Apply output limits
        output = min(self.max_output, max(self.min_output, output))
        
        # Store current error for next iteration
        self.prev_error = error
        
        return output
    
    def reset(self):
        """Reset the controller state."""
        self.prev_error = 0.0
        self.integral = 0.0
