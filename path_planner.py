"""
Path Planner module for differential vehicle simulation.
This module handles path planning and following waypoints.
"""

import numpy as np
from collections import deque

class PathPlanner:
    """
    Path planner for the differential vehicle.
    Handles waypoint management and path following algorithms.
    """
    
    def __init__(self):
        """Initialize the path planner."""
        # List of waypoints (x, y)
        self.waypoints = deque()
        
        # Current target waypoint
        self.current_waypoint = None
        
        # Path following parameters
        self.lookahead_distance = 0.5  # meters
        self.waypoint_tolerance = 0.1  # meters
        self.algorithm = "Linear"  # Default algorithm
        
        # State
        self.following = False
        
        # PID controller for path following
        self.linear_pid = PIDController(kp=5.0, ki=0.0, kd=1.0, max_output=10.0)
        self.angular_pid = PIDController(kp=8.0, ki=0.1, kd=1.0, max_output=10.0)
    
    def add_waypoint(self, x, y):
        """
        Add a waypoint to the path.
        
        Args:
            x (float): X coordinate
            y (float): Y coordinate
        """
        self.waypoints.append((x, y))
        print(f"Added waypoint ({x:.2f}, {y:.2f}), total waypoints: {len(self.waypoints)}")
    
    def clear_waypoints(self):
        """Clear all waypoints."""
        self.waypoints.clear()
        self.current_waypoint = None
        print("Cleared all waypoints")
    
    def start_following(self):
        """Start following the waypoints."""
        if not self.waypoints:
            print("No waypoints to follow")
            return False
        
        self.following = True
        # Reset to the first waypoint
        if self.current_waypoint is None and self.waypoints:
            self.current_waypoint = self.waypoints[0]
        
        print(f"Started following waypoints using {self.algorithm} algorithm")
        return True
    
    def stop_following(self):
        """Stop following waypoints."""
        self.following = False
        print("Stopped following waypoints")
    
    def is_following(self):
        """Check if currently following waypoints."""
        return self.following
    
    def set_algorithm(self, algorithm):
        """
        Set the path following algorithm.
        
        Args:
            algorithm (str): Algorithm name ('Linear', 'Smooth', 'Pure Pursuit')
        """
        self.algorithm = algorithm
        print(f"Set path following algorithm to {algorithm}")
    
    def update(self, dt, vehicle):
        """
        Update the path planner and calculate control signals.
        
        Args:
            dt (float): Time step in seconds
            vehicle: The vehicle model
            
        Returns:
            tuple: (left_voltage, right_voltage) or None if not following
        """
        if not self.following or not self.waypoints:
            return None
        
        # If there's no current waypoint, get the first one
        if self.current_waypoint is None:
            if self.waypoints:
                self.current_waypoint = self.waypoints[0]
            else:
                self.following = False
                return None
        
        # Get vehicle position and orientation
        x, y, theta = vehicle.get_position()
        
        # Check if we've reached the current waypoint
        wx, wy = self.current_waypoint
        distance_to_waypoint = np.sqrt((wx - x)**2 + (wy - y)**2)
        
        if distance_to_waypoint < self.waypoint_tolerance:
            # We've reached the waypoint, remove it and move to the next
            if self.waypoints and self.waypoints[0] == self.current_waypoint:
                self.waypoints.popleft()
            
            # If there are more waypoints, set the next one as the target
            if self.waypoints:
                self.current_waypoint = self.waypoints[0]
                print(f"Reached waypoint, moving to next: ({self.current_waypoint[0]:.2f}, {self.current_waypoint[1]:.2f})")
            else:
                # No more waypoints, stop following
                print("Reached final waypoint")
                self.following = False
                return None
        
        # Calculate control signals based on selected algorithm
        if self.algorithm == "Linear":
            return self._linear_control(x, y, theta, dt)
        elif self.algorithm == "Smooth":
            return self._smooth_control(x, y, theta, dt)
        elif self.algorithm == "Pure Pursuit":
            return self._pure_pursuit_control(x, y, theta, dt, vehicle)
        else:
            # Default to linear control
            return self._linear_control(x, y, theta, dt)
    
    def _linear_control(self, x, y, theta, dt):
        """
        Simple linear control to head directly toward the waypoint.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        wx, wy = self.current_waypoint
        
        # Calculate desired heading angle to the waypoint
        desired_angle = np.arctan2(wy - y, wx - x)
        
        # Calculate angle error (-pi to pi)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        
        # Calculate distance to waypoint
        distance = np.sqrt((wx - x)**2 + (wy - y)**2)
        
        # Use PID controllers to calculate voltages
        linear_speed = self.linear_pid.update(distance, dt)
        angular_correction = self.angular_pid.update(angle_error, dt)
        
        # Convert to wheel voltages
        left_voltage = linear_speed - angular_correction
        right_voltage = linear_speed + angular_correction
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)
    
    def _smooth_control(self, x, y, theta, dt):
        """
        Smooth control that gradually adjusts heading toward the waypoint.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        wx, wy = self.current_waypoint
        
        # Calculate desired heading angle to the waypoint
        desired_angle = np.arctan2(wy - y, wx - x)
        
        # Calculate angle error (-pi to pi)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        
        # Calculate distance to waypoint
        distance = np.sqrt((wx - x)**2 + (wy - y)**2)
        
        # Calculate a smooth speed profile based on distance
        # Slow down as we approach the waypoint
        max_speed = 8.0
        min_speed = 3.0
        slow_distance = 1.0  # Start slowing down at this distance
        
        if distance < self.waypoint_tolerance:
            linear_speed = 0.0
        elif distance < slow_distance:
            linear_speed = min_speed + (max_speed - min_speed) * (distance / slow_distance)
        else:
            linear_speed = max_speed
        
        # Calculate angular speed based on angle error
        # Use sigmoidal function for smooth turning
        max_angular = 5.0
        angular_speed = max_angular * np.tanh(2.0 * angle_error)
        
        # Convert to wheel voltages
        left_voltage = linear_speed - angular_speed
        right_voltage = linear_speed + angular_speed
        
        # Ensure voltages are within bounds (-12V to 12V)
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        
        return (left_voltage, right_voltage)
    
    def _pure_pursuit_control(self, x, y, theta, dt, vehicle):
        """
        Pure pursuit control algorithm for path following.
        
        Args:
            x, y, theta: Current vehicle position and orientation
            dt: Time step
            vehicle: The vehicle model
            
        Returns:
            tuple: (left_voltage, right_voltage)
        """
        # Get lookahead point
        if len(self.waypoints) > 1:
            # If there are multiple waypoints, find the lookahead point on the path
            lookahead_point = self._find_lookahead_point(x, y)
        else:
            # If there's only one waypoint, use that
            lookahead_point = self.current_waypoint
        
        # Calculate angle to lookahead point
        lx, ly = lookahead_point
        
        # Vector to lookahead point in vehicle frame
        dx = lx - x
        dy = ly - y
        
        # Rotate to vehicle frame
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        target_x_local = dx * cos_theta + dy * sin_theta
        target_y_local = -dx * sin_theta + dy * cos_theta
        
        # Calculate curvature
        lookahead_distance = np.sqrt(dx**2 + dy**2)
        if abs(target_y_local) < 1e-6 or lookahead_distance < 1e-6:
            curvature = 0
        else:
            curvature = 2 * target_y_local / (lookahead_distance**2)
        
        # Calculate wheel speeds based on curvature
        wheel_distance = vehicle.wheel_distance
        base_speed = 6.0  # Base voltage
        
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
    
    def _find_lookahead_point(self, x, y):
        """
        Find the lookahead point on the path.
        
        Args:
            x, y: Current vehicle position
            
        Returns:
            tuple: (x, y) of the lookahead point
        """
        # If we only have the current waypoint, use that
        if len(self.waypoints) <= 1:
            return self.current_waypoint
        
        # Convert deque to list for easier indexing
        waypoints_list = list(self.waypoints)
        
        # Find the closest point on each path segment
        closest_distance = float('inf')
        closest_point = None
        
        for i in range(len(waypoints_list) - 1):
            p1 = waypoints_list[i]
            p2 = waypoints_list[i + 1]
            
            # Find closest point on this segment
            closest_on_segment, dist = self._closest_point_on_segment(x, y, p1, p2)
            
            if dist < closest_distance:
                closest_distance = dist
                closest_point = closest_on_segment
        
        # If no closest point found, use the current waypoint
        if closest_point is None:
            return self.current_waypoint
        
        # Find the point that is lookahead_distance further along the path
        remaining_distance = self.lookahead_distance
        current_point = closest_point
        
        for i in range(len(waypoints_list) - 1):
            # Find the index of the segment containing the current point
            for j in range(len(waypoints_list) - 1):
                p1 = waypoints_list[j]
                p2 = waypoints_list[j + 1]
                
                # Check if current_point is on this segment
                if self._is_point_on_segment(current_point, p1, p2):
                    # Found the segment, now move along the path
                    segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    distance_along_segment = np.sqrt((current_point[0] - p1[0])**2 + (current_point[1] - p1[1])**2)
                    
                    if segment_length - distance_along_segment >= remaining_distance:
                        # The lookahead point is on this segment
                        ratio = (distance_along_segment + remaining_distance) / segment_length
                        lookahead_x = p1[0] + ratio * (p2[0] - p1[0])
                        lookahead_y = p1[1] + ratio * (p2[1] - p1[1])
                        return (lookahead_x, lookahead_y)
                    else:
                        # Move to the next segment
                        remaining_distance -= (segment_length - distance_along_segment)
                        current_point = p2
                        break
        
        # If we've gone through all segments and haven't found the lookahead point,
        # use the last waypoint
        return waypoints_list[-1]
    
    def _closest_point_on_segment(self, x, y, p1, p2):
        """
        Find the closest point on a line segment to a given point.
        
        Args:
            x, y: The point to find the closest point to
            p1, p2: The endpoints of the line segment
            
        Returns:
            tuple: ((closest_x, closest_y), distance)
        """
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate the line segment vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the squared length of the line segment
        len_sq = dx**2 + dy**2
        
        # If the segment is actually a point, return the point
        if len_sq < 1e-6:
            return p1, np.sqrt((x - x1)**2 + (y - y1)**2)
        
        # Calculate the projection of the point onto the line containing the segment
        t = ((x - x1) * dx + (y - y1) * dy) / len_sq
        
        # Clamp t to [0, 1] to ensure the closest point is on the segment
        t = max(0, min(1, t))
        
        # Calculate the closest point on the segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Calculate the distance to the closest point
        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        
        return ((closest_x, closest_y), distance)
    
    def _is_point_on_segment(self, point, p1, p2):
        """
        Check if a point is on a line segment.
        
        Args:
            point: The point to check
            p1, p2: The endpoints of the line segment
            
        Returns:
            bool: True if the point is on the segment, False otherwise
        """
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        
        # Calculate the line segment vector
        dx = x2 - x1
        dy = y2 - y1
        
        # Calculate the squared length of the line segment
        len_sq = dx**2 + dy**2
        
        # If the segment is actually a point, check if the point is the segment
        if len_sq < 1e-6:
            return abs(x - x1) < 1e-6 and abs(y - y1) < 1e-6
        
        # Calculate the projection of the point onto the line containing the segment
        t = ((x - x1) * dx + (y - y1) * dy) / len_sq
        
        # Check if the projection is within the segment
        if t < 0 or t > 1:
            return False
        
        # Calculate the closest point on the segment
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        
        # Check if the point is on the segment (within a small tolerance)
        tolerance = 1e-6
        return abs(x - closest_x) < tolerance and abs(y - closest_y) < tolerance


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
