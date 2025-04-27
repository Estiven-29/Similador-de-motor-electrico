"""
Position Controller module for differential vehicle simulation.
This module implements control algorithms to drive the vehicle to a target position and orientation.
"""

import numpy as np

class PositionController:
    def __init__(self, vehicle):
        self.vehicle = vehicle
        self.target_x = 0.0
        self.target_y = 0.0
        self.target_theta = 0.0
        self.algorithm = "PID"
        self.position_pid = PIDController(kp=5.0, ki=0.1, kd=1.0, max_output=10.0)
        self.orientation_pid = PIDController(kp=8.0, ki=0.1, kd=1.0, max_output=10.0)
        self.controlling = False
        self.distance_tolerance = 0.05
        self.angle_tolerance = np.radians(3.0)
    
    def set_target(self, x, y, theta):
        self.target_x = x
        self.target_y = y
        self.target_theta = theta
        self.position_pid.reset()
        self.orientation_pid.reset()
        print(f"Set target to ({x:.2f}, {y:.2f}, {np.degrees(theta):.2f}°)")
    
    def start_control(self):
        self.controlling = True
        print(f"Started position control using {self.algorithm} algorithm")
    
    def stop_control(self):
        self.controlling = False
        print("Stopped position control")
    
    def is_controlling(self):
        return self.controlling
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        print(f"Set position control algorithm to {algorithm}")
    
    def update(self, dt):
        if not self.controlling:
            return None
        x, y, theta = self.vehicle.get_position()
        distance_to_target = np.sqrt((self.target_x - x)**2 + (self.target_y - y)**2)
        angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
        if distance_to_target < self.distance_tolerance and abs(angle_error) < self.angle_tolerance:
            print("Reached target position and orientation")
            self.controlling = False
            return (0.0, 0.0)
        if self.algorithm == "PID":
            return self._pid_control(x, y, theta, dt)
        elif self.algorithm == "Pure Pursuit":
            return self._pure_pursuit_control(x, y, theta, dt)
        elif self.algorithm == "Feedback Linearization":
            return self._feedback_linearization_control(x, y, theta, dt)
        else:
            return self._pid_control(x, y, theta, dt)
    
    def _pid_control(self, x, y, theta, dt):
        desired_angle = np.arctan2(self.target_y - y, self.target_x - x)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        distance = np.sqrt((self.target_x - x)**2 + (self.target_y - y)**2)
        if abs(angle_error) > self.angle_tolerance and distance > self.distance_tolerance:
            angular_speed = self.orientation_pid.update(angle_error, dt)
            linear_speed = 0.0
        else:
            angular_speed = self.orientation_pid.update(angle_error, dt)
            linear_speed = self.position_pid.update(distance, dt)
            slow_distance = 0.5
            if distance < slow_distance:
                linear_speed *= distance / slow_distance
        if distance < self.distance_tolerance:
            final_angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
            angular_speed = self.orientation_pid.update(final_angle_error, dt)
            linear_speed = 0.0
        left_voltage = linear_speed - angular_speed
        right_voltage = linear_speed + angular_speed
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)
    
    def _pure_pursuit_control(self, x, y, theta, dt):
        dx = self.target_x - x
        dy = self.target_y - y
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        target_x_local = dx * cos_theta + dy * sin_theta
        target_y_local = -dx * sin_theta + dy * cos_theta
        lookahead_distance = np.sqrt(dx**2 + dy**2)
        lookahead_angle = np.arctan2(target_y_local, target_x_local)
        if abs(target_y_local) < 1e-6 or lookahead_distance < 1e-6:
            curvature = 0
        else:
            curvature = 2 * target_y_local / (lookahead_distance**2)
        wheel_distance = self.vehicle.wheel_distance
        max_speed = 8.0
        min_speed = 2.0
        slow_distance = 1.0
        if lookahead_distance < self.distance_tolerance:
            return self._orientation_control(theta, dt)
        elif lookahead_distance < slow_distance:
            base_speed = min_speed + (max_speed - min_speed) * (lookahead_distance / slow_distance)
        else:
            base_speed = max_speed
        if abs(curvature) < 1e-6:
            left_voltage = base_speed
            right_voltage = base_speed
        else:
            radius = 1.0 / abs(curvature)
            speed_ratio = (radius - wheel_distance/2) / (radius + wheel_distance/2)
            if curvature > 0:
                left_voltage = base_speed * speed_ratio
                right_voltage = base_speed
            else:
                left_voltage = base_speed
                right_voltage = base_speed * speed_ratio
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)
    
    def _feedback_linearization_control(self, x, y, theta, dt):
        dx = self.target_x - x
        dy = self.target_y - y
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        error_x_local = dx * cos_theta + dy * sin_theta
        error_y_local = -dx * sin_theta + dy * cos_theta
        k_rho = 3.0
        k_alpha = 8.0
        k_beta = -1.5
        rho = np.sqrt(error_x_local**2 + error_y_local**2)
        alpha = np.arctan2(error_y_local, error_x_local)
        beta = -theta - alpha
        if rho < self.distance_tolerance:
            return self._orientation_control(theta, dt)
        v = k_rho * rho
        w = k_alpha * alpha + k_beta * beta
        wheel_radius = self.vehicle.wheel_radius
        wheel_distance = self.vehicle.wheel_distance
        left_speed = (v - w * wheel_distance / 2) / wheel_radius
        right_speed = (v + w * wheel_distance / 2) / wheel_radius
        motor_k = 0.5
        left_voltage = left_speed * motor_k
        right_voltage = right_speed * motor_k
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)
    
    def _orientation_control(self, theta, dt):
        angle_error = np.arctan2(np.sin(self.target_theta - theta), np.cos(self.target_theta - theta))
        angular_speed = self.orientation_pid.update(angle_error, dt)
        left_voltage = -angular_speed
        right_voltage = angular_speed
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)


class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, max_output=float('inf'), min_output=-float('inf')):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.max_output = max_output
        self.min_output = min_output
        self.prev_error = 0.0
        self.integral = 0.0
    
    def update(self, error, dt):
        p_term = self.kp * error
        self.integral += error * dt
        i_term = self.ki * self.integral
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        output = min(self.max_output, max(self.min_output, output))
        self.prev_error = error
        return output
    
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
