import numpy as np
from collections import deque

class PathPlanner:
    def __init__(self):
        self.waypoints = deque()
        self.current_waypoint = None
        self.lookahead_distance = 0.5
        self.waypoint_tolerance = 0.1
        self.algorithm = "Linear"
        self.following = False
        self.linear_pid = PIDController(kp=5.0, ki=0.0, kd=1.0, max_output=10.0)
        self.angular_pid = PIDController(kp=8.0, ki=0.1, kd=1.0, max_output=10.0)
    
    def add_waypoint(self, x, y):
        self.waypoints.append((x, y))
        print(f"Added waypoint ({x:.2f}, {y:.2f}), total waypoints: {len(self.waypoints)}")
    
    def clear_waypoints(self):
        self.waypoints.clear()
        self.current_waypoint = None
        print("Cleared all waypoints")
    
    def start_following(self):
        if not self.waypoints:
            print("No waypoints to follow")
            return False
        self.following = True
        if self.current_waypoint is None and self.waypoints:
            self.current_waypoint = self.waypoints[0]
        print(f"Started following waypoints using {self.algorithm} algorithm")
        return True
    
    def stop_following(self):
        self.following = False
        print("Stopped following waypoints")
    
    def is_following(self):
        return self.following
    
    def set_algorithm(self, algorithm):
        self.algorithm = algorithm
        print(f"Set path following algorithm to {algorithm}")
    
    def update(self, dt, vehicle):
        if not self.following or not self.waypoints:
            return None
        if self.current_waypoint is None:
            if self.waypoints:
                self.current_waypoint = self.waypoints[0]
            else:
                self.following = False
                return None
        x, y, theta = vehicle.get_position()
        wx, wy = self.current_waypoint
        distance_to_waypoint = np.sqrt((wx - x)**2 + (wy - y)**2)
        if distance_to_waypoint < self.waypoint_tolerance:
            if self.waypoints and self.waypoints[0] == self.current_waypoint:
                self.waypoints.popleft()
            if self.waypoints:
                self.current_waypoint = self.waypoints[0]
                print(f"Reached waypoint, moving to next: ({self.current_waypoint[0]:.2f}, {self.current_waypoint[1]:.2f})")
            else:
                print("Reached final waypoint")
                self.following = False
                return None
        if self.algorithm == "Linear":
            return self._linear_control(x, y, theta, dt)
        elif self.algorithm == "Smooth":
            return self._smooth_control(x, y, theta, dt)
        elif self.algorithm == "Pure Pursuit":
            return self._pure_pursuit_control(x, y, theta, dt, vehicle)
        else:
            return self._linear_control(x, y, theta, dt)
    
    def _linear_control(self, x, y, theta, dt):
        wx, wy = self.current_waypoint
        desired_angle = np.arctan2(wy - y, wx - x)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        distance = np.sqrt((wx - x)**2 + (wy - y)**2)
        linear_speed = self.linear_pid.update(distance, dt)
        angular_correction = self.angular_pid.update(angle_error, dt)
        left_voltage = linear_speed - angular_correction
        right_voltage = linear_speed + angular_correction
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)
    
    def _smooth_control(self, x, y, theta, dt):
        wx, wy = self.current_waypoint
        desired_angle = np.arctan2(wy - y, wx - x)
        angle_error = np.arctan2(np.sin(desired_angle - theta), np.cos(desired_angle - theta))
        distance = np.sqrt((wx - x)**2 + (wy - y)**2)
        max_speed = 8.0
        min_speed = 3.0
        slow_distance = 1.0
        if distance < self.waypoint_tolerance:
            linear_speed = 0.0
        elif distance < slow_distance:
            linear_speed = min_speed + (max_speed - min_speed) * (distance / slow_distance)
        else:
            linear_speed = max_speed
        max_angular = 5.0
        angular_speed = max_angular * np.tanh(2.0 * angle_error)
        left_voltage = linear_speed - angular_speed
        right_voltage = linear_speed + angular_speed
        max_voltage = 12.0
        left_voltage = np.clip(left_voltage, -max_voltage, max_voltage)
        right_voltage = np.clip(right_voltage, -max_voltage, max_voltage)
        return (left_voltage, right_voltage)
    
    def _pure_pursuit_control(self, x, y, theta, dt, vehicle):
        if len(self.waypoints) > 1:
            lookahead_point = self._find_lookahead_point(x, y)
        else:
            lookahead_point = self.current_waypoint
        lx, ly = lookahead_point
        dx = lx - x
        dy = ly - y
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        target_x_local = dx * cos_theta + dy * sin_theta
        target_y_local = -dx * sin_theta + dy * cos_theta
        lookahead_distance = np.sqrt(dx**2 + dy**2)
        if abs(target_y_local) < 1e-6 or lookahead_distance < 1e-6:
            curvature = 0
        else:
            curvature = 2 * target_y_local / (lookahead_distance**2)
        wheel_distance = vehicle.wheel_distance
        base_speed = 6.0
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
    
    def _find_lookahead_point(self, x, y):
        if len(self.waypoints) <= 1:
            return self.current_waypoint
        waypoints_list = list(self.waypoints)
        closest_distance = float('inf')
        closest_point = None
        for i in range(len(waypoints_list) - 1):
            p1 = waypoints_list[i]
            p2 = waypoints_list[i + 1]
            closest_on_segment, dist = self._closest_point_on_segment(x, y, p1, p2)
            if dist < closest_distance:
                closest_distance = dist
                closest_point = closest_on_segment
        if closest_point is None:
            return self.current_waypoint
        remaining_distance = self.lookahead_distance
        current_point = closest_point
        for i in range(len(waypoints_list) - 1):
            for j in range(len(waypoints_list) - 1):
                p1 = waypoints_list[j]
                p2 = waypoints_list[j + 1]
                if self._is_point_on_segment(current_point, p1, p2):
                    segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                    distance_along_segment = np.sqrt((current_point[0] - p1[0])**2 + (current_point[1] - p1[1])**2)
                    if segment_length - distance_along_segment >= remaining_distance:
                        ratio = (distance_along_segment + remaining_distance) / segment_length
                        lookahead_x = p1[0] + ratio * (p2[0] - p1[0])
                        lookahead_y = p1[1] + ratio * (p2[1] - p1[1])
                        return (lookahead_x, lookahead_y)
                    else:
                        remaining_distance -= (segment_length - distance_along_segment)
                        current_point = p2
                        break
        return waypoints_list[-1]
    
    def _closest_point_on_segment(self, x, y, p1, p2):
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx**2 + dy**2
        if len_sq < 1e-6:
            return p1, np.sqrt((x - x1)**2 + (y - y1)**2)
        t = ((x - x1) * dx + (y - y1) * dy) / len_sq
        t = max(0, min(1, t))
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        distance = np.sqrt((x - closest_x)**2 + (y - closest_y)**2)
        return ((closest_x, closest_y), distance)
    
    def _is_point_on_segment(self, point, p1, p2):
        x, y = point
        x1, y1 = p1
        x2, y2 = p2
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx**2 + dy**2
        if len_sq < 1e-6:
            return abs(x - x1) < 1e-6 and abs(y - y1) < 1e-6
        t = ((x - x1) * dx + (y - y1) * dy) / len_sq
        if t < 0 or t > 1:
            return False
        closest_x = x1 + t * dx
        closest_y = y1 + t * dy
        tolerance = 1e-6
        return abs(x - closest_x) < tolerance and abs(y - closest_y) < tolerance


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
