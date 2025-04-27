import numpy as np
import matplotlib.pyplot as plt
from motor_model import ElectricMotor

class DifferentialVehicle:
    def __init__(self, wheel_radius=0.05, wheel_distance=0.2, 
                    robot_mass=1.0, wheel_mass=0.1, 
                    left_motor=None, right_motor=None,
                    moment_inertia=None, friction_coef=0.01,
                    wheel_type="center"):
        self.wheel_radius = wheel_radius
        self.wheel_distance = wheel_distance
        self.robot_mass = robot_mass
        self.wheel_mass = wheel_mass
        self.total_mass = robot_mass + 2 * wheel_mass

        if moment_inertia is None:
            self.moment_inertia = (self.robot_mass / 2) * (self.wheel_distance / 2)**2
        else:
            self.moment_inertia = moment_inertia

        self.left_motor = left_motor if left_motor else ElectricMotor(name="Left Motor")
        self.right_motor = right_motor if right_motor else ElectricMotor(name="Right Motor")

        self.friction_coef = friction_coef
        self.wheel_type = wheel_type

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0
        self.omega = 0.0

        self.position_history = [(self.x, self.y)]

    def set_motor_voltages(self, left_voltage, right_voltage):
        self.left_motor.set_voltage(left_voltage)
        self.right_motor.set_voltage(right_voltage)

    def get_position(self):
        return (self.x, self.y, self.theta)

    def get_velocity(self):
        return (self.v, self.omega)

    def get_wheel_speeds(self):
        return (self.left_motor.get_speed(), self.right_motor.get_speed())

    def update(self, dt):
        left_load_torque = self._calculate_wheel_load_torque(self.left_motor)
        right_load_torque = self._calculate_wheel_load_torque(self.right_motor)

        self.left_motor.set_load_torque(left_load_torque)
        self.right_motor.set_load_torque(right_load_torque)

        self.left_motor.update(dt)
        self.right_motor.update(dt)

        left_wheel_speed = self.left_motor.get_speed()
        right_wheel_speed = self.right_motor.get_speed()

        left_v = left_wheel_speed * self.wheel_radius
        right_v = right_wheel_speed * self.wheel_radius

        self.v = (right_v + left_v) / 2.0
        self.omega = (right_v - left_v) / self.wheel_distance

        if abs(self.omega) < 1e-6:
            self.x += self.v * np.cos(self.theta) * dt
            self.y += self.v * np.sin(self.theta) * dt
        else:
            radius = self.v / self.omega
            self.x += radius * (np.sin(self.theta + self.omega * dt) - np.sin(self.theta))
            self.y += radius * (np.cos(self.theta) - np.cos(self.theta + self.omega * dt))
            self.theta += self.omega * dt
            self.theta = np.arctan2(np.sin(self.theta), np.cos(self.theta))

        self.position_history.append((self.x, self.y))

    def _calculate_wheel_load_torque(self, motor):
        rolling_resistance = self.friction_coef * self.total_mass * 9.81 * self.wheel_radius / 2
        return rolling_resistance

    def set_position(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta
        self.position_history.append((self.x, self.y))

    def plot_kinematic_model(self):
        plt.figure(figsize=(15, 10))
        wheel_speeds = np.linspace(-10, 10, 21)
        left_speeds, right_speeds = np.meshgrid(wheel_speeds, wheel_speeds)
        linear_v = self.wheel_radius * (right_speeds + left_speeds) / 2
        angular_v = self.wheel_radius * (right_speeds - left_speeds) / self.wheel_distance

        plt.subplot(2, 2, 1)
        plt.quiver(left_speeds, right_speeds, linear_v, angular_v)
        plt.xlabel('Left Wheel Speed (rad/s)')
        plt.ylabel('Right Wheel Speed (rad/s)')
        plt.title('Vehicle Motion for Wheel Speed Combinations')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        contour = plt.contourf(left_speeds, right_speeds, linear_v, 20, cmap='viridis')
        plt.colorbar(contour, label='Linear Velocity (m/s)')
        plt.xlabel('Left Wheel Speed (rad/s)')
        plt.ylabel('Right Wheel Speed (rad/s)')
        plt.title('Linear Velocity')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        contour = plt.contourf(left_speeds, right_speeds, angular_v, 20, cmap='viridis')
        plt.colorbar(contour, label='Angular Velocity (rad/s)')
        plt.xlabel('Left Wheel Speed (rad/s)')
        plt.ylabel('Right Wheel Speed (rad/s)')
        plt.title('Angular Velocity')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.plot([p[0] for p in self.position_history], [p[1] for p in self.position_history], 'b-')
        plt.scatter(self.x, self.y, color='red', s=100)
        plt.xlabel('X Position (m)')
        plt.ylabel('Y Position (m)')
        plt.title('Vehicle Trajectory')
        plt.axis('equal')
        plt.grid(True)

        plt.tight_layout()
        plt.show(block=True)
        plt.close('all')

    def plot_dynamic_model(self):
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        voltages = np.linspace(-12, 12, 100)
        no_load_speeds = [v / self.left_motor.Ke for v in voltages]
        plt.plot(voltages, no_load_speeds)
        plt.xlabel('Voltage (V)')
        plt.ylabel('No-Load Speed (rad/s)')
        plt.title('Motor Characteristic Curve')
        plt.grid(True)

        plt.subplot(2, 2, 2)
        speeds = np.linspace(-30, 30, 100)
        torques = [self.friction_coef * self.total_mass * 9.81 * self.wheel_radius / 2 for _ in speeds]
        plt.plot(speeds, torques)
        plt.xlabel('Wheel Speed (rad/s)')
        plt.ylabel('Load Torque (Nm)')
        plt.title('Wheel Load Torque vs Speed')
        plt.grid(True)

        plt.subplot(2, 2, 3)
        accelerations = []
        velocities = np.linspace(0, 5, 100)
        for v in velocities:
            f_resistance = self.friction_coef * self.total_mass * 9.81
            f_max = 2 * self.left_motor.Kt * 12 / self.left_motor.R / self.wheel_radius
            a = (f_max - f_resistance) / self.total_mass
            accelerations.append(a)

        plt.plot(velocities, accelerations)
        plt.xlabel('Velocity (m/s)')
        plt.ylabel('Acceleration (m/s²)')
        plt.title('Vehicle Acceleration vs Velocity')
        plt.grid(True)

        plt.subplot(2, 2, 4)
        plt.axis('off')
        plt.text(0.5, 0.9, 'Dynamic Model Equations', horizontalalignment='center', fontsize=12)

        equations = [
            r"$F_{traction} = \frac{2 \cdot torque}{wheel\_radius}$",
            r"$torque = K_t \cdot I$",
            r"$I = \frac{V - K_e \cdot \omega}{R}$",
            r"$F = m \cdot a$",
            r"$\tau = I \cdot \alpha$",
            r"$v = \frac{r(\omega_r + \omega_l)}{2}$",
            r"$\omega = \frac{r(\omega_r - \omega_l)}{L}$"
        ]

        for i, eq in enumerate(equations):
            plt.text(0.5, 0.8 - i*0.1, eq, horizontalalignment='center')

        plt.tight_layout()
        plt.show(block=True)
        plt.close('all')

    def reset(self):
        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.linear_velocity = 0.0
        self.angular_velocity = 0.0
        self.left_motor.speed = 0.0
        self.right_motor.speed = 0.0
        self.left_motor.position = 0.0
        self.right_motor.position = 0.0

if __name__ == "__main__":
    vehicle = DifferentialVehicle()
    dt = 0.05
    sim_time = 10.0

    for t in np.arange(0, sim_time, dt):
        if t < 3.0:
            vehicle.set_motor_voltages(6.0, 6.0)
        elif t < 5.0:
            vehicle.set_motor_voltages(6.0, 3.0)
        elif t < 7.0:
            vehicle.set_motor_voltages(-6.0, -6.0)
        else:
            vehicle.set_motor_voltages(-3.0, -6.0)

        vehicle.update(dt)

    vehicle.plot_kinematic_model()
    vehicle.plot_dynamic_model()
