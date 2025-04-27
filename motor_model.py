import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

class ElectricMotor:
    def __init__(self, name="Motor", voltage_constant=0.01, torque_constant=0.01, 
                resistance=1.0, inertia=0.001, friction=0.01, max_voltage=12.0):
        self.name = name
        self.Ke = voltage_constant
        self.Kt = torque_constant
        self.R = resistance
        self.J = inertia
        self.B = friction
        self.max_voltage = max_voltage
        
        self.voltage = 0.0
        self.current = 0.0
        self.speed = 0.0
        self.position = 0.0
        self.load_torque = 0.0
        
        self.time_data = []
        self.voltage_data = []
        self.speed_data = []
        self.current_data = []
        
    def set_voltage(self, voltage):
        self.voltage = np.clip(voltage, -self.max_voltage, self.max_voltage)
        
    def set_load_torque(self, torque):
        self.load_torque = torque
        
    def update(self, dt):
        back_emf = self.Ke * self.speed
        self.current = (self.voltage - back_emf) / self.R
        motor_torque = self.Kt * self.current
        net_torque = motor_torque - self.load_torque - self.B * self.speed
        acceleration = net_torque / self.J
        self.speed += acceleration * dt
        self.position += self.speed * dt
        self.time_data.append(len(self.time_data) * dt if not self.time_data else self.time_data[-1] + dt)
        self.voltage_data.append(self.voltage)
        self.speed_data.append(self.speed)
        self.current_data.append(self.current)
        
    def get_speed(self):
        return self.speed
        
    def get_position(self):
        return self.position
        
    def get_current(self):
        return self.current
        
    def plot_motor_characteristics(self):
        plt.figure(figsize=(15, 5))
        voltages = np.linspace(-self.max_voltage, self.max_voltage, 100)
        no_load_speeds = [v / self.Ke for v in voltages]
        plt.subplot(1, 3, 1)
        plt.plot(voltages, no_load_speeds)
        plt.xlabel('Voltage (V)')
        plt.ylabel('No-Load Speed (rad/s)')
        plt.title(f'{self.name}: Speed vs Voltage')
        plt.grid(True)
        currents = np.linspace(-20, 20, 100)
        torques = [self.Kt * i for i in currents]
        plt.subplot(1, 3, 2)
        plt.plot(currents, torques)
        plt.xlabel('Current (A)')
        plt.ylabel('Torque (Nm)')
        plt.title(f'{self.name}: Torque vs Current')
        plt.grid(True)
        constant_voltage = self.max_voltage
        torques = np.linspace(0, self.Kt * constant_voltage / self.R, 100)
        speeds = [(constant_voltage - torque / self.Kt * self.R) / self.Ke for torque in torques]
        plt.subplot(1, 3, 3)
        plt.plot(torques, speeds)
        plt.xlabel('Torque (Nm)')
        plt.ylabel('Speed (rad/s)')
        plt.title(f'{self.name}: Speed vs Torque at {constant_voltage}V')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
        
    def plot_time_series(self):
        plt.figure(figsize=(15, 10))
        plt.subplot(3, 1, 1)
        plt.plot(self.time_data, self.voltage_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage (V)')
        plt.title(f'{self.name}: Voltage vs Time')
        plt.grid(True)
        plt.subplot(3, 1, 2)
        plt.plot(self.time_data, self.speed_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Speed (rad/s)')
        plt.title(f'{self.name}: Speed vs Time')
        plt.grid(True)
        plt.subplot(3, 1, 3)
        plt.plot(self.time_data, self.current_data)
        plt.xlabel('Time (s)')
        plt.ylabel('Current (A)')
        plt.title(f'{self.name}: Current vs Time')
        plt.grid(True)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    motor = ElectricMotor(name="Test Motor")
    dt = 0.001
    sim_time = 0.5
    for t in np.arange(0, sim_time, dt):
        if t < 0.1:
            motor.set_voltage(12.0)
        elif t < 0.2:
            motor.set_voltage(-12.0)
        else:
            motor.set_voltage(0.0)
        motor.update(dt)
    motor.plot_time_series()
    motor.plot_motor_characteristics()
