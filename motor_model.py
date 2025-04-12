"""
Electric Motor Model for differential vehicle simulation.
This module simulates real DC motors with physical properties.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

class ElectricMotor:
    """
    Electric motor model based on real DC motor physics.
    
    The model converts input voltage to motor speed considering:
    - Motor's electrical properties (resistance)
    - Motor's mechanical properties (inertia, friction)
    - Motor's electromagnetic properties (voltage and torque constants)
    """
    
    def __init__(self, name="Motor", voltage_constant=0.01, torque_constant=0.01, 
                 resistance=1.0, inertia=0.001, friction=0.01, max_voltage=12.0):
        """
        Initialize motor with physical parameters.
        
        Args:
            name (str): Motor name for identification
            voltage_constant (float): Motor voltage constant Ke [V/(rad/s)]
            torque_constant (float): Motor torque constant Kt [Nm/A]
            resistance (float): Motor winding resistance [Ohms]
            inertia (float): Motor rotor inertia [kg*m^2]
            friction (float): Motor viscous friction coefficient [Nm/(rad/s)]
            max_voltage (float): Maximum allowed voltage [V]
        """
        self.name = name
        self.Ke = voltage_constant
        self.Kt = torque_constant
        self.R = resistance
        self.J = inertia
        self.B = friction
        self.max_voltage = max_voltage
        
        # State variables
        self.voltage = 0.0       # Applied voltage [V]
        self.current = 0.0       # Motor current [A]
        self.speed = 0.0         # Motor angular velocity [rad/s]
        self.position = 0.0      # Motor angular position [rad]
        self.load_torque = 0.0   # External load torque [Nm]
        
        # For time series plotting
        self.time_data = []
        self.voltage_data = []
        self.speed_data = []
        self.current_data = []
        
    def set_voltage(self, voltage):
        """Set the input voltage to the motor with limiting."""
        self.voltage = np.clip(voltage, -self.max_voltage, self.max_voltage)
        
    def set_load_torque(self, torque):
        """Set external load torque applied to the motor shaft."""
        self.load_torque = torque
        
    def update(self, dt):
        """
        Update motor state for one time step.
        
        Args:
            dt (float): Time step in seconds
        """
        # Calculate back-EMF
        back_emf = self.Ke * self.speed
        
        # Calculate current (I = (V - E) / R)
        self.current = (self.voltage - back_emf) / self.R
        
        # Calculate motor torque (T = Kt * I)
        motor_torque = self.Kt * self.current
        
        # Calculate net torque
        net_torque = motor_torque - self.load_torque - self.B * self.speed
        
        # Calculate acceleration (α = T / J)
        acceleration = net_torque / self.J
        
        # Update speed and position using Euler integration
        self.speed += acceleration * dt
        self.position += self.speed * dt
        
        # Record data for plotting
        self.time_data.append(len(self.time_data) * dt if not self.time_data else self.time_data[-1] + dt)
        self.voltage_data.append(self.voltage)
        self.speed_data.append(self.speed)
        self.current_data.append(self.current)
        
    def get_speed(self):
        """Get the current motor speed in rad/s."""
        return self.speed
        
    def get_position(self):
        """Get the current motor position in radians."""
        return self.position
        
    def get_current(self):
        """Get the current in amperes."""
        return self.current
        
    def plot_motor_characteristics(self):
        """
        Plot the motor characteristics including:
        - Speed vs Voltage
        - Torque vs Current
        - Speed vs Torque
        """
        plt.figure(figsize=(15, 5))
        
        # Speed vs Voltage plot
        voltages = np.linspace(-self.max_voltage, self.max_voltage, 100)
        no_load_speeds = [v / self.Ke for v in voltages]  # No-load speed
        
        plt.subplot(1, 3, 1)
        plt.plot(voltages, no_load_speeds)
        plt.xlabel('Voltage (V)')
        plt.ylabel('No-Load Speed (rad/s)')
        plt.title(f'{self.name}: Speed vs Voltage')
        plt.grid(True)
        
        # Torque vs Current plot
        currents = np.linspace(-20, 20, 100)
        torques = [self.Kt * i for i in currents]
        
        plt.subplot(1, 3, 2)
        plt.plot(currents, torques)
        plt.xlabel('Current (A)')
        plt.ylabel('Torque (Nm)')
        plt.title(f'{self.name}: Torque vs Current')
        plt.grid(True)
        
        # Speed vs Torque plot (for constant voltage)
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
        """Plot the time series data of the motor."""
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

# Simple test for the motor model if run directly
if __name__ == "__main__":
    motor = ElectricMotor(name="Test Motor")
    
    # Test with a step input
    dt = 0.001  # 1 ms time step
    sim_time = 0.5  # 0.5 seconds simulation
    
    # Apply 12V for 0.1s, then -12V for 0.1s, then 0V
    for t in np.arange(0, sim_time, dt):
        if t < 0.1:
            motor.set_voltage(12.0)
        elif t < 0.2:
            motor.set_voltage(-12.0)
        else:
            motor.set_voltage(0.0)
            
        motor.update(dt)
    
    # Plot the results
    motor.plot_time_series()
    motor.plot_motor_characteristics()
