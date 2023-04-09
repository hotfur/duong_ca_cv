"""Library to control Jetson Nano based robot
@author: Nguyen Hoang An
@author: Vuong Kha Sieu"""

import Jetson.GPIO as GPIO
import numpy as np

# Constant
# Robot constants
a = 10  # length from front axle to center of mass
b = 10  # length from back axle to center of mass
l = 10 / 2  # length between back wheels (first number)
rw = 10  # drive wheel radius
angle_limit = np.deg2rad(2)  # Smallest angle in degree that the robot movement can correct
max_speed = 255  # Maximum speed allowed (maximum 255 to saturate the GPIO pins)
in1 = 37  # left back
in2 = 35  # left forward
in3 = 33  # right forward
in4 = 31  # right back
GPIO.setmode(GPIO.BOARD)
GPIO.setup(in1, GPIO.OUT)
GPIO.setup(in2, GPIO.OUT)
GPIO.setup(in3, GPIO.OUT)
GPIO.setup(in4, GPIO.OUT)
GPIO.setwarnings(False)


def write_pin(wheel):
    """Control the robot wheel by adjusting the robot speed
    :param wheel: control signal, contains the robot speed by wheel. Wheel 1 is
    :except Any exception occurred while writing to GPIO outputs
    :return None, str if exception occurred"""
    in1_speed, in2_speed, in3_speed, in4_speed = wheel
    try:
        GPIO.output(in1, int(in1_speed))
        GPIO.output(in2, int(in2_speed))
        GPIO.output(in3, int(in3_speed))
        GPIO.output(in4, int(in4_speed))
    except:
        return "Exception occurred while trying to write to GPIO pins"
    return None


def calc_speeds(center_velocity, omega):
    """Calculate speeds for the wheels
    :param center_velocity: the desired linear velocity for the center of mass (bounded 0-1)
    :param omega: the desired steering angle
    :return: wheel speeds, normalized to range 0-GPIO_max_val"""
    linear_speed = max_speed * center_velocity
    angular_speed = max_speed * (1 - center_velocity)
    if -angle_limit < omega < angle_limit:
        return np.full(shape=4, fill_value=linear_speed)
    rotation = np.abs(angular_speed * np.tan(omega) / (a + b))
    radius = (a + b) / np.tan(omega)
    wheel = np.zeros(4)
    # Left steer
    if omega > 0:
        wheel[0] = b ** 2 + (radius - l) ** 2
        wheel[1] = a ** 2 + (radius - l) ** 2
        wheel[2] = a ** 2 + (radius + l) ** 2
        wheel[3] = b ** 2 + (radius + l) ** 2
    # Right steer
    else:
        wheel[0] = b ** 2 + (radius + l) ** 2
        wheel[1] = a ** 2 + (radius + l) ** 2
        wheel[2] = a ** 2 + (radius - l) ** 2
        wheel[3] = b ** 2 + (radius - l) ** 2
    wheel = rotation * np.sqrt(wheel)
    return wheel + linear_speed


def control(center_velocity, omega):
    """Main function to control robot
    :param center_velocity: the desired linear velocity for the center of mass (bounded 0-1)
    :param omega: the desired steering angle
    :except Any exception occurred while writing to GPIO outputs
    :return None, str if exception occurred"""
    return write_pin(calc_speeds(center_velocity, omega))
