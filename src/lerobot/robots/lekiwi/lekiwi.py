#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import time
from functools import cached_property
from itertools import chain

import numpy as np

from lerobot.cameras import make_cameras_from_configs
from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import (
    FeetechMotorsBus,
    OperatingMode,
)
from lerobot.types import RobotAction, RobotObservation
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..robot import Robot
from ..utils import ensure_safe_goal_position
from .config_lekiwi import LEKIWI_BASE_MOTOR_NAMES, LeKiwiConfig

logger = logging.getLogger(__name__)


class LeKiwi(Robot):
    """
    The robot includes a four-wheel mecanum mobile base and a remote follower arm.
    The leader arm is connected locally (on the laptop) and its joint positions are recorded and then
    forwarded to the remote follower arm (after applying a safety clamp).
    In parallel, keyboard teleoperation is used to generate raw velocity commands for the wheels.
    """

    config_class = LeKiwiConfig
    name = "lekiwi"

    def __init__(self, config: LeKiwiConfig):
        super().__init__(config)
        self.config = config
        norm_mode_body = MotorNormMode.DEGREES if config.use_degrees else MotorNormMode.RANGE_M100_100
        self.bus = FeetechMotorsBus(
            port=self.config.port,
            motors={
                # arm
                "arm_shoulder_pan": Motor(1, "sts3215", norm_mode_body),
                "arm_shoulder_lift": Motor(2, "sts3215", norm_mode_body),
                "arm_elbow_flex": Motor(3, "sts3215", norm_mode_body),
                "arm_wrist_flex": Motor(4, "sts3215", norm_mode_body),
                "arm_wrist_roll": Motor(5, "sts3215", norm_mode_body),
                "arm_gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
                # base
                "base_front_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_front_right_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_rear_left_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
                "base_rear_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
            },
            calibration=self.calibration,
        )
        self.arm_motors = [motor for motor in self.bus.motors if motor.startswith("arm")]
        self.base_motors = [motor for motor in self.bus.motors if motor.startswith("base")]
        self._validate_base_wheel_velocity_signs()
        self.cameras = make_cameras_from_configs(config.cameras)

    def _validate_base_wheel_velocity_signs(self) -> None:
        wheel_signs = self.config.base_wheel_velocity_signs
        missing = set(self.base_motors) - set(wheel_signs)
        unknown = set(wheel_signs) - set(self.base_motors)
        invalid = {name: sign for name, sign in wheel_signs.items() if sign not in {-1, 1}}

        if missing or unknown or invalid:
            raise ValueError(
                "base_wheel_velocity_signs must contain exactly one -1 or 1 entry for each base motor. "
                f"Missing={sorted(missing)}, unknown={sorted(unknown)}, invalid={invalid}."
            )

    @property
    def _state_ft(self) -> dict[str, type]:
        return dict.fromkeys(
            (
                "arm_shoulder_pan.pos",
                "arm_shoulder_lift.pos",
                "arm_elbow_flex.pos",
                "arm_wrist_flex.pos",
                "arm_wrist_roll.pos",
                "arm_gripper.pos",
                "x.vel",
                "y.vel",
                "theta.vel",
            ),
            float,
        )

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3) for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._state_ft, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        return self._state_ft

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected and all(cam.is_connected for cam in self.cameras.values())

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        self.bus.connect()
        if not self.is_calibrated and calibrate:
            logger.info(
                "Mismatch between calibration values in the motor and the calibration file or no calibration file found"
            )
            self.calibrate()

        for cam in self.cameras.values():
            cam.connect()

        self.configure()
        logger.info(f"{self} connected.")

    @property
    def is_calibrated(self) -> bool:
        return self.bus.is_calibrated

    def calibrate(self) -> None:
        if self.calibration:
            # Calibration file exists, ask user whether to use it or run new calibration
            user_input = input(
                f"Press ENTER to use provided calibration file associated with the id {self.id}, or type 'c' and press ENTER to run calibration: "
            )
            if user_input.strip().lower() != "c":
                logger.info(f"Writing calibration file associated with the id {self.id} to the motors")
                self.bus.write_calibration(self.calibration)
                return
        logger.info(f"\nRunning calibration of {self}")

        motors = self.arm_motors + self.base_motors

        self.bus.disable_torque(self.arm_motors)
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)

        input("Move robot to the middle of its range of motion and press ENTER....")
        homing_offsets = self.bus.set_half_turn_homings(self.arm_motors)

        homing_offsets.update(dict.fromkeys(self.base_motors, 0))

        full_turn_motor = [
            motor for motor in motors if any(keyword in motor for keyword in ["wheel", "wrist_roll"])
        ]
        unknown_range_motors = [motor for motor in motors if motor not in full_turn_motor]

        print(
            f"Move all arm joints except '{full_turn_motor}' sequentially through their "
            "entire ranges of motion.\nRecording positions. Press ENTER to stop..."
        )
        range_mins, range_maxes = self.bus.record_ranges_of_motion(unknown_range_motors)
        for name in full_turn_motor:
            range_mins[name] = 0
            range_maxes[name] = 4095

        self.calibration = {}
        for name, motor in self.bus.motors.items():
            self.calibration[name] = MotorCalibration(
                id=motor.id,
                drive_mode=0,
                homing_offset=homing_offsets[name],
                range_min=range_mins[name],
                range_max=range_maxes[name],
            )

        self.bus.write_calibration(self.calibration)
        self._save_calibration()
        print("Calibration saved to", self.calibration_fpath)

    def configure(self):
        # Set-up arm actuators (position mode)
        # We assume that at connection time, arm is in a rest position,
        # and torque can be safely disabled to run calibration.
        self.bus.disable_torque()
        self.bus.configure_motors()
        for name in self.arm_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
            # Set P_Coefficient to lower value to avoid shakiness (Default is 32)
            self.bus.write("P_Coefficient", name, 16)
            # Set I_Coefficient and D_Coefficient to default value 0 and 32
            self.bus.write("I_Coefficient", name, 0)
            self.bus.write("D_Coefficient", name, 32)

        for name in self.base_motors:
            self.bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)

        self.bus.enable_torque()

    def setup_motors(self) -> None:
        if self.config.setup_motors == "all":
            motors = chain(reversed(self.arm_motors), reversed(self.base_motors))
        elif self.config.setup_motors == "arm":
            motors = reversed(self.arm_motors)
        elif self.config.setup_motors == "base":
            motors = reversed(self.base_motors)
        else:
            raise ValueError(f"Unknown setup_motors option: {self.config.setup_motors}")

        for motor in motors:
            input(f"Connect the controller board to the '{motor}' motor only and press enter.")
            self.bus.setup_motor(motor)
            print(f"'{motor}' motor id set to {self.bus.motors[motor].id}")

    @staticmethod
    def _degps_to_raw(degps: float) -> int:
        steps_per_deg = 4096.0 / 360.0
        speed_in_steps = degps * steps_per_deg
        speed_int = int(round(speed_in_steps))
        # Cap the value to fit within signed 16-bit range (-32768 to 32767)
        if speed_int > 0x7FFF:
            speed_int = 0x7FFF  # 32767 -> maximum positive value
        elif speed_int < -0x8000:
            speed_int = -0x8000  # -32768 -> minimum negative value
        return speed_int

    @staticmethod
    def _raw_to_degps(raw_speed: int) -> float:
        steps_per_deg = 4096.0 / 360.0
        magnitude = raw_speed
        degps = magnitude / steps_per_deg
        return degps

    @staticmethod
    def _mecanum_kinematics_matrix(lateral_wheelbase: float, longitudinal_wheelbase: float) -> np.ndarray:
        rotation_lever = (lateral_wheelbase + longitudinal_wheelbase) / 2.0
        return np.array(
            [
                [1.0, -1.0, -rotation_lever],
                [1.0, 1.0, rotation_lever],
                [1.0, 1.0, -rotation_lever],
                [1.0, -1.0, rotation_lever],
            ]
        )

    def _base_wheel_velocity_sign_array(self, wheel_signs: dict[str, int] | None = None) -> np.ndarray:
        wheel_signs = self.config.base_wheel_velocity_signs if wheel_signs is None else wheel_signs
        return np.array([wheel_signs[name] for name in LEKIWI_BASE_MOTOR_NAMES], dtype=float)

    def _body_to_wheel_raw(
        self,
        x: float,
        y: float,
        theta: float,
        wheel_radius: float | None = None,
        lateral_wheelbase: float | None = None,
        longitudinal_wheelbase: float | None = None,
        max_raw: int | None = None,
        wheel_signs: dict[str, int] | None = None,
    ) -> dict[str, int]:
        """
        Convert desired body-frame velocities into four mecanum wheel raw commands.

        Parameters:
          x          : Linear velocity in x (m/s). Positive is forward.
          y          : Linear velocity in y (m/s). Positive is left.
          theta      : Rotational velocity (deg/s). Positive is counter-clockwise.
          wheel_radius: Radius of each wheel (meters).
          lateral_wheelbase: Left-right distance between wheel centers (meters).
          longitudinal_wheelbase: Front-rear distance between wheel centers (meters).
          max_raw    : Maximum allowed raw command (ticks) per wheel.

        Returns:
          A dictionary with wheel raw commands:
             {
                 "base_front_left_wheel": value,
                 "base_front_right_wheel": value,
                 "base_rear_left_wheel": value,
                 "base_rear_right_wheel": value,
             }.

        Notes:
          - Internally, the method converts theta to rad/s for the kinematics.
          - The mecanum rollers are assumed to be in top-view X layout.
          - The raw command is computed from the wheels angular speed in deg/s
            using _degps_to_raw(). If any command exceeds max_raw, all commands
            are scaled down proportionally.
        """
        wheel_radius = self.config.base_wheel_radius_m if wheel_radius is None else wheel_radius
        lateral_wheelbase = (
            self.config.base_lateral_wheelbase_m if lateral_wheelbase is None else lateral_wheelbase
        )
        longitudinal_wheelbase = (
            self.config.base_longitudinal_wheelbase_m
            if longitudinal_wheelbase is None
            else longitudinal_wheelbase
        )
        max_raw = self.config.base_max_raw_wheel_speed if max_raw is None else max_raw

        if wheel_radius <= 0 or lateral_wheelbase <= 0 or longitudinal_wheelbase <= 0 or max_raw <= 0:
            raise ValueError("Wheel radius, wheelbase dimensions, and max_raw must be positive.")

        theta_rad = theta * (np.pi / 180.0)
        velocity_vector = np.array([x, y, theta_rad])
        m = self._mecanum_kinematics_matrix(lateral_wheelbase, longitudinal_wheelbase)

        wheel_linear_speeds = m.dot(velocity_vector)
        wheel_angular_speeds = wheel_linear_speeds / wheel_radius
        wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

        steps_per_deg = 4096.0 / 360.0
        raw_floats = np.abs(wheel_degps) * steps_per_deg
        max_raw_computed = float(np.max(raw_floats))
        if max_raw_computed > max_raw:
            scale = max_raw / max_raw_computed
            wheel_degps = wheel_degps * scale

        wheel_raw = np.array([self._degps_to_raw(deg) for deg in wheel_degps], dtype=int)
        wheel_raw = wheel_raw * self._base_wheel_velocity_sign_array(wheel_signs).astype(int)

        return {name: int(raw) for name, raw in zip(LEKIWI_BASE_MOTOR_NAMES, wheel_raw, strict=True)}

    def _wheel_raw_to_body(
        self,
        front_left_wheel_speed,
        front_right_wheel_speed,
        rear_left_wheel_speed,
        rear_right_wheel_speed,
        wheel_radius: float | None = None,
        lateral_wheelbase: float | None = None,
        longitudinal_wheelbase: float | None = None,
        wheel_signs: dict[str, int] | None = None,
    ) -> dict[str, float]:
        """
        Convert four mecanum wheel raw feedback back into body-frame velocities.

        Parameters:
          wheel_raw   : Vector with raw wheel commands, in front-left, front-right, rear-left, rear-right order.
          wheel_radius: Radius of each wheel (meters).
          lateral_wheelbase: Left-right distance between wheel centers (meters).
          longitudinal_wheelbase: Front-rear distance between wheel centers (meters).

        Returns:
          A dict (x.vel, y.vel, theta.vel) all in m/s
        """
        wheel_radius = self.config.base_wheel_radius_m if wheel_radius is None else wheel_radius
        lateral_wheelbase = (
            self.config.base_lateral_wheelbase_m if lateral_wheelbase is None else lateral_wheelbase
        )
        longitudinal_wheelbase = (
            self.config.base_longitudinal_wheelbase_m
            if longitudinal_wheelbase is None
            else longitudinal_wheelbase
        )

        if wheel_radius <= 0 or lateral_wheelbase <= 0 or longitudinal_wheelbase <= 0:
            raise ValueError("Wheel radius and wheelbase dimensions must be positive.")

        wheel_raw = np.array(
            [
                front_left_wheel_speed,
                front_right_wheel_speed,
                rear_left_wheel_speed,
                rear_right_wheel_speed,
            ],
            dtype=float,
        )
        wheel_raw = wheel_raw * self._base_wheel_velocity_sign_array(wheel_signs)

        wheel_degps = np.array([self._raw_to_degps(raw_speed) for raw_speed in wheel_raw])
        wheel_radps = wheel_degps * (np.pi / 180.0)
        wheel_linear_speeds = wheel_radps * wheel_radius

        m = self._mecanum_kinematics_matrix(lateral_wheelbase, longitudinal_wheelbase)
        velocity_vector = np.linalg.pinv(m).dot(wheel_linear_speeds)
        x, y, theta_rad = velocity_vector
        theta = theta_rad * (180.0 / np.pi)

        return {
            "x.vel": float(x),
            "y.vel": float(y),
            "theta.vel": float(theta),
        }  # m/s and deg/s

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        # Read actuators position for arm and vel for base
        start = time.perf_counter()
        arm_pos = self.bus.sync_read("Present_Position", self.arm_motors)
        base_wheel_vel = self.bus.sync_read("Present_Velocity", self.base_motors)

        base_vel = self._wheel_raw_to_body(
            base_wheel_vel["base_front_left_wheel"],
            base_wheel_vel["base_front_right_wheel"],
            base_wheel_vel["base_rear_left_wheel"],
            base_wheel_vel["base_rear_right_wheel"],
        )

        arm_state = {f"{k}.pos": v for k, v in arm_pos.items()}

        obs_dict = {**arm_state, **base_vel}

        dt_ms = (time.perf_counter() - start) * 1e3
        logger.debug(f"{self} read state: {dt_ms:.1f}ms")

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read_latest()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        return obs_dict

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """Command lekiwi to move to a target joint configuration.

        The relative action magnitude may be clipped depending on the configuration parameter
        `max_relative_target`. In this case, the action sent differs from original action.
        Thus, this function always returns the action actually sent.

        Raises:
            RobotDeviceNotConnectedError: if robot is not connected.

        Returns:
            RobotAction: the action sent to the motors, potentially clipped.
        """

        arm_goal_pos = {k: v for k, v in action.items() if k.endswith(".pos")}
        base_goal_vel = {k: v for k, v in action.items() if k.endswith(".vel")}

        base_wheel_goal_vel = self._body_to_wheel_raw(
            base_goal_vel["x.vel"], base_goal_vel["y.vel"], base_goal_vel["theta.vel"]
        )

        # Cap goal position when too far away from present position.
        # /!\ Slower fps expected due to reading from the follower.
        if self.config.max_relative_target is not None:
            present_pos = self.bus.sync_read("Present_Position", self.arm_motors)
            goal_present_pos = {key: (g_pos, present_pos[key]) for key, g_pos in arm_goal_pos.items()}
            arm_safe_goal_pos = ensure_safe_goal_position(goal_present_pos, self.config.max_relative_target)
            arm_goal_pos = arm_safe_goal_pos

        # Send goal position to the actuators
        arm_goal_pos_raw = {k.replace(".pos", ""): v for k, v in arm_goal_pos.items()}
        self.bus.sync_write("Goal_Position", arm_goal_pos_raw)
        self.bus.sync_write("Goal_Velocity", base_wheel_goal_vel)

        return {**arm_goal_pos, **base_goal_vel}

    def stop_base(self):
        self.bus.sync_write("Goal_Velocity", dict.fromkeys(self.base_motors, 0), num_retry=5)
        logger.info("Base motors stopped")

    @check_if_not_connected
    def disconnect(self):
        self.stop_base()
        self.bus.disconnect(self.config.disable_torque_on_disconnect)
        for cam in self.cameras.values():
            cam.disconnect()

        logger.info(f"{self} disconnected.")
