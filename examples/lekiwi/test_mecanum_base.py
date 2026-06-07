#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Smoke-test a four-wheel mecanum LeKiwi base without connecting the arm or cameras.

Examples:

    python examples/lekiwi/test_mecanum_base.py --port COM4
    python examples/lekiwi/test_mecanum_base.py --port COM4 --mode forward
    python examples/lekiwi/test_mecanum_base.py --port COM4 --mode keyboard
    python examples/lekiwi/test_mecanum_base.py --port COM4 --mode stop
    python examples/lekiwi/test_mecanum_base.py --port COM4 --mode left --xy-speed 0.08
"""

import argparse
import time

import numpy as np

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from lerobot.robots.lekiwi.config_lekiwi import LEKIWI_BASE_MOTOR_NAMES
from lerobot.utils.import_utils import require_package


BASE_MOTORS = {
    "base_front_left_wheel": Motor(7, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_front_right_wheel": Motor(8, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_rear_left_wheel": Motor(9, "sts3215", MotorNormMode.RANGE_M100_100),
    "base_rear_right_wheel": Motor(10, "sts3215", MotorNormMode.RANGE_M100_100),
}

BODY_COMMANDS = {
    "forward": (1.0, 0.0, 0.0),
    "backward": (-1.0, 0.0, 0.0),
    "left": (0.0, 1.0, 0.0),
    "right": (0.0, -1.0, 0.0),
    "rotate_left": (0.0, 0.0, 1.0),
    "rotate_right": (0.0, 0.0, -1.0),
}

DEFAULT_SEQUENCE = ("forward", "left", "rotate_left")


def degps_to_raw(degps: float) -> int:
    steps_per_deg = 4096.0 / 360.0
    speed_int = int(round(degps * steps_per_deg))
    return min(0x7FFF, max(-0x8000, speed_int))


def mecanum_kinematics_matrix(lateral_wheelbase: float, longitudinal_wheelbase: float) -> np.ndarray:
    rotation_lever = (lateral_wheelbase + longitudinal_wheelbase) / 2.0
    return np.array(
        [
            [1.0, -1.0, -rotation_lever],
            [1.0, 1.0, rotation_lever],
            [1.0, 1.0, -rotation_lever],
            [1.0, -1.0, rotation_lever],
        ]
    )


def body_to_wheel_raw(
    x: float,
    y: float,
    theta: float,
    *,
    wheel_radius: float,
    lateral_wheelbase: float,
    longitudinal_wheelbase: float,
    max_raw: int,
    wheel_signs: dict[str, int],
) -> dict[str, int]:
    theta_rad = theta * (np.pi / 180.0)
    velocity_vector = np.array([x, y, theta_rad])
    matrix = mecanum_kinematics_matrix(lateral_wheelbase, longitudinal_wheelbase)

    wheel_linear_speeds = matrix.dot(velocity_vector)
    wheel_angular_speeds = wheel_linear_speeds / wheel_radius
    wheel_degps = wheel_angular_speeds * (180.0 / np.pi)

    steps_per_deg = 4096.0 / 360.0
    max_raw_computed = float(np.max(np.abs(wheel_degps) * steps_per_deg))
    if max_raw_computed > max_raw:
        wheel_degps = wheel_degps * (max_raw / max_raw_computed)

    wheel_raw = np.array([degps_to_raw(deg) for deg in wheel_degps], dtype=int)
    sign_array = np.array([wheel_signs[name] for name in LEKIWI_BASE_MOTOR_NAMES], dtype=int)
    wheel_raw = wheel_raw * sign_array

    return {name: int(raw) for name, raw in zip(LEKIWI_BASE_MOTOR_NAMES, wheel_raw, strict=True)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the four-wheel mecanum LeKiwi base.")
    parser.add_argument("--port", default="COM4", help="Feetech controller serial port, e.g. COM4.")
    parser.add_argument(
        "--mode",
        default="sequence",
        choices=("sequence", "keyboard", "stop", *BODY_COMMANDS),
        help="Movement command to test. 'keyboard' enables live WASD control.",
    )
    parser.add_argument("--xy-speed", type=float, default=0.1, help="Linear speed in m/s.")
    parser.add_argument("--theta-speed", type=float, default=45.0, help="Angular speed in deg/s.")
    parser.add_argument("--speed-step", type=float, default=0.02, help="Linear speed step for R/F keys.")
    parser.add_argument(
        "--theta-speed-step",
        type=float,
        default=10.0,
        help="Angular speed step in deg/s for R/F keys.",
    )
    parser.add_argument("--min-xy-speed", type=float, default=0.02)
    parser.add_argument("--max-xy-speed", type=float, default=0.3)
    parser.add_argument("--min-theta-speed", type=float, default=10.0)
    parser.add_argument("--max-theta-speed", type=float, default=120.0)
    parser.add_argument("--control-hz", type=float, default=30.0, help="Keyboard control loop frequency.")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds to run each command.")
    parser.add_argument("--pause", type=float, default=0.8, help="Seconds to pause between commands.")
    parser.add_argument("--wheel-radius-m", type=float, default=0.05)
    parser.add_argument("--lateral-wheelbase-m", type=float, default=0.15)
    parser.add_argument("--longitudinal-wheelbase-m", type=float, default=0.12)
    parser.add_argument("--max-raw", type=int, default=3000)
    parser.add_argument("--front-left-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--front-right-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--rear-left-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--rear-right-sign", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--skip-configure", action="store_true", help="Do not set velocity mode first.")
    parser.add_argument("--yes", action="store_true", help="Skip the safety confirmation prompt.")
    return parser.parse_args()


def configure_base(bus: FeetechMotorsBus) -> None:
    motor_names = list(BASE_MOTORS)
    bus.disable_torque(motor_names)
    bus.configure_motors()
    for name in motor_names:
        bus.write("Operating_Mode", name, OperatingMode.VELOCITY.value)
    bus.enable_torque(motor_names)


def run_command(bus: FeetechMotorsBus, label: str, args: argparse.Namespace) -> None:
    x_scale, y_scale, theta_scale = BODY_COMMANDS[label]
    wheel_signs = {
        "base_front_left_wheel": args.front_left_sign,
        "base_front_right_wheel": args.front_right_sign,
        "base_rear_left_wheel": args.rear_left_sign,
        "base_rear_right_wheel": args.rear_right_sign,
    }
    cmd = body_to_wheel_raw(
        x_scale * args.xy_speed,
        y_scale * args.xy_speed,
        theta_scale * args.theta_speed,
        wheel_radius=args.wheel_radius_m,
        lateral_wheelbase=args.lateral_wheelbase_m,
        longitudinal_wheelbase=args.longitudinal_wheelbase_m,
        max_raw=args.max_raw,
        wheel_signs=wheel_signs,
    )

    print(f"\nTesting {label}: {cmd}")
    bus.sync_write("Goal_Velocity", cmd, normalize=False)
    time.sleep(args.duration)
    stop_base(bus)
    time.sleep(args.pause)


def run_keyboard_control(bus: FeetechMotorsBus, args: argparse.Namespace) -> None:
    require_package("pynput", extra="pynput-dep")
    from pynput import keyboard

    pressed_keys: set[str] = set()
    should_quit = False
    xy_speed = args.xy_speed
    theta_speed = args.theta_speed

    wheel_signs = {
        "base_front_left_wheel": args.front_left_sign,
        "base_front_right_wheel": args.front_right_sign,
        "base_rear_left_wheel": args.rear_left_sign,
        "base_rear_right_wheel": args.rear_right_sign,
    }

    def normalize_key(key) -> str | None:
        if isinstance(key, keyboard.KeyCode) and key.char:
            return key.char.lower()
        if key == keyboard.Key.space:
            return "space"
        if key == keyboard.Key.esc:
            return "esc"
        return None

    def on_press(key) -> None:
        nonlocal should_quit, xy_speed, theta_speed
        key_name = normalize_key(key)
        if key_name is None:
            return

        if key_name == "q" or key_name == "esc":
            should_quit = True
            return
        if key_name == "r" and key_name not in pressed_keys:
            xy_speed = min(args.max_xy_speed, xy_speed + args.speed_step)
            theta_speed = min(args.max_theta_speed, theta_speed + args.theta_speed_step)
            print(f"Speed: xy={xy_speed:.2f} m/s, theta={theta_speed:.0f} deg/s")
        elif key_name == "f" and key_name not in pressed_keys:
            xy_speed = max(args.min_xy_speed, xy_speed - args.speed_step)
            theta_speed = max(args.min_theta_speed, theta_speed - args.theta_speed_step)
            print(f"Speed: xy={xy_speed:.2f} m/s, theta={theta_speed:.0f} deg/s")

        pressed_keys.add(key_name)

    def on_release(key) -> None:
        key_name = normalize_key(key)
        if key_name is not None:
            pressed_keys.discard(key_name)

    listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    listener.start()

    print("\nKeyboard control active:")
    print("  W/S: forward/backward")
    print("  A/D: left/right strafe")
    print("  Z/X: rotate left/right")
    print("  R/F: speed up/down")
    print("  Space: stop while held")
    print("  Q or Esc: quit")

    try:
        period_s = 1.0 / args.control_hz
        while not should_quit:
            loop_start = time.perf_counter()

            x = 0.0
            y = 0.0
            theta = 0.0

            if "space" not in pressed_keys:
                if "w" in pressed_keys:
                    x += xy_speed
                if "s" in pressed_keys:
                    x -= xy_speed
                if "a" in pressed_keys:
                    y += xy_speed
                if "d" in pressed_keys:
                    y -= xy_speed
                if "z" in pressed_keys:
                    theta += theta_speed
                if "x" in pressed_keys:
                    theta -= theta_speed

            cmd = body_to_wheel_raw(
                x,
                y,
                theta,
                wheel_radius=args.wheel_radius_m,
                lateral_wheelbase=args.lateral_wheelbase_m,
                longitudinal_wheelbase=args.longitudinal_wheelbase_m,
                max_raw=args.max_raw,
                wheel_signs=wheel_signs,
            )
            bus.sync_write("Goal_Velocity", cmd, normalize=False)

            elapsed_s = time.perf_counter() - loop_start
            time.sleep(max(0.0, period_s - elapsed_s))
    finally:
        listener.stop()
        stop_base(bus)


def stop_base(bus: FeetechMotorsBus) -> None:
    bus.sync_write("Goal_Velocity", dict.fromkeys(BASE_MOTORS, 0), normalize=False, num_retry=5)


def main() -> None:
    args = parse_args()

    if not args.yes and args.mode != "stop":
        input(
            "Lift the base so all wheels are off the ground, then press ENTER to start. "
            "Press Ctrl+C to cancel."
        )

    modes = DEFAULT_SEQUENCE if args.mode == "sequence" else (args.mode,)
    bus = FeetechMotorsBus(port=args.port, motors=BASE_MOTORS)

    bus.connect()
    try:
        if args.mode == "stop":
            stop_base(bus)
            print("Base stopped.")
            return

        if not args.skip_configure:
            configure_base(bus)

        if args.mode == "keyboard":
            run_keyboard_control(bus, args)
        else:
            for mode in modes:
                run_command(bus, mode, args)
    finally:
        stop_base(bus)
        bus.disconnect()


if __name__ == "__main__":
    main()
