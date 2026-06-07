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

from unittest.mock import MagicMock, patch

import pytest

from lerobot.robots.lekiwi.config_lekiwi import LEKIWI_BASE_MOTOR_NAMES, LeKiwiConfig
from lerobot.robots.lekiwi.lekiwi import LeKiwi

_MODULE = "lerobot.robots.lekiwi.lekiwi"


@pytest.fixture
def lekiwi(tmp_path):
    bus_mock = MagicMock(name="FeetechBusMock")
    bus_mock.is_connected = True

    def _bus_side_effect(*_args, **kwargs):
        bus_mock.motors = kwargs["motors"]
        return bus_mock

    with (
        patch(f"{_MODULE}.FeetechMotorsBus", side_effect=_bus_side_effect),
        patch(f"{_MODULE}.make_cameras_from_configs", return_value={}),
    ):
        cfg = LeKiwiConfig(port="/dev/null", cameras={}, calibration_dir=tmp_path)
        yield LeKiwi(cfg)


def test_mecanum_base_uses_four_wheel_ids(lekiwi):
    wheel_ids = {name: lekiwi.bus.motors[name].id for name in LEKIWI_BASE_MOTOR_NAMES}

    assert wheel_ids == {
        "base_front_left_wheel": 7,
        "base_front_right_wheel": 8,
        "base_rear_left_wheel": 9,
        "base_rear_right_wheel": 10,
    }
    assert tuple(lekiwi.base_motors) == LEKIWI_BASE_MOTOR_NAMES


def test_mecanum_body_to_wheel_signs(lekiwi):
    forward = lekiwi._body_to_wheel_raw(0.1, 0.0, 0.0, max_raw=32767)
    assert len(set(forward.values())) == 1
    assert forward["base_front_left_wheel"] > 0

    left = lekiwi._body_to_wheel_raw(0.0, 0.1, 0.0, max_raw=32767)
    assert left["base_front_left_wheel"] < 0
    assert left["base_front_right_wheel"] > 0
    assert left["base_rear_left_wheel"] > 0
    assert left["base_rear_right_wheel"] < 0

    rotate_left = lekiwi._body_to_wheel_raw(0.0, 0.0, 45.0, max_raw=32767)
    assert rotate_left["base_front_left_wheel"] < 0
    assert rotate_left["base_front_right_wheel"] > 0
    assert rotate_left["base_rear_left_wheel"] < 0
    assert rotate_left["base_rear_right_wheel"] > 0


def test_mecanum_wheel_to_body_roundtrip(lekiwi):
    raw = lekiwi._body_to_wheel_raw(0.12, -0.06, 45.0, max_raw=32767)
    body = lekiwi._wheel_raw_to_body(*(raw[name] for name in LEKIWI_BASE_MOTOR_NAMES))

    assert body["x.vel"] == pytest.approx(0.12, abs=1e-3)
    assert body["y.vel"] == pytest.approx(-0.06, abs=1e-3)
    assert body["theta.vel"] == pytest.approx(45.0, abs=0.1)


def test_mecanum_wheel_commands_are_clipped(lekiwi):
    raw = lekiwi._body_to_wheel_raw(10.0, 10.0, 720.0)

    assert max(abs(value) for value in raw.values()) <= lekiwi.config.base_max_raw_wheel_speed


def test_send_action_writes_four_base_wheel_velocities(lekiwi):
    action = {f"{motor}.pos": 0.0 for motor in lekiwi.arm_motors}
    action |= {"x.vel": 0.1, "y.vel": 0.0, "theta.vel": 0.0}

    returned = lekiwi.send_action(action)

    assert returned == action
    velocity_call = lekiwi.bus.sync_write.call_args_list[1]
    assert velocity_call.args[0] == "Goal_Velocity"
    assert set(velocity_call.args[1]) == set(LEKIWI_BASE_MOTOR_NAMES)


def test_setup_motors_can_target_base_only(lekiwi):
    lekiwi.config.setup_motors = "base"

    with patch("builtins.input", return_value=""):
        lekiwi.setup_motors()

    assert [call.args[0] for call in lekiwi.bus.setup_motor.call_args_list] == list(
        reversed(LEKIWI_BASE_MOTOR_NAMES)
    )
