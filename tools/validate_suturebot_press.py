# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Validate the retained approximate-rig contact experiment, separately from replay.

The runtime demo plans targets for the imported PSM1 jaws and measures force
on free active FEM nodes, tissue displacement, and recovery after release.
This tool supplies no separate trajectory or direct pose/particle edits.
The needle stays a free rigid body throughout the ordinary coupled dynamics.

Run: uv run --locked python tools/validate_suturebot_press.py --device cuda:0
"""

import argparse
import json
import math
from importlib.metadata import version
from pathlib import Path

import warp as wp
from newton.viewer import ViewerNull

from newton_solver_fem.examples._suturebot_contact_demo import Example


def press_and_release(example):
    """Exercise the shared runtime demo, including its final release interval."""
    if not example.args.demo:
        raise ValueError("The pressure validator requires the ordinary --demo runtime path")
    example.step()
    example.test_final()
    plan = example.press_demo
    if plan is None:
        raise AssertionError(f"The runtime did not create an active-tissue press plan: {example.demo_phase}")
    if not math.isfinite(plan.duration) or plan.duration <= 0:
        raise ValueError("The runtime press duration must be positive and finite")
    frames = math.ceil(plan.duration / example.frame_dt)
    phase = example.demo_phase
    print(f"SutureBot region {example.selected_region_ids}, runtime demo: {phase}", flush=True)
    for _ in range(1, frames):
        example.step()
        example.test_final()
        if example.press_demo is not plan:
            raise AssertionError("The runtime unexpectedly replaced its press plan during validation")
        if example.demo_phase != phase:
            phase = example.demo_phase
            print(f"Runtime demo: {phase}", flush=True)
    plan.validate()
    report = dict(plan.report())
    if report["frames"] != frames:
        raise AssertionError("The runtime did not observe every validated frame")
    report["runtime_path"] = "_suturebot_contact_demo.Example --demo --regions " + " ".join(
        map(str, example.selected_region_ids)
    )
    report["duration_seconds"] = plan.duration
    report["every_frame_finite_positive_and_fixed_anchors_exact"] = True
    return report


def validate(device, backend="auto"):
    arguments = ["--demo", "--regions", "1", "--device", device, "--mujoco-backend", backend, "--viewer", "null"]
    args = Example.create_parser().parse_args(arguments)
    with wp.ScopedDevice(device):
        viewer = ViewerNull()
        try:
            report = press_and_release(Example(viewer, args))
            report["arguments"] = arguments
            report["dependencies"] = {name: version(name) for name in ("newton", "warp-lang", "mujoco", "mujoco-warp")}
            return report
        finally:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--mujoco-backend", choices=("auto", "cpu", "warp"), default="auto")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate(args.device, args.mujoco_backend)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
