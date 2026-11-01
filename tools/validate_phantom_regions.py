"""Replay the phantom video workload without rendering and validate every frame."""

import argparse
import json
import runpy
from importlib.metadata import version
from pathlib import Path

import newton.viewer
import numpy as np
import warp as wp

from newton_solver_fem.examples.example_fem_phantom_regions import Example

ROOT = Path(__file__).resolve().parents[1]
NAME = "fem_phantom_regions"


def validate(device):
    profile = json.loads((ROOT / "docs/videos.json").read_text())[NAME]
    frames = profile["frames"]
    hooks = runpy.run_path(str(ROOT / "tools/video_scenarios.py"))
    args = Example.create_parser().parse_args(
        [*profile["args"], "--device", device, "--viewer", "null", "--num-frames", str(frames)]
    )
    with wp.ScopedDevice(device):
        viewer = newton.viewer.ViewerNull(num_frames=frames)
        try:
            example = Example(viewer, args)
            hooks["configure"](NAME, example)
            minima, last_revision = {}, -1
            for frame in range(frames):
                hooks["before_frame"](NAME, example, frame)
                example.step()
                example.render()
                example.test_final()
                if len(example.selection.indices):
                    if example.selection_revision != last_revision:
                        corners = example.selection.vertices[example.selection.indices]
                        rest_determinants = np.linalg.det(corners[:, 1:] - corners[:, :1])
                        last_revision = example.selection_revision
                    positions = example.state_0.particle_q.numpy()
                    corners = positions[example.selection.indices].astype(np.float64)
                    jacobians = np.linalg.det(corners[:, 1:] - corners[:, :1]) / rest_determinants
                    key = ",".join(str(value) for value in example.selected_region_ids)
                    minima[key] = min(minima.get(key, 1.0), float(jacobians.min()))
                if frame % 90 == 0:
                    print(f"Validated null frame {frame + 1}/{frames}", flush=True)
            return {
                "frames": frames,
                "device": device,
                "arguments": profile["args"],
                "minimum_jacobian": min(minima.values()),
                "minimum_jacobian_by_selection": minima,
                "every_frame_finite_positive_and_fixed_anchors_exact": True,
                "scenario": hooks["recording_metadata"](NAME, example),
                "dependencies": {name: version(name) for name in ("newton-solver-fem", "newton", "warp-lang")},
                "performance_benchmark": False,
            }
        finally:
            viewer.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    report = validate(options.device)
    text = json.dumps(report, indent=2) + "\n"
    if options.output:
        options.output.parent.mkdir(parents=True, exist_ok=True)
        options.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
