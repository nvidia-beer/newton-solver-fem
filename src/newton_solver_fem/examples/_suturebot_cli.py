# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Choose a prepared SutureBot recording before opening the viewer."""

import sys

from newton_solver_fem.examples._suturebot_recording import DEFAULT_RECORDING, Recording, bundled_recordings


def _catalog(parser):
    paths = bundled_recordings()
    if not paths:
        parser.error("No bundled SutureBot recordings found. Supply --sample /path/to/recording.json.")
    names = sorted(paths, key=lambda name: (paths[name].resolve() != DEFAULT_RECORDING.resolve(), name))
    return [(name, Recording(paths[name])) for name in names]


def _show(recordings, *, file):
    print("SutureBot recordings:", file=file)
    for index, (name, recording) in enumerate(recordings, 1):
        manifest = recording.manifest
        features = ["needle" if "needle" in manifest else "no needle"]
        if "point_labels" in manifest:
            features.append("green point labels")
        details = ", ".join(features)
        default = " [default]" if index == 1 else ""
        print(
            f"{index:2}) {name}{default} | region {manifest['region_id']} | "
            f"frames {recording.first_frame}-{recording.last_frame} | {recording.duration:.3f} s | {details}",
            file=file,
        )
        if not manifest.get("alignment", {}).get("accepted", False):
            print("    Alignment estimate: full tool/pad geometry is not validated.", file=file)
        phase = manifest.get("playback", {}).get("description")
        if phase:
            print(f"    {phase}", file=file)
        if manifest.get("source_episode"):
            print(f"    Source: {manifest['source_episode']}", file=file)


def _choose(recordings):
    _show(recordings, file=sys.stderr)
    names = [name for name, _ in recordings]
    while True:
        print(
            f"Select recording (1-{len(names)}, name, Enter for 1, q to quit): ",
            end="",
            file=sys.stderr,
            flush=True,
        )
        line = sys.stdin.readline()
        if not line or line.strip().lower() == "q":
            raise SystemExit(0)
        choice = line.strip() or "1"
        if choice in names:
            return choice
        if choice.isdigit() and 1 <= int(choice) <= len(names):
            return names[int(choice) - 1]
        print("Invalid recording selection.", file=sys.stderr)


def configure_sample(parser):
    """Resolve the CLI sample default without initializing a viewer or simulation.

    Explicit samples and noninteractive runs retain their existing behavior.
    ``--choose-sample`` also allows a deliberately piped selection.
    """
    args = parser.parse_args()
    if args.list_samples:
        _show(_catalog(parser), file=sys.stdout)
        raise SystemExit(0)
    if args.sample is not None:
        return
    interactive = (
        sys.stdin.isatty()
        and args.viewer not in ("null", "usd")
        and not args.headless
        and not args.test
        and not args.demo
        and args.benchmark is False
    )
    sample = _choose(_catalog(parser)) if args.choose_sample or interactive else str(DEFAULT_RECORDING)
    parser.set_defaults(sample=sample)
