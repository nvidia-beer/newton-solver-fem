# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Recording selection stays interactive only for deliberate terminal launches."""

import argparse
import io
import unittest
from contextlib import redirect_stderr, redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

from newton_solver_fem.examples import _suturebot_cli as cli
from newton_solver_fem.examples._suturebot_recording import DEFAULT_RECORDING, Recording, bundled_recordings


class TestSutureBotCLI(unittest.TestCase):
    def parser(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--sample")
        parser.add_argument("--choose-sample", action="store_true")
        parser.add_argument("--list-samples", action="store_true")
        parser.add_argument("--viewer", default="gl")
        for flag in ("headless", "test", "demo"):
            parser.add_argument(f"--{flag}", action="store_true")
        parser.add_argument("--benchmark", nargs="?", default=False, const=None, type=int)
        return parser

    def configure(self, arguments=(), *, text="", terminal=False):
        parser = self.parser()
        stream = io.StringIO(text)
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch("sys.argv", ["fem_suturebot", *arguments]),
            patch("sys.stdin", stream),
            patch.object(stream, "isatty", return_value=terminal),
            redirect_stdout(output),
            redirect_stderr(errors),
        ):
            cli.configure_sample(parser)
            args = parser.parse_args()
        return args.sample, output.getvalue(), errors.getvalue()

    def test_terminal_selects_number_name_and_default(self):
        names = sorted(
            bundled_recordings(include_aliases=False), key=lambda name: (name != DEFAULT_RECORDING.parent.name, name)
        )
        for text, expected in (("\n", names[0]), ("2\n", names[1]), (f"{names[-1]}\n", names[-1])):
            with self.subTest(text=text):
                sample, output, errors = self.configure(text=text, terminal=True)
                self.assertEqual(sample, expected)
                self.assertFalse(output)
                self.assertIn("Select recording", errors)

    def test_explicit_sample_is_preserved_even_when_menu_requested(self):
        for arguments in (
            ["--sample", "recordings/custom recording.json"],
            ["--sample=tissue3_222119_approach", "--choose-sample"],
        ):
            with self.subTest(arguments=arguments):
                with patch.object(cli, "bundled_recordings", side_effect=AssertionError("must not discover")):
                    sample, output, errors = self.configure(arguments, terminal=True)
                self.assertEqual(sample, self.parser().parse_args(arguments).sample)
                self.assertFalse(output or errors)

    def test_compatibility_alias_is_hidden_but_explicit_sample_still_works(self):
        recordings = cli._catalog(self.parser())
        name = "tissue3_215300_mesh_preview"
        self.assertNotIn(name, [key for key, _ in recordings])
        self.assertIn(name, bundled_recordings())
        sample, output, errors = self.configure(["--sample", name], terminal=True)
        self.assertEqual(sample, name)
        self.assertFalse(output or errors)
        self.assertEqual(Recording(sample).scene_path, Recording(DEFAULT_RECORDING).scene_path)

    def test_scripted_and_headless_launches_keep_default_without_reading_stdin(self):
        cases = (
            ([], False),
            (["--viewer", "null"], True),
            (["--viewer", "usd"], True),
            (["--headless"], True),
            (["--test"], True),
            (["--demo"], True),
            (["--benchmark"], True),
            (["--benchmark", "0"], True),
            (["--benchmark", "3"], True),
        )
        for arguments, terminal in cases:
            with self.subTest(arguments=arguments, terminal=terminal):
                with patch.object(cli, "_catalog", side_effect=AssertionError("must not discover")):
                    sample, output, errors = self.configure(arguments, text="q\n", terminal=terminal)
                self.assertEqual(sample, str(DEFAULT_RECORDING))
                self.assertFalse(output or errors)

    def test_forced_menu_accepts_piped_choice_and_retries_invalid_input(self):
        sample, output, errors = self.configure(["--choose-sample", "--viewer", "null"], text="bad\n0\n1\n")
        self.assertEqual(sample, DEFAULT_RECORDING.parent.name)
        self.assertFalse(output)
        self.assertEqual(errors.count("Invalid recording selection."), 2)

    def test_quit_and_end_of_input_exit_without_viewer(self):
        for text in ("", "q\n", "Q\n"):
            with self.subTest(text=text), self.assertRaises(SystemExit) as raised:
                self.configure(["--choose-sample"], text=text)
            self.assertEqual(raised.exception.code, 0)

    def test_listing_uses_actual_playback_scope_and_features(self):
        output = io.StringIO()
        errors = io.StringIO()
        with (
            patch("sys.argv", ["fem_suturebot", "--list-samples"]),
            redirect_stdout(output),
            redirect_stderr(errors),
            self.assertRaises(SystemExit) as raised,
        ):
            cli.configure_sample(self.parser())
        self.assertEqual(raised.exception.code, 0)
        self.assertFalse(errors.getvalue())
        listing = output.getvalue()
        self.assertTrue(listing.splitlines()[1].startswith(f" 1) {DEFAULT_RECORDING.parent.name} [default]"))
        for name, path in bundled_recordings(include_aliases=False).items():
            recording = Recording(path)
            line = next(line for line in listing.splitlines() if f") {name}" in line)
            self.assertIn(f"region {recording.manifest['region_id']}", line)
            self.assertIn(f"frames {recording.first_frame}-{recording.last_frame}", line)
            self.assertIn(f"{recording.duration:.3f} s", line)
            self.assertNotIn("needle", line)
            self.assertNotIn("thread", line)
            self.assertEqual("green point labels" in line, "point_labels" in recording.manifest)
            phase = recording.manifest.get("playback", {}).get("description")
            if phase:
                self.assertIn(phase, listing)
        self.assertIn("Alignment estimate: full tool/pad geometry is not validated.", listing)

    def test_known_alignment_failure_is_not_presented_as_only_unvalidated(self):
        reason = "Outer pad boundary differs from the source wrist views."
        recording = SimpleNamespace(
            manifest={"region_id": 3, "alignment": {"accepted": False, "failure_reason": reason}},
            first_frame=4,
            last_frame=106,
            duration=3.4,
        )
        output = io.StringIO()
        cli._show([("known_failed_pad", recording)], file=output)
        self.assertIn(f"Alignment check failed: {reason}", output.getvalue())
        self.assertNotIn("Alignment estimate:", output.getvalue())
        self.assertNotIn("thread", output.getvalue())


if __name__ == "__main__":
    unittest.main()
