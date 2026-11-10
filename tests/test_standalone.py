# SPDX-License-Identifier: Apache-2.0
"""The installed package discovers its demos and loads its own bundled resources."""

import ast
import importlib.metadata
import importlib.resources
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from urllib.parse import urlparse

import newton_solver_fem
from newton_solver_fem.examples.__main__ import discover

_PACKAGE = Path(importlib.resources.files(newton_solver_fem)).resolve()
_REPOSITORY = Path(__file__).resolve().parents[1]
_DISTRIBUTION = "newton-solver-fem"
_SOLVER_NAMES = re.compile(r"\bnewton(?:_solver_[A-Za-z0-9_]+|-solver-[A-Za-z0-9_-]+)")


class StandalonePackageTests(unittest.TestCase):
    def test_only_the_three_fem_examples_are_discovered_from_the_package(self):
        examples = discover()
        self.assertEqual(set(examples), {"linear_fem", "linear_fem_baymax_cut", "diffsim_bear"})
        for description, _headless, module in examples.values():
            with self.subTest(module=module):
                self.assertTrue(description.strip())
                spec = importlib.util.find_spec(f"newton_solver_fem.examples.{module}")
                self.assertIsNotNone(spec)
                self.assertTrue(Path(spec.origin).resolve().is_relative_to(_PACKAGE))

    def test_runtime_has_no_foreign_solver_import_or_external_symlink(self):
        for path in _PACKAGE.rglob("*"):
            if path.is_symlink():
                self.assertTrue(path.resolve().is_relative_to(_PACKAGE), str(path))
        # Scan the installed package rather than a fixed source-file inventory.
        # String constants also catch importlib/__import__ targets and sys.path
        # workarounds that an ordinary Import/ImportFrom check would miss.
        for path in _PACKAGE.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                values = []
                if isinstance(node, ast.Import):
                    values = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    values = [node.module or ""]
                elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                    values = [node.value]
                for value in values:
                    with self.subTest(file=path.relative_to(_PACKAGE), line=getattr(node, "lineno", 0)):
                        for name in _SOLVER_NAMES.findall(value):
                            self.assertEqual(name.replace("_", "-"), _DISTRIBUTION)

    def test_metadata_has_no_sibling_or_local_external_dependency(self):
        distribution = importlib.metadata.distribution("newton-solver-fem")
        self.assertEqual(distribution.metadata["Name"], "newton-solver-fem")
        for requirement in distribution.requires or ():
            name = re.match(r"[A-Za-z0-9_.-]+", requirement).group().lower().replace("_", "-")
            if name.startswith("newton-solver-"):
                self.assertEqual(name, _DISTRIBUTION)
            if " @ " in requirement:
                self.assertIn(urlparse(requirement.split(" @ ", 1)[1]).scheme, {"https", "http", "git+https"})

        project = tomllib.loads((_REPOSITORY / "pyproject.toml").read_text())
        for name, source in project.get("tool", {}).get("uv", {}).get("sources", {}).items():
            if name.replace("_", "-").startswith("newton-solver-"):
                self.assertEqual(name.replace("_", "-"), _DISTRIBUTION)
            for entry in source if isinstance(source, list) else [source]:
                if "path" in entry:
                    self.assertTrue((_REPOSITORY / entry["path"]).resolve().is_relative_to(_REPOSITORY), name)
        lock = tomllib.loads((_REPOSITORY / "uv.lock").read_text())
        for package in lock["package"]:
            name = package["name"].replace("_", "-")
            if name.startswith("newton-solver-"):
                self.assertEqual(name, _DISTRIBUTION)
            for kind in ("directory", "editable", "path"):
                if kind in package.get("source", {}):
                    path = (_REPOSITORY / package["source"][kind]).resolve()
                    self.assertTrue(path.is_relative_to(_REPOSITORY), package["name"])

    def test_installed_resources_load_from_unrelated_working_directory_without_sibling(self):
        script = r"""
import importlib
import importlib.abc
import importlib.resources
import json
import pkgutil
import sys
from pathlib import Path

def foreign_solver(fullname):
    top_level = fullname.split(".", 1)[0]
    return top_level.startswith("newton_solver_") and top_level != "newton_solver_fem"

class BlockForeignSolver(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if foreign_solver(fullname):
            raise AssertionError("Runtime tried to import another solver package: " + fullname)

sys.meta_path.insert(0, BlockForeignSolver())
import newton_solver_fem as package
import newton_solver_fem._src as runtime
from pxr import Usd, UsdGeom
from newton_solver_fem.examples.example_linear_fem import _resolve_path
from newton_solver_fem.examples.example_linear_fem_baymax_cut import load_baymax
from newton_solver_fem.examples.diffsim._bear_training import ASSET_PATH

for module in pkgutil.walk_packages(runtime.__path__, runtime.__name__ + "."):
    importlib.import_module(module.name)
root = Path(importlib.resources.files(package)).resolve()
assets = root / "examples" / "assets"
paths = [Path(_resolve_path("assets/tet_grid.usda")), Path(_resolve_path("assets/hex_grid.usda"))]
for path, element_type in zip(paths, ("tet", "hex"), strict=True):
    mesh = package.load_mesh(path)[0]
    assert mesh.element_type == element_type and len(mesh.vertices) and len(mesh.indices)
points, elements = load_baymax()
assert len(points) and len(elements), "Bundled cutting geometry is unavailable"
paths.extend((assets / "tet_baymax.usda", ASSET_PATH))
stage = Usd.Stage.Open(str(ASSET_PATH))
bear = UsdGeom.Mesh(stage.GetPrimAtPath("/root/bear/bear"))
assert len(bear.GetPointsAttr().Get()) and len(bear.GetPrim().GetAttribute("tetraIndices").Get())
for path in paths:
    assert path.is_file() and path.resolve().is_relative_to(assets), str(path)
assert not any(foreign_solver(name) for name in sys.modules)
print(json.dumps({"resources": ["tet", "hex", "cutting", "bear"]}))
"""
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", NEWTON_TEST_DEVICE="cpu", OPENBLAS_NUM_THREADS="1")
        environment.pop("PYTHONPATH", None)
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-I", "-c", script],
                cwd=directory,
                env=environment,
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        report = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(report["resources"], ["tet", "hex", "cutting", "bear"])


if __name__ == "__main__":
    unittest.main()
