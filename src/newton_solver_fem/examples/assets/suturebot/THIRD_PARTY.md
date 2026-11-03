# SutureBot simulation rig sources

The CAD inputs remain in MSc/SutureBot-workbench/assets. The manifest records input SHA-256 hashes.

- dVRK instrument meshes: jhu-dvrk/dvrk_model, commit 90472a9, MIT according to the original MSc/SutureBot/README.md asset inventory.
- Needle CAD: surgical-robotics-ai/surgical_robotics_challenge, commit 26bb2f6, according to the original asset inventory; no LICENSE file recorded.
- Trocar-cage CAD: retained SutureBot asset collection. No per-asset license file is present; do not infer MIT or CC-BY coverage.
- SutureBot dataset: CC-BY-4.0 according to the original README; https://huggingface.co/datasets/jchen396/SutureBot.
- PSM2 jaws, camera proxies and collision primitives are explicitly approximate geometry derived during local simulation preparation.

This export is a local simulation preparation artifact, not a claim of measured geometry, dynamics or optical calibration.
