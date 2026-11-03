# Sources and notices

Recorded images and motion: [SutureBot](https://suturebot.github.io/), [dataset](https://huggingface.co/datasets/jchen396/SutureBot), episode `tissue_3/2_needle_throw/20250428-215300-489212`. The retained dataset inventory identifies its license as CC-BY-4.0.

Instrument CAD: [jhu-dvrk/dvrk_model](https://github.com/jhu-dvrk/dvrk_model), commit 90472a9, MIT. The original notice is preserved in `DVRK_MODEL_LICENSE.txt`. The Large Needle Driver CAD is used for both tools because exact second-tool CAD was unavailable.

Phantom geometry and source hashes are documented in `provenance.json`; no additional CAD license is inferred from the dataset license. The retained RCM cage STL has no standalone CAD license supplied. All source identities in that file are provenance, not dependencies.

Needle CAD: surgical-robotics-ai/surgical_robotics_challenge, commit 26bb2f6, according to the retained asset inventory; no standalone LICENSE file was recorded. The source mesh hash and preparation scale are retained in `recording.json` under `needle`. The dataset license is not assigned to this CAD.
