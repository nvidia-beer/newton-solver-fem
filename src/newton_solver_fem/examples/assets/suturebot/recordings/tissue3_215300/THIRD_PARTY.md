# Sources and notices

Recorded images and motion: [SutureBot](https://suturebot.github.io/), [dataset](https://huggingface.co/datasets/jchen396/SutureBot), episode `tissue_3/2_needle_throw/20250428-215300-489212`. The retained dataset inventory identifies its license as CC-BY-4.0.

Instrument CAD: [jhu-dvrk/dvrk_model](https://github.com/jhu-dvrk/dvrk_model), commit 90472a9, MIT. The original notice is preserved in `DVRK_MODEL_LICENSE.txt`. PSM1 retains the Large Needle Driver CAD with the source-derived shaft diameter. PSM2 derives its body from that CAD and uses locally reconstructed, photo-estimated DeBakey forceps jaws; it is not manufacturer DeBakey CAD. `tool_geometry.json` records the separate tool assets, hashes, dimensions and geometry limitations.

Phantom geometry and source hashes are documented in `provenance.json`; no additional CAD license is inferred from the dataset license. The retained RCM cage STL has no standalone CAD license supplied. All source identities in that file are provenance, not dependencies.
