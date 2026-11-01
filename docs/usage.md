# Interactive phantom regions

The `fem_phantom_regions` example loads the bundled, grouped phantom USD. It
contains the final ten groups: the base includes the “3-D Med” lettering, and
the paired upper strips share one group. No SutureBot checkout is needed
to run the example. Install the `examples` extra for USD loading and the OpenGL
viewer.

```bash
./docker/run.sh fem_phantom_regions
# Local equivalent:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions
```

Click a region on the upright 2D pad map to activate FEM for that region.
Selecting another region replaces the current selection, so only one region
is active at a time. Clicking the current region leaves its deformation intact;
use **None** to disable FEM. The map shows numbered regions from above with an
active selection highlight. The 3D pad remains
uniform pink, inspired by the [3-Dmed soft tissue suture pad](https://www.3-dmed.com/product/soft-tissue-suture-pad/),
and inactive material stays fixed. The displayed active tetrahedron count is
the actual FEM model size. Changing selection rebuilds the model from the
reference shape and clears existing holds.

Camera translation is scaled to the pad size (about 68 mm/s with WASD), and
mouse orbit, pan and zoom use a pivot near the pad. Use **Frame pad** to restore
this view.

Right-drag an active region to pull it with the same mouse tool as the Baymax
cutting example. Release to keep a spring hold, middle-click its marker to
release it, or use **Clear holds**. The inactive reference cannot be grabbed.
The selected region includes its local base. Nodes on interfaces with inactive
regions are fixed, and the pad underside is supported as well.

```bash
# Start with region 7; click the 2D pad map to change selection.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 7
```

The material controls are demonstration parameters, not values fitted to the
tissue videos. The example retains the source tetrahedral geometry and group
membership. See the [asset provenance](../src/newton_solver_fem/examples/assets/phantom_pad/README.md)
for the bundled USD and its source.

The [recorded demonstration](media/fem_phantom_regions.mp4) clicks the 2D map,
pulls region 7, switches to region 9, disables FEM, then returns to region 7. Its script
sends mouse input through the map and attachment controls; deformation comes
from the FEM solve. Regenerate its MP4, animated preview and capture metadata
with:

```bash
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
python3 tools/build_readme.py --require-videos
```
