# Interactive phantom regions

The `fem_phantom_regions` example loads the bundled phantom USD with one region
numbering scheme: regions 1–6 are the wounds in [SutureBot's setup figure](https://suturebot.github.io/),
and regions 7–12 cover the remaining pad areas. Region 1 is the dataset's
training wound; regions 2–6 are its generalization wounds.

One original connected practice feature is divided into three selectable
regions: wound 5 and its thin upper rim are region 5, the remainder including
the lower-left tail is region 8, and the upper slit between wounds 3 and 4 is
region 12. This lets
FEM simulate these areas independently. The other nine original region payloads
and all tetrahedral geometry are preserved. No SutureBot checkout is needed to
run the example. Install the `examples` extra for USD loading and the OpenGL
viewer.

The default 3D camera looks from the native +Y lettering edge, so “3-D Med”
faces the viewer. The USD coordinates and the established upright 2D map
orientation are unchanged.

```bash
./docker/run.sh fem_phantom_regions
# Local equivalent:
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions
```

Click a region on the upright 2D pad map to activate FEM for that region.
Selecting another region replaces the current selection, so only one region
is active at a time. Clicking the current region leaves its deformation intact;
use **None** to disable FEM. The map, active selection and command-line
`--regions` option all use the same numbers, 1–12. The 3D pad remains
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
# Start with the dataset's training wound; click the map to change selection.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 1
# Activate only the newly isolated area around dataset wound 5.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 5
# Activate the upper slit between dataset wounds 3 and 4.
uv run --locked --extra examples python -m newton_solver_fem.examples fem_phantom_regions --regions 12
```

The default selection is region 1.

The material controls are demonstration parameters, not values fitted to the
tissue videos. The example retains the source tetrahedral geometry and native
coordinates. Prepared polygons assign the slit/oval and its thin upper rim to
region 5, the upper slit to region 12, and the remaining material to region 8.
These partitions assign existing tetrahedra without changing their geometry.
See the [asset provenance](../src/newton_solver_fem/examples/assets/phantom_pad/README.md)
for the bundled USD and its source.

The [recorded demonstration](media/fem_phantom_regions.mp4) clicks the 2D map,
pulls wound 4, switches to wound 6, disables FEM, then returns to wound 4. Its script
sends mouse input through the map and attachment controls; deformation comes
from the FEM solve. Regenerate its MP4, animated preview and capture metadata
with:

```bash
uv run --locked --extra examples python tools/record_examples.py fem_phantom_regions --width 1280 --height 720
python3 tools/build_readme.py --require-videos
```

Prepared-recording setup and sample details are in the [replay guide](suturebot-recording.md).
