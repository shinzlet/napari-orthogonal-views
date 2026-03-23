# napari-orthogonal-views

[![License BSD-3](https://img.shields.io/pypi/l/napari-orthogonal-views.svg?color=green)](https://github.com/AnniekStok/napari-orthogonal-views/raw/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/napari-orthogonal-views.svg?color=green)](https://pypi.org/project/napari-orthogonal-views)
[![Python Version](https://img.shields.io/pypi/pyversions/napari-orthogonal-views.svg?color=green)](https://python.org)
[![tests](https://github.com/AnniekStok/napari-orthogonal-views/workflows/tests/badge.svg)](https://github.com/AnniekStok/napari-orthogonal-views/actions)
[![codecov](https://codecov.io/gh/AnniekStok/napari-orthogonal-views/branch/main/graph/badge.svg)](https://codecov.io/gh/AnniekStok/napari-orthogonal-views)
[![napari hub](https://img.shields.io/endpoint?url=https://api.napari-hub.org/shields/napari-orthogonal-views)](https://napari-hub.org/plugins/napari-orthogonal-views)
[![npe2](https://img.shields.io/badge/plugin-npe2-blue?link=https://napari.org/stable/plugins/index.html)](https://napari.org/stable/plugins/index.html)
[![Copier](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/copier-org/copier/master/img/badge/badge-grayscale-inverted-border-purple.json)](https://github.com/copier-org/copier)

A napari plugin for orthogonal views with synced events, crosshairs, and **interactive point-based affine registration** between two image layers.

----------------------------------

![orthoviews](https://github.com/user-attachments/assets/9d1ea326-866d-4af7-9ea6-8e56046cf6f2)

This plugin builds on the upstream [napari-orthogonal-views](https://github.com/AnniekStok/napari-orthogonal-views) by adding a **point picker** for manually selecting matched landmarks across two image layers and computing an affine transform to align them. The orthogonal views, crosshairs, and syncing infrastructure come from the original plugin; the registration workflow is the main addition.

## Installation

```
uv add https://github.com/shinzlet/napari-orthogonal-views.git
```

## Point picker & affine registration

The quickest way to get started is `show_point_picker`, which opens orthogonal views with crosshairs, zoom sync, and center sync enabled, and switches to the Point Picker tab:

```python
import napari
from napari_orthogonal_views import show_point_picker

viewer = napari.Viewer()
viewer.add_image(fixed, name="Fixed", colormap="green", blending="additive")
viewer.add_image(moving, name="Moving", colormap="magenta", blending="additive")

manager = show_point_picker(viewer, layer1_name="Fixed", layer2_name="Moving")
```

### Workflow

1. Click **Add new pair** to create a new correspondence row.
2. Navigate to a feature in the fixed image using the orthogonal views.
3. Press **T** to snap the crosshair to your mouse position.
4. Click **Update** in the fixed-image column to save that coordinate.
5. Find the same feature in the moving image and click **Update** in the moving-image column.
6. Repeat until you have at least 4 pairs (more is better).
7. Click **Apply Estimated Affine** to transform the moving image to match the fixed image.

### Programmatic access

```python
# Retrieve point pairs (dict keyed by layer name)
pairs = manager.get_registration_points()
# {"Fixed": [(z,y,x), ...], "Moving": [(z,y,x), ...]}

# Get the estimated affine matrix (4x4 homogeneous for 3D)
affine = manager.get_estimated_affine()

# Load previously saved point pairs
manager.load_registration_points(pairs)
```

See `demo_point_picker.py` for a complete runnable example with synthetic data.

## Orthogonal views

Commands are available in Views > Commands Palette (Cmd+Shift+P):
  - Show Orthogonal Views
  - Hide Orthogonal Views
  - Toggle Orthogonal Views
  - Remove Orthogonal Views

Or from the console:

```python
from napari_orthogonal_views.ortho_view_manager import show_orthogonal_views
show_orthogonal_views(viewer)
```

The view panes can be resized by dragging the splitter handles. Pressing **T** centers all views on the mouse position. All events (including label painting) are synced across views via a shared data array.

### Syncing properties

By default all layer properties are synced. For finer control, call `set_sync_filters` *before* showing orthoviews:

```python
from napari_orthogonal_views.ortho_view_manager import _get_manager
from napari.layers import Tracks, Labels

m = _get_manager(viewer)
m.set_sync_filters({
    Tracks: {"forward_exclude": "*", "reverse_exclude": "*"},
    Labels: {"forward_exclude": "contour"},
})
```

### Screen recording

The **Screen Recorder** tab can save a stitched screenshot or sweep along an axis to produce an AVI video.

## Known issues

- Deprecation warnings on `Window._qt_window`, `LayerList._get_step_size`, `LayerList._get_extent_world` (suppressed for now).
- After removing the OrthoViewManager with `delete_and_cleanup` (Remove Orthogonal Views command), the canvas may become temporarily unresponsive. Clicking outside of napari and back usually fixes this.

## Contributing

Contributions are very welcome. Tests can be run with [tox]; please ensure coverage at least stays the same before submitting a pull request.

## License

Distributed under the terms of the [BSD-3] license, "napari-orthogonal-views" is free and open source software.

## Issues

If you encounter any problems, please [file an issue](https://github.com/AnniekStok/napari-orthogonal-views/issues/) along with a detailed description.

[napari]: https://github.com/napari/napari
[copier]: https://copier.readthedocs.io/en/stable/
[BSD-3]: http://opensource.org/licenses/BSD-3-Clause
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[tox]: https://tox.readthedocs.io/en/latest/
[pip]: https://pypi.org/project/pip/
[PyPI]: https://pypi.org/
