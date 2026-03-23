try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .ortho_view_manager import show_point_picker
from .ortho_view_widget import OrthoViewWidget
from .point_picker_widget import PointPickerWidget, estimate_affine_from_points

__all__ = (
    "OrthoViewWidget",
    "PointPickerWidget",
    "estimate_affine_from_points",
    "show_point_picker",
)
