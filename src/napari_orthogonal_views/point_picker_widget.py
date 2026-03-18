from dataclasses import dataclass

import numpy as np
from napari.viewer import Viewer
from scipy.ndimage import affine_transform as scipy_affine_transform
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QHBoxLayout,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)


@dataclass
class PointPair:
    """Store a matched pair of coordinates for registration."""

    pair_id: int
    layer1_coords: tuple | None = None
    layer2_coords: tuple | None = None


class ShowUpdateWidget(QWidget):
    """Widget with Show and Update buttons for a single coordinate."""

    show_clicked = Signal()
    update_clicked = Signal()

    def __init__(self):
        super().__init__()

        self.show_button = QPushButton("Show")
        self.update_button = QPushButton("Update")

        self.show_button.clicked.connect(self.show_clicked.emit)
        self.update_button.clicked.connect(self.update_clicked.emit)

        layout = QHBoxLayout()
        layout.addWidget(self.show_button)
        layout.addWidget(self.update_button)
        layout.setContentsMargins(2, 2, 2, 2)
        self.setLayout(layout)


class PointPickerWidget(QWidget):
    """Widget for picking matched point pairs across two image layers."""

    affine_applied = Signal(np.ndarray)

    def __init__(self, viewer: Viewer):
        super().__init__()

        self.viewer = viewer
        self.point_pairs: dict[int, PointPair] = {}  # Use dict for stable pair_id lookup
        self._next_pair_id = 0
        self.transform_snapshot: dict | None = None

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Pair", "Image 1", "Image 2", ""])
        self.table.horizontalHeader().setStretchLastSection(False)
        self.table.setColumnWidth(0, 80)
        self.table.setColumnWidth(1, 150)
        self.table.setColumnWidth(2, 150)
        self.table.setColumnWidth(3, 30)

        # Add new pair button
        self.add_button = QPushButton("Add new pair")
        self.add_button.clicked.connect(self.add_pair)

        # Transform buttons
        self.apply_button = QPushButton("Apply Estimated Affine")
        self.apply_button.clicked.connect(self._apply_affine)
        self.apply_button.setEnabled(False)

        self.reset_button = QPushButton("Reset Transform")
        self.reset_button.clicked.connect(self._reset_transform)
        self.reset_button.setEnabled(False)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.add_button)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def add_pair(self) -> None:
        """Add a new point pair row to the table."""

        pair_id = self._next_pair_id
        self._next_pair_id += 1

        point_pair = PointPair(pair_id=pair_id)
        self.point_pairs[pair_id] = point_pair

        row = self.table.rowCount()
        self.table.insertRow(row)

        # Column 0: Pair name
        pair_item = QTableWidgetItem(f"Pair {pair_id}")
        pair_item.setFlags(pair_item.flags() & ~Qt.ItemIsEditable)
        pair_item.setData(Qt.UserRole, pair_id)  # Store pair_id in the item
        self.table.setItem(row, 0, pair_item)

        # Column 1: Image 1 show/update widget
        layer1_widget = ShowUpdateWidget()
        layer1_widget.show_clicked.connect(
            lambda pid=pair_id: self._show_coordinate(pid, "layer1")
        )
        layer1_widget.update_clicked.connect(
            lambda pid=pair_id: self._update_coordinate(pid, "layer1")
        )
        self.table.setCellWidget(row, 1, layer1_widget)

        # Column 2: Image 2 show/update widget
        layer2_widget = ShowUpdateWidget()
        layer2_widget.show_clicked.connect(
            lambda pid=pair_id: self._show_coordinate(pid, "layer2")
        )
        layer2_widget.update_clicked.connect(
            lambda pid=pair_id: self._update_coordinate(pid, "layer2")
        )
        self.table.setCellWidget(row, 2, layer2_widget)

        # Column 3: Delete button
        delete_button = QPushButton("×")
        delete_button.setMaximumWidth(30)
        delete_button.clicked.connect(lambda pid=pair_id: self._delete_pair(pid))
        self.table.setCellWidget(row, 3, delete_button)

        # Update button states
        self._update_button_states()

    def _show_coordinate(self, pair_id: int, layer: str) -> None:
        """Move crosshair to the saved coordinate for the given pair_id and layer."""

        if pair_id not in self.point_pairs:
            return

        point_pair = self.point_pairs[pair_id]
        coords = (
            point_pair.layer1_coords
            if layer == "layer1"
            else point_pair.layer2_coords
        )

        if coords is None:
            # No coordinate saved yet, default to origin
            coords = tuple([0] * self.viewer.dims.ndim)

        self.viewer.dims.current_step = coords

    def _update_coordinate(self, pair_id: int, layer: str) -> None:
        """Save the current crosshair position for the given pair_id and layer."""

        if pair_id not in self.point_pairs:
            return

        current_coords = tuple(self.viewer.dims.current_step)
        point_pair = self.point_pairs[pair_id]

        if layer == "layer1":
            point_pair.layer1_coords = current_coords
        else:
            point_pair.layer2_coords = current_coords

        # Update button states
        self._update_button_states()

    def _delete_pair(self, pair_id: int) -> None:
        """Delete the point pair with the given pair_id."""

        if pair_id not in self.point_pairs:
            return

        # Remove from data
        del self.point_pairs[pair_id]

        # Find and remove the row from the table
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.data(Qt.UserRole) == pair_id:
                self.table.removeRow(row)
                break

        # Update button states
        self._update_button_states()

    def get_point_pairs(self) -> list[dict]:
        """Return the list of point pairs as dictionaries.

        Returns:
            List of dicts with keys 'pair_id', 'layer1_coords', 'layer2_coords'
        """

        return [
            {
                "pair_id": pair.pair_id,
                "layer1_coords": pair.layer1_coords,
                "layer2_coords": pair.layer2_coords,
            }
            for pair in self.point_pairs.values()
        ]

    def clear_pairs(self) -> None:
        """Clear all point pairs."""

        self.point_pairs.clear()
        self.table.setRowCount(0)
        self._next_pair_id = 0

    def _update_button_states(self) -> None:
        """Update the enabled state of the Apply and Reset buttons."""

        # Get valid pairs (both coordinates set)
        valid_pairs = [
            p
            for p in self.point_pairs.values()
            if p.layer1_coords is not None and p.layer2_coords is not None
        ]

        # Enable Apply button if we have at least 4 valid pairs
        self.apply_button.setEnabled(len(valid_pairs) >= 4)

        # Enable Reset button if we have a snapshot
        self.reset_button.setEnabled(self.transform_snapshot is not None)

    def _estimate_affine_transform(self) -> np.ndarray | None:
        """Estimate affine transform from point pairs using least squares.

        Returns:
            Homogeneous affine matrix (4x4 for 3D, 3x3 for 2D) or None if insufficient pairs.
        """

        # Get valid pairs
        valid_pairs = [
            p
            for p in self.point_pairs.values()
            if p.layer1_coords is not None and p.layer2_coords is not None
        ]

        if len(valid_pairs) < 4:
            return None

        n = len(valid_pairs)
        ndim = len(valid_pairs[0].layer1_coords)

        # Build source and target point matrices
        # We want to transform Image 2 to match Image 1, so:
        # - Source points are from Image 2 (layer2_coords)
        # - Target points are from Image 1 (layer1_coords)
        src_pts = np.array([p.layer2_coords for p in valid_pairs])  # Image 2 points
        tgt_pts = np.array([p.layer1_coords for p in valid_pairs])  # Image 1 points

        # Add homogeneous coordinate to source points
        src_homo = np.hstack([src_pts, np.ones((n, 1))])

        # Solve least squares: tgt = src_homo @ T.T
        # We solve for T.T because napari uses row vectors
        T, residuals, rank, s = np.linalg.lstsq(src_homo, tgt_pts, rcond=None)

        # Build full homogeneous matrix
        affine_matrix = np.eye(ndim + 1)
        affine_matrix[:ndim, :] = T.T

        return affine_matrix

    def _apply_affine(self) -> None:
        """Apply the estimated affine transform to Image 2 layer.

        Pre-transforms the image data using scipy rather than setting
        layer.affine, because napari does not fully support non-orthogonal
        slicing — off-diagonal affine components (shear/rotation) are
        stripped in non-displayed dimensions, causing orthoviews to render
        incorrectly.
        """

        # Check if Image 2 layer exists
        if "Image 2" not in self.viewer.layers:
            return

        layer = self.viewer.layers["Image 2"]

        # Snapshot current state if not already done
        if self.transform_snapshot is None:
            self.transform_snapshot = {
                "affine": layer.affine.affine_matrix.copy(),
                "data": layer.data,
            }

        # Compute affine
        affine = self._estimate_affine_transform()
        if affine is not None:
            # Pre-transform the data so the layer can keep an identity affine.
            inv_affine = np.linalg.inv(affine)
            ndim = affine.shape[0] - 1

            # Use Image 1's shape as output so the result covers the
            # reference image's coordinate space.
            if "Image 1" in self.viewer.layers:
                output_shape = self.viewer.layers["Image 1"].data.shape
            else:
                output_shape = self.transform_snapshot["data"].shape

            transformed = scipy_affine_transform(
                self.transform_snapshot["data"],
                inv_affine[:ndim, :ndim],
                offset=inv_affine[:ndim, ndim],
                output_shape=output_shape,
                order=1,
            )
            layer.data = transformed
            # Restore original affine (identity in most cases) so no
            # non-orthogonal slicing occurs.
            layer.affine = self.transform_snapshot["affine"]

            self.affine_applied.emit(affine)
            self._update_button_states()

    def _reset_transform(self) -> None:
        """Reset the transform of Image 2 layer to the snapshot state."""

        if self.transform_snapshot is None:
            return

        if "Image 2" not in self.viewer.layers:
            return

        layer = self.viewer.layers["Image 2"]
        layer.data = self.transform_snapshot["data"]
        layer.affine = self.transform_snapshot["affine"]
        self.transform_snapshot = None
        self._update_button_states()

    def get_estimated_affine(self) -> np.ndarray | None:
        """Get the estimated affine transform matrix.

        Returns:
            Homogeneous affine matrix or None if insufficient valid pairs.
        """

        return self._estimate_affine_transform()
