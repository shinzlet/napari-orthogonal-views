from dataclasses import dataclass

import numpy as np
from napari.viewer import Viewer
from scipy.ndimage import affine_transform as scipy_affine_transform
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QCheckBox,
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


def estimate_affine_from_points(
        source_points: np.ndarray, target_points: np.ndarray, allow_scaling: bool
) -> np.ndarray:
    """Estimate an affine transform from matched point pairs using least squares.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.
    allow_scaling: bool
        If true, the affine matrix uses all DOFs. If false, only shear, rotation, and translation are applied.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix.

    Raises
    ------
    ValueError
        If fewer than ndim+1 point pairs are provided.
    """
    source_points = np.asarray(source_points)
    target_points = np.asarray(target_points)

    if source_points.ndim != 2 or target_points.ndim != 2:
        raise ValueError("Point arrays must be 2-dimensional (N, ndim).")
    if source_points.shape != target_points.shape:
        raise ValueError("Source and target point arrays must have the same shape.")

    n, ndim = source_points.shape
    if n < ndim + 1:
        raise ValueError(
            f"Need at least {ndim + 1} point pairs for {ndim}D affine estimation, "
            f"got {n}."
        )

    if allow_scaling:
        return _estimate_affine_with_scaling(source_points, target_points)
    else:
        return _estimate_affine_no_scaling(source_points, target_points)


def _estimate_affine_with_scaling(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    n, ndim = source_points.shape

    # Add homogeneous coordinate
    src_homo = np.hstack([source_points, np.ones((n, 1))])

    # Solve least squares: tgt = src_homo @ T.T
    T, _, _, _ = np.linalg.lstsq(src_homo, target_points, rcond=None)

    # Build full homogeneous matrix
    affine_matrix = np.eye(ndim + 1)
    affine_matrix[:ndim, :] = T.T

    return affine_matrix

def _estimate_affine_no_scaling(
    source_points: np.ndarray, target_points: np.ndarray
) -> np.ndarray:
    """Estimate an affine transform without scaling (diagonal of linear part fixed to 1).

    Solves constrained least squares column-by-column. For each target dimension j,
    the coefficient T[j,j] (the scale along axis j) is fixed to 1 and the remaining
    coefficients (off-diagonal shear/rotation + translation) are optimized.

    Parameters
    ----------
    source_points : np.ndarray
        (N, ndim) array of source coordinates.
    target_points : np.ndarray
        (N, ndim) array of target coordinates.

    Returns
    -------
    np.ndarray
        (ndim+1, ndim+1) homogeneous affine matrix with ones on the linear diagonal.
    """
    n, ndim = source_points.shape

    src_homo = np.hstack([source_points, np.ones((n, 1))])
    affine_matrix = np.eye(ndim + 1)

    for j in range(ndim):
        # Move the known diagonal contribution (1 * source_j) to the RHS
        rhs = target_points[:, j] - source_points[:, j]

        # Solve for all coefficients except the diagonal one
        other_cols = [i for i in range(ndim + 1) if i != j]
        lhs = src_homo[:, other_cols]

        coeffs, _, _, _ = np.linalg.lstsq(lhs, rhs, rcond=None)

        # affine_matrix[j, j] is already 1 from np.eye
        for k, col_idx in enumerate(other_cols):
            if col_idx < ndim:
                affine_matrix[j, col_idx] = coeffs[k]
            else:
                affine_matrix[j, ndim] = coeffs[k]  # translation

    return affine_matrix


class PointPickerWidget(QWidget):
    """Widget for picking matched point pairs across two image layers."""

    affine_applied = Signal(np.ndarray)

    def __init__(self, viewer: Viewer, layer1_name: str = "Image 1", layer2_name: str = "Image 2"):
        super().__init__()

        self.viewer = viewer
        self.layer1_name = layer1_name
        self.layer2_name = layer2_name
        self.point_pairs: dict[int, PointPair] = {}  # Use dict for stable pair_id lookup
        self._next_pair_id = 0
        self.transform_snapshot: dict | None = None
        self._applied_affine: np.ndarray | None = None
        self._padding: np.ndarray | None = None

        # Create table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Pair", layer1_name, layer2_name, ""])
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

        # Scaling checkbox
        self.allow_scaling_checkbox = QCheckBox("Allow scaling")
        self.allow_scaling_checkbox.setChecked(False)

        # Button layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.apply_button)
        button_layout.addWidget(self.reset_button)

        # Main layout
        layout = QVBoxLayout()
        layout.addWidget(self.table)
        layout.addWidget(self.add_button)
        layout.addWidget(self.allow_scaling_checkbox)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def _transform_coords(self, coords: tuple, affine: np.ndarray) -> tuple:
        """Apply a homogeneous affine matrix to step-index coordinates."""
        ndim = affine.shape[0] - 1
        p = np.array(coords[:ndim], dtype=float)
        transformed = affine[:ndim, :ndim] @ p + affine[:ndim, ndim]
        result = list(coords)  # preserve any extra dims unchanged
        result[:ndim] = np.round(transformed).astype(int).tolist()
        return tuple(result)

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

        # Column 1: fixed layer show/update widget
        layer1_widget = ShowUpdateWidget()
        layer1_widget.show_clicked.connect(
            lambda pid=pair_id: self._show_coordinate(pid, "layer1")
        )
        layer1_widget.update_clicked.connect(
            lambda pid=pair_id: self._update_coordinate(pid, "layer1")
        )
        self.table.setCellWidget(row, 1, layer1_widget)

        # Column 2: moving layer show/update widget
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
        delete_button.clicked.connect(lambda checked, pid=pair_id: self._delete_pair(pid))
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

        if layer == "layer2" and self._applied_affine is not None:
            coords = self._transform_coords(coords, self._applied_affine)

        self.viewer.dims.current_step = coords

    def _update_coordinate(self, pair_id: int, layer: str) -> None:
        """Save the current crosshair position for the given pair_id and layer."""

        if pair_id not in self.point_pairs:
            return

        current_coords = tuple(self.viewer.dims.current_step)

        if layer == "layer2" and self._applied_affine is not None:
            inv = np.linalg.inv(self._applied_affine)
            current_coords = self._transform_coords(current_coords, inv)

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

    def _unpad_coords(self, coords: tuple | None) -> tuple | None:
        """Subtract padding offset from coordinates if padding is active."""
        if coords is None or self._padding is None:
            return coords
        arr = np.array(coords)
        arr[: len(self._padding)] -= self._padding
        return tuple(arr.tolist())

    def get_point_pairs(self) -> dict:
        """Return point pairs as two parallel coordinate lists.

        When padding has been applied, coordinates are returned in the
        original (unpadded) coordinate space.

        Returns:
            Dict keyed by layer name, each a list of (z, y, x) tuples.
            Only pairs where both coordinates are set are included.
        """
        layer1 = []
        layer2 = []
        for pair in self.point_pairs.values():
            if pair.layer1_coords is not None and pair.layer2_coords is not None:
                layer1.append(self._unpad_coords(pair.layer1_coords))
                layer2.append(self._unpad_coords(pair.layer2_coords))
        return {self.layer1_name: layer1, self.layer2_name: layer2}

    def clear_pairs(self) -> None:
        """Clear all point pairs."""

        self.point_pairs.clear()
        self.table.setRowCount(0)
        self._next_pair_id = 0

    def load_point_pairs(self, pairs: dict) -> None:
        """Load point pairs into the widget.

        Parameters
        ----------
        pairs : dict
            Dict keyed by layer name, each a list of (z, y, x) coordinate
            tuples. Matches the format returned by ``get_point_pairs()``.
        """
        self.clear_pairs()
        for l1, l2 in zip(pairs[self.layer1_name], pairs[self.layer2_name]):
            self.add_pair()
            pair_id = self._next_pair_id - 1
            point_pair = self.point_pairs[pair_id]
            point_pair.layer1_coords = tuple(l1)
            point_pair.layer2_coords = tuple(l2)
        self._update_button_states()

    def apply_padding(self, fraction: float) -> None:
        """Pad both image layers with zeros so affine transforms have room to work.

        Parameters
        ----------
        fraction : float
            Fraction of the (element-wise) maximum shape to pad on each side.
        """
        if self.layer1_name not in self.viewer.layers or self.layer2_name not in self.viewer.layers:
            return

        layer1 = self.viewer.layers[self.layer1_name]
        layer2 = self.viewer.layers[self.layer2_name]

        shape1 = np.array(layer1.data.shape)
        shape2 = np.array(layer2.data.shape)
        pad_amounts = np.maximum(shape1, shape2) * fraction
        pad_amounts = np.round(pad_amounts).astype(int)

        pad_width = [(p, p) for p in pad_amounts]
        layer1.data = np.pad(layer1.data, pad_width, mode="constant")
        layer2.data = np.pad(layer2.data, pad_width, mode="constant")

        self._padding = pad_amounts

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

        # Transform the moving layer to match the fixed layer:
        # - Source points are from the moving layer (layer2_coords)
        # - Target points are from the fixed layer (layer1_coords)
        src_pts = np.array([p.layer2_coords for p in valid_pairs])
        tgt_pts = np.array([p.layer1_coords for p in valid_pairs])

        allow_scaling = self.allow_scaling_checkbox.isChecked()
        affine = estimate_affine_from_points(src_pts, tgt_pts, allow_scaling)

        return affine

    def _apply_affine(self) -> None:
        """Apply the estimated affine transform to the moving layer.

        Pre-transforms the image data using scipy rather than setting
        layer.affine, because napari does not fully support non-orthogonal
        slicing — off-diagonal affine components (shear/rotation) are
        stripped in non-displayed dimensions, causing orthoviews to render
        incorrectly.
        """

        # Check if target layer exists
        if self.layer2_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.layer2_name]

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

            # Use the fixed layer's shape as output so the result covers
            # the reference image's coordinate space.
            if self.layer1_name in self.viewer.layers:
                output_shape = self.viewer.layers[self.layer1_name].data.shape
            else:
                output_shape = self.transform_snapshot["data"].shape

            target_dtype = self.transform_snapshot["data"].dtype
            transformed = scipy_affine_transform(
                # Affine only works on float data, but we convert back after
                self.transform_snapshot["data"].astype(np.float32),
                inv_affine[:ndim, :ndim],
                offset=inv_affine[:ndim, ndim],
                output_shape=output_shape,
                order=1,
            )
            layer.data = transformed.astype(target_dtype)
            # Restore original affine (identity in most cases) so no
            # non-orthogonal slicing occurs.
            layer.affine = self.transform_snapshot["affine"]

            self._applied_affine = affine
            self.affine_applied.emit(affine)
            self._update_button_states()

    def _reset_transform(self) -> None:
        """Reset the transform of the moving layer to the snapshot state."""

        if self.transform_snapshot is None:
            return

        if self.layer2_name not in self.viewer.layers:
            return

        layer = self.viewer.layers[self.layer2_name]
        layer.data = self.transform_snapshot["data"]
        layer.affine = self.transform_snapshot["affine"]
        self.transform_snapshot = None
        self._applied_affine = None
        self._update_button_states()

    def get_estimated_affine(self) -> np.ndarray | None:
        """Get the estimated affine transform matrix.

        When padding has been applied, the affine is conjugated so that it
        operates in the original (unpadded) coordinate space.

        Returns:
            Homogeneous affine matrix or None if insufficient valid pairs.
        """

        affine = self._estimate_affine_transform()
        if affine is None or self._padding is None:
            return affine

        ndim = affine.shape[0] - 1
        pad = self._padding[:ndim].astype(float)

        # T_neg translates by -padding, T_pos by +padding
        T_neg = np.eye(ndim + 1)
        T_neg[:ndim, ndim] = -pad

        T_pos = np.eye(ndim + 1)
        T_pos[:ndim, ndim] = pad

        return T_neg @ affine @ T_pos
