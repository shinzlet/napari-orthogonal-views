#!/usr/bin/env python
"""Demo script for testing the point picker with real zarr data.

Loads t0c0 and t0c2 from a 5D zarr [T, C, Z, Y, X], extracts the bottom
quarter (y > 3/4), flips image 2 along Y, and pads each image with 50% of
the cropped height so the overlap region is offset vertically.
"""

import sys
from pathlib import Path

import napari
import numpy as np
import tensorstore as ts
from scipy.fft import fftn, ifftn
from scipy.ndimage import affine_transform, shift

# Add src to path so we can import the plugin without installing
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

import napari_orthogonal_views

# ── Configuration ───────────────────────────────────────────────────────
# Dense
# ZARR_PATH = "/Users/seth.hinz/Downloads/20260211_hand2GFP_H2BmSc_overlap_1891mm.ome.zarr/SN302483/561nm/pos0/1"

# Sparse
# ZARR_PATH = "/Users/seth.hinz/Downloads/20260211_hand2GFP_H2BmSc_overlap_1891mm.ome.zarr 2/SN304410/488nm/pos0/1"

# beads
ZARR_PATH = "/Users/seth.hinz/Downloads/hpc/scratch/loupe-1/imaging-2/dev/2025_10_15_daxi_test/fused.ome.zarr/1"

THETA = 0  # deskew angle in degrees, rotation about X axis in the ZY plane
# ────────────────────────────────────────────────────────────────────────

# Open the zarr store via tensorstore  [T, C, Z, Y, X]
spec = {
    "driver": "zarr",
    "kvstore": {"driver": "file", "path": ZARR_PATH},
}
dataset = ts.open(spec).result()
full_shape = dataset.shape  # (T, C, Z, Y, X)
print(f"Full dataset shape (T,C,Z,Y,X): {full_shape}")

# Load t=0, c=0 and t=0, c=2 as numpy arrays  → (Z, Y, X)
image1 = dataset[0, 0].read().result()
image2 = dataset[0, 2].read().result()
print(f"Loaded image1 (t0c0): {image1.shape}, image2 (t0c2): {image2.shape}")

# ── Deskew both volumes by THETA about the X axis (rotation in ZY plane) ─
def deskew(vol, theta_deg):
    """Deskew a ZYX volume by rotating theta degrees about the X axis.

    Builds a 3×3 rotation matrix that acts on the Z and Y axes while
    leaving X unchanged, then resamples with scipy.ndimage.affine_transform.
    The output shape is enlarged to contain the full rotated volume.
    """
    theta = np.radians(theta_deg)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation in ZY plane (about X): Z' = c*Z - s*Y, Y' = s*Z + c*Y
    R = np.array([
        [ c, -s, 0],
        [ s,  c, 0],
        [ 0,  0, 1],
    ])
    # Compute output shape that contains the whole rotated volume
    in_shape = np.array(vol.shape, dtype=float)
    corners = np.array(np.meshgrid(*[[0, s - 1] for s in vol.shape])).reshape(3, -1).T
    rotated_corners = corners @ R.T
    out_min = rotated_corners.min(axis=0)
    out_max = rotated_corners.max(axis=0)
    out_shape = np.ceil(out_max - out_min + 1).astype(int)
    # Offset so that the rotated origin maps to (0,0,0) in the output
    center_in = (in_shape - 1) / 2
    center_out = (out_shape - 1) / 2
    offset = center_in - R @ center_out
    return affine_transform(vol, R, offset=offset, output_shape=tuple(out_shape), order=1)

if abs(THETA) > 0.1:
    print(f"Deskewing both volumes by {THETA}° about the X axis...")
    image1 = deskew(image1, THETA)
    image2 = deskew(image2, THETA)
    print(f"After deskew: {image1.shape}")

# Extract the bottom fraction
frac = 0.1
y_cut = int((1 - frac) * image1.shape[1])
image1 = image1[:, y_cut:, :]
image2 = image2[:, y_cut:, :]
print(f"After Y crop (bottom quarter): {image1.shape}")

# Flip image 2 along the Y axis (axis=1 in ZYX)
image2 = np.flip(image2, axis=1).copy()

# Pad each image with a percent of cropped height in zeros along Y:
#   image1: zeros added below (+Y side)
#   image2: zeros added above (-Y side, i.e. at the start)
pad_y = int(image1.shape[1] * 0.3)
z, y, x = image1.shape

pad_above = np.zeros((z, pad_y, x), dtype=image1.dtype)
pad_below = np.zeros((z, pad_y, x), dtype=image2.dtype)

image1 = np.concatenate([pad_above, image1], axis=1)  # zeros below
image2 = np.concatenate([image2, pad_below], axis=1)  # zeros above
print(f"After padding: image1 {image1.shape}, image2 {image2.shape}")

# ── Rough alignment via 3D ZNCC (phase correlation on normalised images) ─
def zncc_shift(fixed, moving):
    """Estimate the integer shift that maximises zero-mean normalised
    cross-correlation between *fixed* and *moving* using the FFT."""
    f = fixed.astype(np.float64)
    m = moving.astype(np.float64)
    f -= f.mean()
    m -= m.mean()
    f_std = f.std()
    m_std = m.std()
    if f_std == 0 or m_std == 0:
        return np.zeros(f.ndim)
    f /= f_std
    m /= m_std
    cc = np.real(ifftn(fftn(f) * np.conj(fftn(m))))
    peak = np.unravel_index(np.argmax(cc), cc.shape)
    # Wrap around: shifts > half the axis length are negative
    shifts = np.array(peak, dtype=float)
    for i, s in enumerate(shifts):
        if s > cc.shape[i] // 2:
            shifts[i] -= cc.shape[i]
    return shifts

# Shift (disabled for now)
# print("Computing ZNCC shift for rough alignment...")
# rough_shift = zncc_shift(image1, image2)
# print(f"Rough shift (ZYX): {rough_shift}")
# image2 = shift(image2, rough_shift, order=1)
# print("Applied rough shift to image2")

# ── Launch napari with the plugin ───────────────────────────────────────
viewer = napari.Viewer()
viewer.add_image(image1, name="Image 1", colormap="green", blending="additive")
viewer.add_image(image2, name="Image 2", colormap="magenta", blending="additive")

print("\n" + "=" * 70)
print("Zarr Point Picker Demo")
print("=" * 70)
print("1. Show Orthogonal Views: Cmd+Shift+P → 'Show Orthogonal Views'")
print("2. Use the 'Point Picker' tab to mark corresponding features")
print("3. After 4+ pairs, click 'Apply Estimated Affine'")
print("")
print("API ACCESS:")
print("  >>> from napari_orthogonal_views.ortho_view_manager import _get_manager")
print("  >>> m = _get_manager(viewer)")
print("  >>> points = m.get_registration_points()")
print("  >>> affine = m.get_estimated_affine()")
print("=" * 70 + "\n")

napari.run()
