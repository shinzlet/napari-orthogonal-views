#!/usr/bin/env python
"""Demo script for testing the point picker UI with napari orthogonal views."""

import sys
from pathlib import Path

import napari
import numpy as np
from scipy.ndimage import affine_transform

# Add src to path so we can import the plugin without installing
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

# Import needed for the plugin to be available in napari
import napari_orthogonal_views

# Create two 3D demo images
r = 2
r2 = r ** r
N = 100

# Image 1: 3D gaussian blobs
shape = (200, 500, 500)
image1 = np.zeros(shape)

# Image 2: Similar but slightly shifted/transformed
image2 = np.zeros(shape)

# Store blob centers for auto-population
blob_centers_img1 = []
blob_centers_img2 = []

for _ in range(N):
    z = np.random.randint(2 * r, shape[0] - 2 * r)
    y = np.random.randint(2 * r, shape[1] - 2 * r)
    x = np.random.randint(2 * r, shape[2] - 2 * r)
    zz, yy, xx = np.ogrid[:shape[0], :shape[1], :shape[2]]

    # Add blobs in image 1
    mask = ((zz - z)**2 / r2 + (yy - y)**2 / r2 + (xx - x)**2 / r2) < 1
    image1[mask] = np.random.uniform(0.5, 1.0)
    blob_centers_img1.append((z, y, x))

    # Add the similar blobs in image 2 (before shear transform)
    z2, y2, x2 = z - 2, y + 3, x - 5  # shifted positions
    mask = ((zz - z2)**2 / r2 + (yy - y2)**2 / r2 + (xx - x2)**2 / r2) < 1
    image2[mask] = np.random.uniform(0.5, 1.0)
    blob_centers_img2.append((z2, y2, x2))

# Apply a small shear transformation to image2 to make registration more realistic
# Create shear matrix (small shear in y-x plane)
shear_matrix = np.array([
    [1.0, 0.0, 0.0],
    [0.01, 1.0, 0.15],
    [0.0, 0.05, 1.0]
])

# Apply the affine transformation with scipy
image2 = affine_transform(image2, shear_matrix, order=1)

# Transform the blob centers to match the shear
# Note: scipy.ndimage.affine_transform uses inverse transform for the image,
# so to get the new position of points, we need to apply the inverse of shear_matrix
shear_inverse = np.linalg.inv(shear_matrix)
blob_centers_img2_transformed = []
for z, y, x in blob_centers_img2:
    pt = np.array([z, y, x])
    pt_transformed = shear_inverse @ pt  # apply inverse to get actual position
    blob_centers_img2_transformed.append(tuple(np.round(pt_transformed).astype(int)))

# Create napari viewer
viewer = napari.Viewer()

# Add the two image layers
viewer.add_image(image1, name="Image 1", colormap="green", blending="additive")
viewer.add_image(image2, name="Image 2", colormap="magenta", blending="additive")

print("\n" + "="*70)
print("Point Picker Demo - Instructions:")
print("="*70)
print("SETUP:")
print("1. In napari, go to View > Commands Palette (Cmd+Shift+P)")
print("2. Type 'Show Orthogonal Views' and press Enter")
print("3. Check the 'Show cross hairs', 'Sync zoom', and 'Sync center' boxes")
print("")
print("USAGE:")
print("1. Look for the 'Point Picker' tab in the bottom-right controls panel")
print("2. Click 'Add new pair' to create a new point pair")
print("3. Navigate to a feature in Image 1 (use orthogonal views)")
print("4. Press 'T' to center the crosshair on your mouse position")
print("5. Click 'Update' in the 'Image 1' column to save that coordinate")
print("6. Find the corresponding point in Image 2")
print("7. Click 'Update' in the 'Image 2' column to save that coordinate")
print("8. Repeat for more point pairs (need at least 4 for registration)")
print("9. Use 'Show' buttons to revisit saved coordinates")
print("10. Use 'X' button to delete unwanted pairs")
print("")
print("REGISTRATION:")
print("11. Once you have 4+ valid pairs, click 'Apply Estimated Affine'")
print("    This will transform Image 2 to align with Image 1")
print("12. Click 'Reset Transform' to undo the transformation")
print("")
print("API ACCESS:")
print("To access points and affine programmatically from Python console:")
print("  >>> from napari_orthogonal_views.ortho_view_manager import _get_manager")
print("  >>> m = _get_manager(viewer)")
print("  >>> points = m.get_registration_points()")
print("  >>> affine = m.get_estimated_affine()  # 4x4 matrix for 3D")
print("="*70 + "\n")

# Auto-populate function (can be called after orthogonal views are shown)
def auto_populate_point_picker():
    """Auto-populate point picker with first 5 blob center pairs."""
    from napari_orthogonal_views.ortho_view_manager import _get_manager

    try:
        # Get the manager
        m = _get_manager(viewer)

        if not m.is_shown():
            print("Please show orthogonal views first (Cmd+Shift+P -> 'Show Orthogonal Views')")
            return

        # Get the point picker widget
        picker = m.point_picker_widget

        # Add first 5 point pairs
        n_pairs = max(5, len(blob_centers_img1))
        for i in range(n_pairs):
            # Add a new pair
            picker.add_pair()

            # Get the pair_id of the just-added pair
            pair_id = picker._next_pair_id - 1

            # Update coordinates for both images
            picker.point_pairs[pair_id].layer1_coords = blob_centers_img1[i]
            picker.point_pairs[pair_id].layer2_coords = blob_centers_img2_transformed[i]

        # Update button states
        picker._update_button_states()

        print(f"\n✓ Auto-populated {n_pairs} point pairs from blob centroids")
        print("  These are approximate centers - you may want to refine them")
        print("  Click 'Apply Estimated Affine' to test the registration")

    except Exception as e:
        print(f"Could not auto-populate: {e}")
        print("Make sure orthogonal views are shown first")

# Make auto_populate available in the console
import builtins
builtins.auto_populate_point_picker = auto_populate_point_picker

# Add instruction for auto-population
print("\nAUTO-POPULATE:")
print("After showing orthogonal views, run in the console:")
print("  >>> auto_populate_point_picker()")
print("This will add 5 known blob center correspondences for testing")
print("="*70 + "\n")

napari.run()
