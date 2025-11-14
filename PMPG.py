import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import glob
import os
import sys
from scipy import ndimage
from skimage import measure

# Use non-interactive backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# User parameters
# ----------------------------------------------------------
data_dir = "results/2rectangleSlow"# <-- change this, ambench25ch7: "results/2squareFast", "results/2squareSlow", "results/2rectangleSlow", "results/2rectangleFast"
x_section = 3.06                   # Cross-section location in mm (None = auto-center) ambench25ch7: (2.96, 5.045), (3.06)
melt_T = 1570.0                    # K
tol_x = 0.1                        # tolerance for finding x cross-section (mm)

# Set to None to process ALL files, or to an int to limit for testing
max_files = None 

output_png = "cross_section_PMPG.png"
output_npy = "PMPG_envelope.npy"
# ----------------------------------------------------------

# ----------------------------------------------------------
# Collect Level3 files
# ----------------------------------------------------------
pattern = os.path.join(data_dir, "Level3*.vtr")
all_files = sorted(glob.glob(pattern))
if not all_files:
    raise RuntimeError(f"No Level3*.vtr files found in {data_dir}")

if max_files is not None and len(all_files) > max_files:
    print(f"Found {len(all_files)} Level3 files, limiting to first {max_files}.")
    files = all_files[:max_files]
else:
    print(f"Found {len(all_files)} Level3 files, processing all of them.")
    files = all_files

# ----------------------------------------------------------
# Helper: load VTR file -> vtkRectilinearGrid
# ----------------------------------------------------------
def load_vtr(fname):
    reader = vtk.vtkXMLRectilinearGridReader()
    reader.SetFileName(fname)
    reader.Update()
    return reader.GetOutput()

# ----------------------------------------------------------
# PASS 1: Determine global Y-Z grid extents
# ----------------------------------------------------------
print("\n=== PASS 1: Scanning to determine Y-Z grid extents ===")

min_y = np.inf
max_y = -np.inf
min_z = np.inf
max_z = -np.inf

# Also determine dy, dz from first file
first_grid = load_vtr(files[0])
x_coords_vtk = first_grid.GetXCoordinates()
y_coords_vtk = first_grid.GetYCoordinates()
z_coords_vtk = first_grid.GetZCoordinates()

x_coords0 = np.array([x_coords_vtk.GetValue(i) for i in range(x_coords_vtk.GetNumberOfTuples())])
y_coords0 = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])
z_coords0 = np.array([z_coords_vtk.GetValue(k) for k in range(z_coords_vtk.GetNumberOfTuples())])

if len(y_coords0) < 2 or len(z_coords0) < 2:
    raise RuntimeError("Not enough points to infer dy/dz from first file.")

# Auto-determine x_section if not specified
if x_section is None:
    x_section = (x_coords0.min() + x_coords0.max()) / 2.0
    print(f"Auto-selected X cross-section: {x_section:.3f} mm (center of domain)")

dy = y_coords0[1] - y_coords0[0]
dz = z_coords0[1] - z_coords0[0]
print(f"Inferred spacing: dy = {dy}, dz = {dz}")

total_files = len(files)
for idx, f in enumerate(files):
    if idx % max(1, total_files // 20) == 0:
        print(f"  [PASS1] {idx}/{total_files} files scanned...")
        sys.stdout.flush()

    grid = load_vtr(f)
    y_coords_vtk = grid.GetYCoordinates()
    z_coords_vtk = grid.GetZCoordinates()

    y_coords = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])
    z_coords = np.array([z_coords_vtk.GetValue(k) for k in range(z_coords_vtk.GetNumberOfTuples())])

    min_y = min(min_y, y_coords[0])
    max_y = max(max_y, y_coords[-1])
    min_z = min(min_z, z_coords[0])
    max_z = max(max_z, z_coords[-1])

# Snap to grid
min_y = round(min_y / dy) * dy
min_z = round(min_z / dz) * dz

Ny_global = int(round((max_y - min_y) / dy)) + 1
Nz_global = int(round((max_z - min_z) / dz)) + 1

print(f"\nGlobal Y-Z grid:")
print(f"  Ny = {Ny_global}, Nz = {Nz_global}")
print(f"  Y range: {min_y} → {max_y}")
print(f"  Z range: {min_z} → {max_z}\n")

# ----------------------------------------------------------
# PASS 2: Build cumulative melt pool envelope AND track individual tracks
# ----------------------------------------------------------
print(f"=== PASS 2: Building melt pool envelope at x = {x_section:.3f} mm ===")

# Initialize cumulative envelope: will store maximum temperature ever seen at each location
envelope = np.full((Nz_global, Ny_global), -np.inf, dtype=np.float32)

# Initialize for tracking individual tracks
current_track_envelope = np.full((Nz_global, Ny_global), -np.inf, dtype=np.float32)
track_envelopes = []  # List to store each track's envelope
previous_has_molten = False
track_gap_threshold = 0.5  # mm - if melt pool center moves more than this, consider it a new track

for t_idx, f in enumerate(files):
    print(f"  [PASS2] timestep {t_idx+1}/{total_files}: {os.path.basename(f)}")
    sys.stdout.flush()

    grid = load_vtr(f)
    Nx_loc, Ny_loc, Nz_loc = grid.GetDimensions()

    # Get coordinates
    x_coords_vtk = grid.GetXCoordinates()
    y_coords_vtk = grid.GetYCoordinates()
    z_coords_vtk = grid.GetZCoordinates()

    x_coords = np.array([x_coords_vtk.GetValue(i) for i in range(x_coords_vtk.GetNumberOfTuples())])
    y_coords = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])
    z_coords = np.array([z_coords_vtk.GetValue(k) for k in range(z_coords_vtk.GetNumberOfTuples())])

    # Find the x-index closest to x_section
    i_section = None
    min_dist = np.inf
    for i, x_val in enumerate(x_coords):
        dist = abs(x_val - x_section)
        if dist < min_dist:
            min_dist = dist
            i_section = i
    
    if i_section is None or min_dist > tol_x:
        print(f"    Warning: x = {x_section:.3f} mm not found in this file (closest: {x_coords[i_section]:.3f} mm), skipping...")
        continue

    # Get temperature array
    temps_vtk = grid.GetPointData().GetArray("Temperature (K)")
    if temps_vtk is None:
        raise RuntimeError(f"'Temperature (K)' array not found in {f}")
    temps_np = vtk_to_numpy(temps_vtk)

    # Reshape to (Nz, Ny, Nx)
    temps_3d = temps_np.reshape((Nz_loc, Ny_loc, Nx_loc), order="C")

    # Extract cross-section at i_section
    T_section = temps_3d[:, :, i_section]  # (Nz_loc, Ny_loc)

    # Map this local window to global Y-Z grid
    y0 = y_coords[0]
    z0 = z_coords[0]

    jy_start = int(round((y0 - min_y) / dy))
    kz_start = int(round((z0 - min_z) / dz))

    jy_end = jy_start + Ny_loc
    kz_end = kz_start + Nz_loc

    if jy_start < 0 or kz_start < 0 or jy_end > Ny_global or kz_end > Nz_global:
        print(f"    Warning: Local window out of global bounds, skipping...")
        continue

    # Check if there's molten material in this timestep
    has_molten = np.any(T_section > melt_T)
    
    # Detect track transition: if we had molten material before but now there's a gap
    if previous_has_molten and not has_molten:
        # Save current track and start new one
        if np.any(current_track_envelope > melt_T):
            track_envelopes.append(current_track_envelope.copy())
            print(f"    Track {len(track_envelopes)} completed at timestep {t_idx}")
        current_track_envelope = np.full((Nz_global, Ny_global), -np.inf, dtype=np.float32)
    
    previous_has_molten = has_molten
    
    # Store maximum temperature at each location across all timesteps
    for kz_loc in range(Nz_loc):
        for jy_loc in range(Ny_loc):
            jy_global = jy_start + jy_loc
            kz_global = kz_start + kz_loc
            
            T_curr = T_section[kz_loc, jy_loc]
            
            # Update cumulative envelope
            envelope[kz_global, jy_global] = max(envelope[kz_global, jy_global], T_curr)
            
            # Update current track envelope
            current_track_envelope[kz_global, jy_global] = max(current_track_envelope[kz_global, jy_global], T_curr)

# Save the last track if it exists
if np.any(current_track_envelope > melt_T):
    track_envelopes.append(current_track_envelope.copy())
    print(f"    Track {len(track_envelopes)} completed (final)")

print(f"\nTotal tracks detected: {len(track_envelopes)}")

# ----------------------------------------------------------
# Save results
# ----------------------------------------------------------
np.save(output_npy, envelope)
print(f"\nSaved envelope array: {output_npy}")

# ----------------------------------------------------------
# Extract accurate contour and compute metrics
# ----------------------------------------------------------
print("\n=== Extracting accurate melt boundary contour ===")

y_coords_global = min_y + dy * np.arange(Ny_global)
z_coords_global = min_z + dz * np.arange(Nz_global)

# Create binary mask of molten region
molten_mask = envelope > melt_T

# Find contours at melt temperature using marching squares
# Need to handle coordinates properly for accurate area calculation
try:
    contours = measure.find_contours(envelope, melt_T)
    print(f"Found {len(contours)} contour(s)")
    
    # Convert contour indices to actual coordinates
    contours_mm = []
    for contour in contours:
        # contour has shape (N, 2) where columns are (z_index, y_index)
        z_indices = contour[:, 0]
        y_indices = contour[:, 1]
        
        # Convert to mm
        z_mm = min_z + z_indices * dz
        y_mm = min_y + y_indices * dy
        
        contours_mm.append(np.column_stack([y_mm, z_mm]))
    
    # Calculate accurate area using shoelace formula for largest contour
    if contours_mm:
        largest_contour = max(contours_mm, key=len)
        y_contour = largest_contour[:, 0]
        z_contour = largest_contour[:, 1]
        
        # Shoelace formula for polygon area
        molten_area_accurate = 0.5 * np.abs(
            np.dot(y_contour[:-1], z_contour[1:]) - np.dot(z_contour[:-1], y_contour[1:]) +
            y_contour[-1] * z_contour[0] - z_contour[-1] * y_contour[0]
        )
    else:
        molten_area_accurate = 0.0
        
except ValueError:
    print("No contours found at melt temperature")
    contours_mm = []
    molten_area_accurate = 0.0

# Simple area estimate from grid
molten_area_grid = np.sum(molten_mask) * dy * dz

# Calculate melt pool depth and width
if np.sum(molten_mask) > 0:
    # Find extent of molten region
    molten_z_indices = np.where(np.any(molten_mask, axis=1))[0]
    molten_y_indices = np.where(np.any(molten_mask, axis=0))[0]
    
    max_depth = z_coords_global[molten_z_indices[0]]  # Lowest Z (most negative)
    max_width = y_coords_global[molten_y_indices[-1]] - y_coords_global[molten_y_indices[0]]
else:
    max_depth = 0.0
    max_width = 0.0

# ----------------------------------------------------------
# Create visualization with last track overlay
# ----------------------------------------------------------
print("\n=== Generating PNG visualization ===")

plt.figure(figsize=(12, 8))

# Set up the plot axes with proper limits
ax = plt.gca()
ax.set_xlim(y_coords_global[0], y_coords_global[-1])
ax.set_ylim(z_coords_global[0], z_coords_global[-1])
ax.set_aspect('equal')

# Base layer: all molten regions in red using smooth contour fill
if contours_mm:
    for contour in contours_mm:
        plt.fill(contour[:, 0], contour[:, 1], facecolor=[0.8, 0.3, 0.3], edgecolor='none', alpha=0.9)

# Overlay: last track using smooth contour-filled region (dark gray)
if len(track_envelopes) >= 1:
    last_track_idx = len(track_envelopes) - 1
    last_track_env = track_envelopes[last_track_idx]
    
    try:
        last_track_contours = measure.find_contours(last_track_env, melt_T)
        if len(last_track_contours) > 0:
            # Only fill the single largest contour (main melt pool)
            # This avoids double-plotting and overlays
            largest_contour = max(last_track_contours, key=len)
            z_indices = largest_contour[:, 0]
            y_indices = largest_contour[:, 1]
            
            # Convert to mm coordinates
            z_mm = min_z + z_indices * dz
            y_mm = min_y + y_indices * dy
            
            # Fill the contour with dark gray - use full opacity and ensure single fill
            plt.fill(y_mm, z_mm, facecolor=[0.3, 0.3, 0.3], edgecolor='none', alpha=1.0, label=f'Last track (#{last_track_idx + 1})')
    except (ValueError, IndexError):
        pass  # No contour found for last track

# Overlay accurate contours
if contours_mm:
    for idx, contour in enumerate(contours_mm):
        # Only label the first contour to avoid duplicates
        label = 'Melt boundary' if idx == 0 else None
        plt.plot(contour[:, 0], contour[:, 1], 'k-', linewidth=1, label=label)

plt.legend(loc='upper left', bbox_to_anchor=(1.02, 0), fontsize=16)
plt.xlabel("y (mm)", fontsize=20)
plt.ylabel("z (mm)", fontsize=18)
plt.tick_params(axis='both', which='major', labelsize=18)
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
# Trim 1.25mm from left and 1.125mm from right of y-axis
plt.xlim(y_coords_global[0] + 1.25, y_coords_global[-1] - 1.125)
plt.savefig(output_png, dpi=300, bbox_inches="tight")
print(f"Saved PNG: {output_png}")
plt.close()

# ----------------------------------------------------------
# Print statistics
# ----------------------------------------------------------
total_nodes = Ny_global * Nz_global
molten_nodes = np.sum(molten_mask)
print(f"\n=== Statistics ===")
print(f"Cross-section location: X = {x_section:.3f} mm")
print(f"Molten nodes: {molten_nodes} / {total_nodes} ({100*molten_nodes/total_nodes:.2f}%)")
print(f"Molten area (grid-based): {molten_area_grid:.6f} mm²")
print(f"Molten area (contour-based): {molten_area_accurate:.6f} mm²")
print(f"Grid spacing: dy = {dy*1e3:.2f} μm, dz = {dz*1e3:.2f} μm")
print(f"\n=== Melt Pool Geometry ===")
print(f"Maximum depth: {max_depth*1e3:.2f} μm")
print(f"Maximum width: {max_width:.3f} mm")

# ----------------------------------------------------------
# Analyze depth profile: valleys and peaks across Y direction
# ----------------------------------------------------------
print("\n=== Analyzing depth profile for valleys and peaks ===")

if np.sum(molten_mask) > 0:
    # For each Y location, find the minimum (deepest) Z where material was molten
    depth_profile = np.full(Ny_global, np.nan)
    
    for jy in range(Ny_global):
        # Find all z-indices where material was molten in this column
        z_column = molten_mask[:, jy]
        molten_z_indices = np.where(z_column)[0]
        
        if len(molten_z_indices) > 0:
            # Take the minimum (deepest, most negative) z value
            kz_min = molten_z_indices[0]  # Lowest index = most negative Z
            depth_profile[jy] = z_coords_global[kz_min]
    
    # Remove NaN values for analysis
    valid_depth = depth_profile[~np.isnan(depth_profile)]
    valid_y_depth = y_coords_global[~np.isnan(depth_profile)]
    
    if len(valid_depth) > 5:  # Need enough points for meaningful analysis
        # Smooth the profile to reduce noise
        from scipy.ndimage import gaussian_filter1d
        smoothed_depth = gaussian_filter1d(valid_depth, sigma=2)
        
        # Find local minima (valleys - deepest penetration) and maxima (peaks - shallowest, overlap regions)
        diff = np.diff(smoothed_depth)
        sign_changes = np.diff(np.sign(diff))
        
        valley_indices = np.where(sign_changes > 0)[0] + 1  # Local minima (deepest points)
        peak_indices = np.where(sign_changes < 0)[0] + 1    # Local maxima (shallowest points)
        
        if len(valley_indices) > 0:
            valley_depths = smoothed_depth[valley_indices]
            avg_valley_depth = np.mean(np.abs(valley_depths))  # Absolute value
            print(f"Found {len(valley_indices)} valleys (deep penetration points)")
            print(f"Average valley depth: {avg_valley_depth*1e3:.2f} μm")
            print(f"Valley depths: {[f'{abs(v)*1e3:.2f}' for v in valley_depths]} μm")
        else:
            avg_valley_depth = 0.0
            print("No valleys found in depth profile")
        
        if len(peak_indices) > 0:
            peak_depths = smoothed_depth[peak_indices]
            avg_peak_depth = np.mean(np.abs(peak_depths))  # Absolute value
            print(f"\nFound {len(peak_indices)} peaks (shallow overlap points)")
            print(f"Average peak depth: {avg_peak_depth*1e3:.2f} μm")
            print(f"Peak depths: {[f'{abs(p)*1e3:.2f}' for p in peak_depths]} μm")
        else:
            avg_peak_depth = 0.0
            print("\nNo peaks found in depth profile")
        
        # Calculate valley-peak difference if both exist
        if len(valley_indices) > 0 and len(peak_indices) > 0:
            depth_variation = avg_valley_depth - avg_peak_depth
            print(f"\nDepth variation (valley - peak): {depth_variation*1e3:.2f} μm")
        
    else:
        print("Not enough data points for depth profile analysis")
else:
    print("No molten region found for depth profile analysis")

# ----------------------------------------------------------
# Analyze individual track widths using accurate contour interpolation
# ----------------------------------------------------------
print("\n=== Analyzing individual track widths ===")

track_widths = []
track_contours_list = []  # Store contours for each track

for track_idx, track_env in enumerate(track_envelopes):
    track_mask = track_env > melt_T
    
    if np.sum(track_mask) > 0:
        # Find accurate contours using marching squares
        try:
            track_contours = measure.find_contours(track_env, melt_T)
            
            if len(track_contours) > 0:
                # Convert to mm coordinates
                track_contours_mm = []
                for contour in track_contours:
                    z_indices = contour[:, 0]
                    y_indices = contour[:, 1]
                    z_mm = min_z + z_indices * dz
                    y_mm = min_y + y_indices * dy
                    track_contours_mm.append(np.column_stack([y_mm, z_mm]))
                
                track_contours_list.append(track_contours_mm)
                
                # Find width from the largest contour (main melt pool boundary)
                largest_contour = max(track_contours_mm, key=len)
                y_values = largest_contour[:, 0]
                
                # Accurate width from interpolated contour
                track_width = np.max(y_values) - np.min(y_values)
                track_widths.append(track_width)
                print(f"Track {track_idx + 1}: width = {track_width:.4f} mm ({track_width*1e3:.2f} μm)")
            else:
                track_widths.append(0.0)
                track_contours_list.append([])
                print(f"Track {track_idx + 1}: No contour found")
                
        except ValueError:
            # No contour found at melt temperature
            track_widths.append(0.0)
            track_contours_list.append([])
            print(f"Track {track_idx + 1}: No contour found")
    else:
        track_widths.append(0.0)
        track_contours_list.append([])
        print(f"Track {track_idx + 1}: No molten region")

if len(track_widths) > 0:
    valid_widths = [w for w in track_widths if w > 0]
    if valid_widths:
        avg_track_width = np.mean(valid_widths)
        print(f"\nAverage track width (interpolated): {avg_track_width:.4f} mm ({avg_track_width*1e3:.2f} μm)")
    else:
        avg_track_width = 0.0
        print("\nNo valid track widths found")
else:
    avg_track_width = 0.0
    print("\nNo tracks found for width analysis")

# ----------------------------------------------------------
# Save summary statistics and detailed data to text file
# ----------------------------------------------------------
output_stats_file = "statistics_PMPG.txt"

with open(output_stats_file, 'w') as f:
    f.write("=" * 70 + "\n")
    f.write("PMPG (Pad Melt Pool Geometry) Analysis Results\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("SUMMARY STATISTICS\n")
    f.write("-" * 70 + "\n")
    
    # Average values
    if 'avg_valley_depth' in locals() and avg_valley_depth > 0:
        f.write(f"Average Depth (Valley):        {avg_valley_depth*1e3:.2f} μm\n")
    else:
        f.write(f"Average Depth (Valley):        N/A\n")
    
    if 'avg_peak_depth' in locals() and avg_peak_depth > 0:
        f.write(f"Average Overlap Depth (Peak):  {avg_peak_depth*1e3:.2f} μm\n")
    else:
        f.write(f"Average Overlap Depth (Peak):  N/A\n")
    
    if avg_track_width > 0:
        f.write(f"Average Track Width:           {avg_track_width*1e3:.2f} μm\n")
    else:
        f.write(f"Average Track Width:           N/A\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Detailed depth data (valleys)
    f.write("DETAILED DATA: Valley Depths (μm)\n")
    f.write("-" * 70 + "\n")
    if 'valley_depths' in locals() and len(valley_depths) > 0:
        for i, depth in enumerate(valley_depths):
            f.write(f"Valley {i+1:3d}:  {abs(depth)*1e3:8.2f} μm\n")
    else:
        f.write("No valley data available\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Detailed overlap depth data (peaks)
    f.write("DETAILED DATA: Peak Depths / Overlap Depths (μm)\n")
    f.write("-" * 70 + "\n")
    if 'peak_depths' in locals() and len(peak_depths) > 0:
        for i, depth in enumerate(peak_depths):
            f.write(f"Peak {i+1:3d}:  {abs(depth)*1e3:8.2f} μm\n")
    else:
        f.write("No peak data available\n")
    
    f.write("\n" + "=" * 70 + "\n\n")
    
    # Detailed track width data
    f.write("DETAILED DATA: Track Widths (μm)\n")
    f.write("-" * 70 + "\n")
    if len(track_widths) > 0:
        for i, width in enumerate(track_widths):
            if width > 0:
                f.write(f"Track {i+1:3d}:  {width*1e3:8.2f} μm\n")
            else:
                f.write(f"Track {i+1:3d}:  N/A\n")
    else:
        f.write("No track width data available\n")
    
    f.write("\n" + "=" * 70 + "\n")
    f.write(f"Cross-section location: X = {x_section:.3f} mm\n")
    f.write(f"Data directory: {data_dir}\n")
    f.write("=" * 70 + "\n")

print(f"\nSaved statistics to: {output_stats_file}")

print("\n=== All done ===")
