import vtk
from vtk.util.numpy_support import vtk_to_numpy
import numpy as np
import glob
import os
import sys

# Use non-interactive backend for headless servers
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------
# User parameters
# ----------------------------------------------------------
data_dir = "results/2squareFast"   # <-- change this
melt_T = 1570.0                  # K
T_solidus = 1533.0               # K
T_solidus_low = 1423.0           # K (T_solidus - 110)
dt = 0.25e-5                     # time step [s]
tol_z = 1e-12                    # tolerance for z = 0 plane

# Set to None to process ALL files, or to an int to limit for testing
max_files = None

output_npy_TAM    = "max_time_above_melt_2D.npy"
output_npy_SCR    = "solidus_cooling_rate_2D.npy"
output_npy_params = "surface_grid_params.npy"
output_txt_stats  = "TAM_SCR_statistics.txt"
output_png_TAM    = "TAM_surface.png"
output_png_SCR    = "SCR_surface.png"
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
# PASS 1: determine global grid + dx, dy + k_surface
# ----------------------------------------------------------
print("\n=== PASS 1: scanning coordinates to build global grid ===")

# Read first file to get dx, dy, and k_surface
first_grid = load_vtr(files[0])
Nx_local, Ny_local, Nz_local = first_grid.GetDimensions()

x_coords_vtk = first_grid.GetXCoordinates()
y_coords_vtk = first_grid.GetYCoordinates()
z_coords_vtk = first_grid.GetZCoordinates()

x_coords0 = np.array([x_coords_vtk.GetValue(i) for i in range(x_coords_vtk.GetNumberOfTuples())])
y_coords0 = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])
z_coords0 = np.array([z_coords_vtk.GetValue(k) for k in range(z_coords_vtk.GetNumberOfTuples())])

if len(x_coords0) < 2 or len(y_coords0) < 2:
    raise RuntimeError("Not enough points to infer dx/dy from first file.")

dx = x_coords0[1] - x_coords0[0]
dy = y_coords0[1] - y_coords0[0]
print(f"Inferred spacing: dx = {dx}, dy = {dy}")

# Find k_surface where z == 0
k_surface = None
for k, zv in enumerate(z_coords0):
    if abs(zv - 0.0) < tol_z:
        k_surface = k
        break
if k_surface is None:
    raise RuntimeError("Could not find z = 0 in first file's ZCoordinates.")
print(f"Surface z=0 is at k_surface = {k_surface}")

# Global extents
min_x = np.inf
max_x = -np.inf
min_y = np.inf
max_y = -np.inf

total_files = len(files)
for idx, f in enumerate(files):
    if idx % max(1, total_files // 20) == 0:
        print(f"  [PASS1] {idx}/{total_files} files scanned...")
        sys.stdout.flush()

    grid = load_vtr(f)
    x_coords_vtk = grid.GetXCoordinates()
    y_coords_vtk = grid.GetYCoordinates()

    x_coords = np.array([x_coords_vtk.GetValue(i) for i in range(x_coords_vtk.GetNumberOfTuples())])
    y_coords = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])

    min_x = min(min_x, x_coords[0])
    max_x = max(max_x, x_coords[-1])
    min_y = min(min_y, y_coords[0])
    max_y = max(max_y, y_coords[-1])

# Snap minima to grid
min_x = round(min_x / dx) * dx
min_y = round(min_y / dy) * dy

Nx_global = int(round((max_x - min_x) / dx)) + 1
Ny_global = int(round((max_y - min_y) / dy)) + 1
T_global  = len(files)

print("\nGlobal surface grid:")
print(f"  Nx = {Nx_global}, Ny = {Ny_global}, T = {T_global}")
print(f"  X range: {min_x} → {max_x}")
print(f"  Y range: {min_y} → {max_y}\n")

# ----------------------------------------------------------
# PASS 2: streaming TAM and SCR computation
# ----------------------------------------------------------
print("=== PASS 2: streaming through files to compute TAM and SCR ===")

# Arrays for TAM computation
best_run    = np.zeros((Ny_global, Nx_global), dtype=np.int32)
current_run = np.zeros((Ny_global, Nx_global), dtype=np.int32)
hot         = np.zeros((Ny_global, Nx_global), dtype=bool)

# Arrays for SCR computation
tam_end_time    = np.full((Ny_global, Nx_global), -1, dtype=np.int32)  # When best TAM ended
in_cooling      = np.zeros((Ny_global, Nx_global), dtype=bool)          # Currently in cooling phase after best TAM
reached_1533    = np.zeros((Ny_global, Nx_global), dtype=bool)          # Reached T_solidus after TAM
time_at_1533    = np.full((Ny_global, Nx_global), -1, dtype=np.int32)  # Timestep when reached 1533K
reached_1423    = np.zeros((Ny_global, Nx_global), dtype=bool)          # Reached T_solidus_low after 1533K
time_at_1423    = np.full((Ny_global, Nx_global), -1, dtype=np.int32)  # Timestep when reached 1423K
scr_computed    = np.zeros((Ny_global, Nx_global), dtype=bool)          # SCR has been computed for this node

for t_idx, f in enumerate(files):
    print(f"  [PASS2] timestep {t_idx+1}/{T_global}: {os.path.basename(f)}")
    sys.stdout.flush()

    grid = load_vtr(f)
    Nx_loc, Ny_loc, Nz_loc = grid.GetDimensions()

    # Coordinates for this window
    x_coords_vtk = grid.GetXCoordinates()
    y_coords_vtk = grid.GetYCoordinates()

    x_coords = np.array([x_coords_vtk.GetValue(i) for i in range(x_coords_vtk.GetNumberOfTuples())])
    y_coords = np.array([y_coords_vtk.GetValue(j) for j in range(y_coords_vtk.GetNumberOfTuples())])

    # Temperature
    temps_vtk = grid.GetPointData().GetArray("Temperature (K)")
    if temps_vtk is None:
        raise RuntimeError(f"'Temperature (K)' array not found in {f}")
    temps_np = vtk_to_numpy(temps_vtk)

    # reshape to (Nz, Ny, Nx) with x fastest
    temps_3d = temps_np.reshape((Nz_loc, Ny_loc, Nx_loc), order="C")

    # surface slice at z=0
    Tsurf = temps_3d[k_surface, :, :]  # (Ny_loc, Nx_loc)

    # Compute global indices where this window maps
    x0 = x_coords[0]
    y0 = y_coords[0]

    ix_start = int(round((x0 - min_x) / dx))
    iy_start = int(round((y0 - min_y) / dy))

    ix_end = ix_start + Nx_loc
    iy_end = iy_start + Ny_loc

    if ix_start < 0 or iy_start < 0 or ix_end > Nx_global or iy_end > Ny_global:
        raise RuntimeError(f"Window out of global bounds for timestep {t_idx}")

    # Clear hot array and set only the current window region
    hot.fill(False)
    hot_local = (Tsurf > melt_T)
    hot[iy_start:iy_end, ix_start:ix_end] = hot_local

    # Update run lengths
    # where hot: increment, else reset
    current_run[hot] += 1
    
    # Detect where TAM just ended (was hot, now cooling)
    just_cooled = (~hot) & (current_run > 0)
    
    # Update best_run and track when best TAM ends
    old_best = best_run.copy()
    np.maximum(best_run, current_run, out=best_run)
    
    # If we just set a new record, mark this as the TAM end time
    new_record = (best_run > old_best) & just_cooled
    tam_end_time[new_record] = t_idx
    in_cooling[new_record] = True
    # Reset SCR tracking for nodes with new records
    reached_1533[new_record] = False
    reached_1423[new_record] = False
    time_at_1533[new_record] = -1
    time_at_1423[new_record] = -1
    scr_computed[new_record] = False
    
    # For nodes that just cooled and matched their best run (even if not a new record)
    matched_best = (current_run == best_run) & just_cooled
    tam_end_time[matched_best] = t_idx
    in_cooling[matched_best] = True
    # Reset SCR tracking
    reached_1533[matched_best] = False
    reached_1423[matched_best] = False
    time_at_1533[matched_best] = -1
    time_at_1423[matched_best] = -1
    scr_computed[matched_best] = False
    
    # EDGE CASE: If a node re-enters melt zone, reset cooling phase and SCR tracking
    re_melted = hot & in_cooling
    in_cooling[re_melted] = False
    reached_1533[re_melted] = False
    reached_1423[re_melted] = False
    time_at_1533[re_melted] = -1
    time_at_1423[re_melted] = -1
    # Note: scr_computed stays as-is; we'll only update when we complete a valid cooling cycle
    
    # Reset current_run for nodes that are not hot
    current_run[~hot] = 0
    
    # --- SCR tracking for nodes in cooling phase ---
    # Extract temperature for the current window
    Tsurf_window = Tsurf  # (Ny_loc, Nx_loc)
    
    # Map to global coordinates
    # We need to work only within the window that has temperature data
    for j_loc in range(Ny_loc):
        for i_loc in range(Nx_loc):
            i_global = ix_start + i_loc
            j_global = iy_start + j_loc
            
            T_current = Tsurf_window[j_loc, i_loc]
            
            # Only track SCR if we're in cooling phase and haven't computed SCR yet
            if in_cooling[j_global, i_global] and not scr_computed[j_global, i_global]:
                # Check if we've reached 1533K
                if not reached_1533[j_global, i_global] and T_current <= T_solidus:
                    reached_1533[j_global, i_global] = True
                    time_at_1533[j_global, i_global] = t_idx
                
                # Check if we've reached 1423K (only after reaching 1533K)
                elif reached_1533[j_global, i_global] and not reached_1423[j_global, i_global] and T_current <= T_solidus_low:
                    reached_1423[j_global, i_global] = True
                    time_at_1423[j_global, i_global] = t_idx
                    scr_computed[j_global, i_global] = True  # Mark as done

# Convert runs to time (seconds)
max_time = best_run.astype(np.float32) * dt

# ----------------------------------------------------------
# Compute SCR values
# ----------------------------------------------------------
print("\n=== Computing final SCR values ===")

# Initialize SCR array with placeholder value
scr_array = np.full((Ny_global, Nx_global), 0.0, dtype=np.float32)

# Case 1: SCR computed successfully (reached both 1533K and 1423K)
valid_scr = reached_1533 & reached_1423
time_diff = (time_at_1423[valid_scr] - time_at_1533[valid_scr]) * dt
scr_array[valid_scr] = 110.0 / time_diff  # K/s

# Find minimum observed SCR for error value calculation
if np.any(valid_scr):
    min_observed_scr = np.min(scr_array[valid_scr])
    print(f"  Minimum observed SCR: {min_observed_scr:.2f} K/s")
else:
    min_observed_scr = 1.0  # Fallback if no valid SCR computed
    print(f"  No valid SCR computed, using fallback value")

# Case 2: Had a TAM but never reached 1533K (shouldn't happen)
had_tam = (best_run > 0)
case_2 = had_tam & (~reached_1533)
scr_array[case_2] = min_observed_scr / 1.2

# Case 3: Reached 1533K but never reached 1423K (shouldn't happen)
case_1 = reached_1533 & (~reached_1423)
scr_array[case_1] = min_observed_scr / 1.5

# Case 4: Never had a TAM (never got hot enough) -> -1e7
scr_array[~had_tam] = -1e7

num_valid = np.sum(valid_scr)
num_case1 = np.sum(case_1)
num_case2 = np.sum(case_2)
num_no_tam = np.sum(~had_tam)

print(f"  Valid SCR computed: {num_valid} nodes")
print(f"  Reached 1533K but not 1423K: {num_case1} nodes (SCR = {min_observed_scr/1.5:.2f})")
print(f"  Had TAM but never reached 1533K: {num_case2} nodes (SCR = {min_observed_scr/1.2:.2f})")
print(f"  Never experienced TAM: {num_no_tam} nodes (SCR = -1e7)")

# ----------------------------------------------------------
# Compute statistics for TAM and SCR
# ----------------------------------------------------------
print("\n=== Computing TAM and SCR Statistics ===")

# TAM statistics (only values > 0)
tam_positive = max_time[max_time > 0]
if len(tam_positive) > 0:
    tam_mean = np.mean(tam_positive)
    tam_median = np.median(tam_positive)
    tam_std = np.std(tam_positive)
    print(f"\nTAM Statistics (values > 0):")
    print(f"  Mean:   {tam_mean:.6e} s")
    print(f"  Median: {tam_median:.6e} s")
    print(f"  Std:    {tam_std:.6e} s")
else:
    tam_mean = tam_median = tam_std = 0.0
    print("\nNo positive TAM values found")

# SCR statistics (only values > 0)
scr_positive = scr_array[scr_array > 0]
if len(scr_positive) > 0:
    scr_mean = np.mean(scr_positive)
    scr_median = np.median(scr_positive)
    scr_std = np.std(scr_positive)
    print(f"\nSCR Statistics (values > 0):")
    print(f"  Mean:   {scr_mean:.6e} K/s")
    print(f"  Median: {scr_median:.6e} K/s")
    print(f"  Std:    {scr_std:.6e} K/s")
else:
    scr_mean = scr_median = scr_std = 0.0
    print("\nNo positive SCR values found")

# Save statistics to text file
with open(output_txt_stats, 'w') as f:
    f.write("TAM and SCR Statistics\n")
    f.write("=" * 60 + "\n\n")
    
    f.write("TAM (Time Above Melt) Statistics (values > 0):\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Mean:         {tam_mean:.6e} s\n")
    f.write(f"  Median:       {tam_median:.6e} s\n")
    f.write(f"  Std Dev:      {tam_std:.6e} s\n")
    f.write(f"  Node Count:   {len(tam_positive)}\n")
    f.write(f"  Total Nodes:  {Nx_global * Ny_global}\n\n")
    
    f.write("SCR (Solidus Cooling Rate) Statistics (values > 0):\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Mean:         {scr_mean:.6e} K/s\n")
    f.write(f"  Median:       {scr_median:.6e} K/s\n")
    f.write(f"  Std Dev:      {scr_std:.6e} K/s\n")
    f.write(f"  Node Count:   {len(scr_positive)}\n")
    f.write(f"  Total Nodes:  {Nx_global * Ny_global}\n\n")
    f.write("SCR Node Counts by Case:\n")
    f.write("-" * 60 + "\n")
    f.write(f"  Valid SCR (1533K→1423K): {num_valid}\n")
    f.write(f"  Had TAM, never reached 1533K (min/1.2): {num_case2}\n")
    f.write(f"  Reached 1533K, never reached 1423K (min/1.5): {num_case1}\n")
    f.write(f"  Never experienced TAM (-1e7): {num_no_tam}\n\n")
    f.write("=" * 60 + "\n")
    f.write(f"Data Directory: {data_dir}\n")
    f.write(f"Melt Temperature: {melt_T} K\n")
    f.write(f"Solidus Temperature: {T_solidus} K\n")
    f.write(f"Solidus Low Temperature: {T_solidus_low} K\n")

# ----------------------------------------------------------
# Save arrays
# ----------------------------------------------------------
np.save(output_npy_TAM, max_time.astype(np.float32))
np.save(output_npy_SCR, scr_array)
np.save(
    output_npy_params,
    np.array([min_x, min_y, dx, dy, Nx_global, Ny_global, T_global], dtype=np.float64)
)

print("\nSaved:")
print(f"  {output_npy_TAM}")
print(f"  {output_npy_SCR}")
print(f"  {output_npy_params}")
print(f"  {output_txt_stats}")

# ----------------------------------------------------------
# Save PNGs
# ----------------------------------------------------------
print("\n=== Generating PNG plots ===")

x_coords_global = min_x + dx * np.arange(Nx_global)
y_coords_global = min_y + dy * np.arange(Ny_global)

# TAM plot
plt.figure(figsize=(12, 5))
plt.imshow(
    max_time,
    origin="lower",
    extent=[x_coords_global[0], x_coords_global[-1],
            y_coords_global[0], y_coords_global[-1]],
    aspect='equal'
)
plt.colorbar(label="Max time above melt (s)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Max-time-above-melt (TAM)")
plt.savefig(output_png_TAM, dpi=300, bbox_inches="tight")
print(f"Saved PNG: {output_png_TAM}")
plt.close()

# SCR plot (full field including negative values)
plt.figure(figsize=(12, 5))
plt.imshow(
    scr_array,
    origin="lower",
    extent=[x_coords_global[0], x_coords_global[-1],
            y_coords_global[0], y_coords_global[-1]],
    aspect='equal',
    cmap='viridis'
)
plt.colorbar(label="Solidus Cooling Rate (K/s)")
plt.xlabel("X coordinate")
plt.ylabel("Y coordinate")
plt.title("Solidus Cooling Rate (1533K → 1423K)\n(-1e7: No TAM, min/1.2: No 1533K, min/1.5: No 1423K)")
plt.savefig(output_png_SCR, dpi=300, bbox_inches="tight")
print(f"Saved PNG: {output_png_SCR}")
plt.close()

print("\n=== All done ===")

