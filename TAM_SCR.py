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
output_txt_stats  = "statistics_TAM_SCR.txt"
output_png_TAM    = "surface_TAM.png"
output_png_SCR    = "surface_SCR.png"
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

# ----------------------------------------------------------
# Read toolpath file and build cumulative time array
# ----------------------------------------------------------
toolpath_file = os.path.join(data_dir, "toolpath.txt")
if not os.path.exists(toolpath_file):
    raise RuntimeError(f"Toolpath file not found: {toolpath_file}")

dt_list = []
with open(toolpath_file, "r") as f:
    for line_num, line in enumerate(f, 1):
        parts = line.strip().split(",")
        if len(parts) >= 7:
            try:
                # dt is in column 5 (0-indexed), which is the 6th column
                dt_val = float(parts[5])
                dt_list.append(dt_val)
            except ValueError as e:
                print(f"Warning line {line_num}: Could not parse dt from '{parts[5]}': {e}")
                dt_list.append(dt)
        else:
            print(f"Warning line {line_num}: Not enough columns ({len(parts)})")
            dt_list.append(dt)  # fallback to default if missing

dt_array = np.array(dt_list)
n_tool_steps = len(dt_array)
n_vtr_files_total = len(all_files)  # Use total VTR count, not limited subset
n_vtr_files_processed = len(files)

if n_tool_steps == n_vtr_files_total:
    # 1:1 mapping
    dt_array_per_frame = dt_array.copy()
    print(f"Toolpath steps matches total VTR frames: {n_tool_steps}")
elif n_tool_steps % n_vtr_files_total == 0:
    # exact integer grouping (common case: record_step > 1)
    group = n_tool_steps // n_vtr_files_total
    dt_array_per_frame = dt_array.reshape((n_vtr_files_total, group)).sum(axis=1)
    print(f"Aggregated toolpath dt into {n_vtr_files_total} frames by summing groups of {group} steps (toolpath rows per VTR).")
else:
    # try best-effort grouping (rounding)
    approx_group = int(round(n_tool_steps / n_vtr_files_total))
    if approx_group <= 0:
        raise RuntimeError("Invalid grouping computed for toolpath/vtr mapping")
    dt_array_per_frame = np.empty(n_vtr_files_total, dtype=dt_array.dtype)
    for i in range(n_vtr_files_total):
        start = i * approx_group
        end = start + approx_group
        if i == n_vtr_files_total - 1:
            end = n_tool_steps
        dt_array_per_frame[i] = dt_array[start:end].sum()
    print(f"Warning: toolpath steps ({n_tool_steps}) not divisible by total VTR files ({n_vtr_files_total}).")
    print(f"Using approx group={approx_group}; last frame may have different count.")

# If processing only a subset, slice the dt array accordingly
if n_vtr_files_processed < n_vtr_files_total:
    dt_array = dt_array_per_frame[:n_vtr_files_processed]
    print(f"Processing subset: using first {n_vtr_files_processed} of {n_vtr_files_total} aggregated dt values.")
else:
    dt_array = dt_array_per_frame
cum_time = np.cumsum(dt_array)
print(f"Loaded {n_tool_steps} toolpath steps, aggregated to {len(dt_array)} VTR frames. Total simulated time: {cum_time[-1]:.6e} s")

# Arrays for interpolation-based TAM and SCR computation
prev_temp       = np.full((Ny_global, Nx_global), -1.0, dtype=np.float32)  # Previous temperature
in_melt_zone    = np.zeros((Ny_global, Nx_global), dtype=bool)              # Currently above melt_T
melt_entry_time = np.full((Ny_global, Nx_global), -1.0, dtype=np.float32)   # Time when entered current melt zone
current_tam     = np.zeros((Ny_global, Nx_global), dtype=np.float32)        # TAM for current melt episode
best_tam        = np.zeros((Ny_global, Nx_global), dtype=np.float32)        # Best (longest) TAM so far

# Arrays for SCR computation
in_cooling      = np.zeros((Ny_global, Nx_global), dtype=bool)               # Currently in cooling phase after best TAM
time_at_1533    = np.full((Ny_global, Nx_global), -1.0, dtype=np.float32)   # Interpolated time when crossed 1533K
time_at_1423    = np.full((Ny_global, Nx_global), -1.0, dtype=np.float32)   # Interpolated time when crossed 1423K
reached_1533    = np.zeros((Ny_global, Nx_global), dtype=bool)               # Crossed 1533K during cooling
reached_1423    = np.zeros((Ny_global, Nx_global), dtype=bool)               # Crossed 1423K during cooling
scr_computed    = np.zeros((Ny_global, Nx_global), dtype=bool)               # SCR has been computed

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

    # Current time at this frame
    current_time = cum_time[t_idx]
    dt_current = dt_array[t_idx]
    prev_time = current_time - dt_current
    
    # Process each node in the window
    for j_loc in range(Ny_loc):
        for i_loc in range(Nx_loc):
            i_global = ix_start + i_loc
            j_global = iy_start + j_loc
            
            T_curr = Tsurf[j_loc, i_loc]
            T_prev = prev_temp[j_global, i_global]
            
            # Skip first frame (no previous temperature)
            if T_prev < 0:
                prev_temp[j_global, i_global] = T_curr
                continue
            
            # === TAM TRACKING WITH INTERPOLATION ===
            was_in_melt = in_melt_zone[j_global, i_global]
            is_in_melt = (T_curr > melt_T)
            
            if not was_in_melt and is_in_melt:
                # Entered melt zone: interpolate entry time
                if T_prev < melt_T:
                    # Crossed threshold during this time step
                    fraction = (melt_T - T_prev) / (T_curr - T_prev) if T_curr != T_prev else 0.0
                    entry_time = prev_time + fraction * dt_current
                else:
                    # Already above threshold at start
                    entry_time = prev_time
                melt_entry_time[j_global, i_global] = entry_time
                in_melt_zone[j_global, i_global] = True
                in_cooling[j_global, i_global] = False  # Reset cooling if re-entering
                
            elif was_in_melt and not is_in_melt:
                # Exited melt zone: interpolate exit time and accumulate TAM
                if T_curr < melt_T:
                    # Crossed threshold during this time step
                    fraction = (melt_T - T_prev) / (T_curr - T_prev) if T_curr != T_prev else 1.0
                    exit_time = prev_time + fraction * dt_current
                else:
                    # Still above at end (shouldn't happen given is_in_melt check)
                    exit_time = current_time
                
                entry_time = melt_entry_time[j_global, i_global]
                if entry_time >= 0:
                    tam_duration = exit_time - entry_time
                    current_tam[j_global, i_global] = tam_duration
                    
                    # Update best TAM if this is the longest
                    if tam_duration > best_tam[j_global, i_global]:
                        best_tam[j_global, i_global] = tam_duration
                        # Start cooling phase for SCR tracking
                        in_cooling[j_global, i_global] = True
                        reached_1533[j_global, i_global] = False
                        reached_1423[j_global, i_global] = False
                        time_at_1533[j_global, i_global] = -1.0
                        time_at_1423[j_global, i_global] = -1.0
                        scr_computed[j_global, i_global] = False
                
                in_melt_zone[j_global, i_global] = False
                melt_entry_time[j_global, i_global] = -1.0
            
            elif was_in_melt and is_in_melt:
                # Still in melt zone: no action needed
                pass
            
            # === SCR TRACKING WITH INTERPOLATION (during cooling) ===
            if in_cooling[j_global, i_global] and not scr_computed[j_global, i_global]:
                # Check for 1533K crossing
                if not reached_1533[j_global, i_global]:
                    if T_prev > T_solidus and T_curr <= T_solidus:
                        # Crossed 1533K: interpolate crossing time
                        fraction = (T_solidus - T_prev) / (T_curr - T_prev) if T_curr != T_prev else 1.0
                        cross_time = prev_time + fraction * dt_current
                        time_at_1533[j_global, i_global] = cross_time
                        reached_1533[j_global, i_global] = True
                    elif T_curr <= T_solidus:
                        # Already below at start of this frame
                        time_at_1533[j_global, i_global] = prev_time
                        reached_1533[j_global, i_global] = True
                
                # Check for 1423K crossing (only after 1533K)
                if reached_1533[j_global, i_global] and not reached_1423[j_global, i_global]:
                    if T_prev > T_solidus_low and T_curr <= T_solidus_low:
                        # Crossed 1423K: interpolate crossing time
                        fraction = (T_solidus_low - T_prev) / (T_curr - T_prev) if T_curr != T_prev else 1.0
                        cross_time = prev_time + fraction * dt_current
                        time_at_1423[j_global, i_global] = cross_time
                        reached_1423[j_global, i_global] = True
                        scr_computed[j_global, i_global] = True
                    elif T_curr <= T_solidus_low:
                        # Already below at start of this frame
                        time_at_1423[j_global, i_global] = prev_time
                        reached_1423[j_global, i_global] = True
                        scr_computed[j_global, i_global] = True
            
            # Update previous temperature
            prev_temp[j_global, i_global] = T_curr

# TAM is already computed during the loop (stored in best_tam)
max_time = best_tam

# Find minimum observed TAM for error value calculation (positive values only)
tam_positive_values = max_time[max_time > 0]
if len(tam_positive_values) > 0:
    min_observed_tam = np.min(tam_positive_values)
    print(f"\n  Minimum observed TAM (positive): {min_observed_tam:.6e} s")
else:
    min_observed_tam = 1e-6  # Fallback if no positive TAM computed
    print(f"\n  No positive TAM computed, using fallback value")

# ----------------------------------------------------------
# Compute SCR values
# ----------------------------------------------------------
print("\n=== Computing final SCR values ===")

scr_array = np.full((Ny_global, Nx_global), 0.0, dtype=np.float32)

# Case 1: SCR computed successfully (reached both 1533K and 1423K)
valid_scr = reached_1533 & reached_1423
scr_time_diff = np.zeros_like(scr_array)
for j in range(Ny_global):
    for i in range(Nx_global):
        if valid_scr[j, i]:
            t1 = time_at_1533[j, i]
            t2 = time_at_1423[j, i]
            if t1 >= 0 and t2 > t1:
                # Use interpolated times directly (already in seconds)
                scr_time_diff[j, i] = t2 - t1
                scr_array[j, i] = 110.0 / scr_time_diff[j, i] if scr_time_diff[j, i] > 0 else 0.0

# Find minimum observed SCR for error value calculation
if np.any(valid_scr):
    min_observed_scr = np.min(scr_array[valid_scr])
    print(f"  Minimum observed SCR: {min_observed_scr:.2f} K/s")
else:
    min_observed_scr = 1.0  # Fallback if no valid SCR computed
    print(f"  No valid SCR computed, using fallback value")

# Case 2: Had a positive TAM but never reached 1533K -> use min_tam / 1.1
had_tam = (max_time > 0)
case_2 = had_tam & (~reached_1533)
scr_array[case_2] = min_observed_tam / 1.1

# Case 3: Reached 1533K but never reached 1423K -> use min_tam / 1.2
case_1 = reached_1533 & (~reached_1423)
scr_array[case_1] = min_observed_tam / 1.2

# Case 4: Never had a TAM (never got hot enough) -> -1e6
scr_array[~had_tam] = -1e6

num_valid = np.sum(valid_scr)
num_case1 = np.sum(case_1)
num_case2 = np.sum(case_2)
num_no_tam = np.sum(~had_tam)

print(f"  Valid SCR computed: {num_valid} nodes")
print(f"  Reached 1533K but not 1423K: {num_case1} nodes (SCR = {min_observed_tam / 1.5:.6e})")
print(f"  Had TAM but never reached 1533K: {num_case2} nodes (SCR = {min_observed_tam / 1.2:.6e})")
print(f"  Never experienced TAM: {num_no_tam} nodes (SCR = -1e6)")

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
    f.write(f"  Had TAM, never reached 1533K (min_TAM/1.1): {num_case2}\n")
    f.write(f"  Reached 1533K, never reached 1423K (min_TAM/1.2): {num_case1}\n")
    f.write(f"  Never experienced TAM (-1e6): {num_no_tam}\n\n")
    f.write(f"  Minimum observed TAM used for error values: {min_observed_tam:.6e} s\n\n")
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
plt.title("Solidus Cooling Rate (1533K → 1423K)\n(-1e6: No TAM, min_TAM/1.1: No 1533K, min_TAM/1.2: No 1423K)")
plt.savefig(output_png_SCR, dpi=300, bbox_inches="tight")
print(f"Saved PNG: {output_png_SCR}")
plt.close()

print("\n=== All done ===")
