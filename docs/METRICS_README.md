# Performance Metrics Tracking for GO-MELT

## Overview

The performance metrics tracker automatically collects and saves comprehensive performance data whenever you run a GO-MELT simulation. This allows you to analyze computational efficiency, memory usage, and mesh statistics across different simulations.

## What Gets Tracked

### Mesh Statistics
- **Node counts per level**: Level 1, 2, 3, and 4 (heat source) node counts
- **Total node count**: From the base substrate mesh (Level 0)
- **Level 3 cell count**: Number of computational cells in the finest mesh
- **Domain size**: Physical dimensions in mm (x_len_mm, y_len_mm, z_len_mm)
- **Cell size**: Edge length of cubic cells in micrometers (μm)
- **Time-step**: The Level 3 timestep used in the simulation (seconds)

### Memory Usage
- **Maximum GPU memory**: Peak GPU memory during simulation
- **Minimum GPU memory**: Baseline GPU memory during simulation
- **Memory usage**: Difference between max and min (calculated after 100-step warmup)

### Execution Time
- **Total execution time**: Complete simulation duration (seconds and hours)
- **Total steps**: Number of time-steps executed
- **Time after warmup**: Execution time excluding the first 100 steps
- **Steps after warmup**: Number of steps used for performance metrics

### Computational Efficiency (calculated after 100-step warmup)
- **Average time per time-step**: Mean computation time per step (seconds and milliseconds)
- **Computation time per simulated time**: How long it takes to compute 1 ms or 1 s of simulation
- **Speedup factor**: Ratio of simulated time to computation time
  - Values > 1 mean faster than real-time
  - Values < 1 mean slower than real-time

## How It Works

### Minimal Changes to Existing Code

Only **4 lines** were added to `go_melt.py`:

1. **Import the tracker** (line 11):
   ```python
   from metrics_tracker import MetricsTracker
   ```

2. **Create tracker instance** (in `go_melt()` function):
   ```python
   metrics_tracker = MetricsTracker(
       solver_input=solver_input,
       save_path=solver_input.get("nonmesh", {}).get("save_path", "./results/"),
       warmup_steps=100
   )
   ```

3. **Initialize with mesh data** (after mesh setup):
   ```python
   metrics_tracker.initialize(Levels)
   ```

4. **Update each step** (in main simulation loop):
   ```python
   metrics_tracker.update_step(time_inc, t_loop)
   ```

5. **Finalize and save** (at end of simulation):
   ```python
   metrics_tracker.finalize()
   metrics_tracker.save_metrics()
   ```

### Warmup Period

The first **100 time-steps** are excluded from computational efficiency calculations to avoid:
- Initial JIT compilation overhead
- Memory allocation overhead
- Cache warming effects

All other metrics (mesh statistics, total execution time, etc.) include the full simulation.

## Output

### File Location

Metrics are automatically saved as `performance_metrics.json` in the same directory as your simulation results:

```
results/
  example/
    ├── performance_metrics.json  ← NEW
    ├── Level0_00000000.vtr
    ├── Level1_00000001.vtr
    └── ...
```

### File Format

The output is a JSON file with the following structure:

```json
{
    "mesh_statistics": {
        "level_1_node_count": 32451,
        "level_2_node_count": 112211,
        "level_3_node_count": 123321,
        "level_3_cell_count": 100000,
        "level_4_node_count_heat_source": 123321,
        "total_node_count": 520000,
        "domain_size_mm": {
            "x_len_mm": 10.0,
            "y_len_mm": 4.0,
            "z_len_mm": 6.0
        },
        "cell_size_um": 20.0,
        "timestep_L3_seconds": 1e-05
    },
    "memory_usage": {
        "max_gpu_memory_mb": 4523.45,
        "min_gpu_memory_mb": 1234.56,
        "memory_usage_mb": 3288.89,
        "note": "Maximum GPU memory - Minimum GPU memory during run (after warmup)"
    },
    "execution_time": {
        "total_execution_time_seconds": 3625.5,
        "total_execution_time_hours": 1.0071,
        "total_steps": 5000,
        "time_after_warmup_seconds": 3550.2,
        "steps_after_warmup": 4900
    },
    "computational_efficiency": {
        "avg_computation_time_per_timestep_seconds": 0.7245,
        "avg_computation_time_per_timestep_ms": 724.5,
        "warmup_steps_excluded": 100,
        "computation_time_per_sim_ms_seconds": 72450.0,
        "computation_time_per_sim_s_seconds": 72.45,
        "preferred_unit": "seconds per simulated second",
        "speedup_factor": 1.38e-05,
        "speedup_note": "Positive values > 1 mean faster than real-time, < 1 means slower than real-time"
    },
    "metadata": {
        "warmup_steps": 100,
        "note": "All computational efficiency metrics calculated after warmup period",
        "generated_at": "2025-11-07 14:23:45"
    }
}
```

### Console Output

At the end of each simulation, a summary is printed:

```
[Metrics] Performance metrics saved to: /path/to/results/example/performance_metrics.json

============================================================
PERFORMANCE METRICS SUMMARY
============================================================

Mesh Statistics:
  Total nodes: 520,000
  Level 1 nodes: 32,451
  Level 2 nodes: 112,211
  Level 3 nodes: 123,321
  Level 3 cells: 100,000

Domain Size: 10.00 x 4.00 x 6.00 mm
Cell Size: 20.00 μm

Memory Usage:
  GPU memory used: 3288.89 MB

Execution Time:
  Total time: 1.0071 hours
  Total steps: 5,000

Computational Efficiency (after 100 warmup steps):
  Avg time per step: 724.50 ms
  Computation time: 72.45 s per simulated second
  Speed: 72450.00x slower than real-time

============================================================
```

## Usage

Just run your simulations as usual:

```bash
# In WSL2 Ubuntu
cd ~/bin/GO-MELT
./go-melt-docker  # or whatever your startup script is

# Then in the container
python go_melt/go_melt.py 0 examples/example.json
```

The metrics will be automatically collected and saved alongside your results!

## Customization

### Change Warmup Period

If you want to use a different warmup period, modify the initialization in `go_melt.py`:

```python
metrics_tracker = MetricsTracker(
    solver_input=solver_input,
    save_path=solver_input.get("nonmesh", {}).get("save_path", "./results/"),
    warmup_steps=200  # Change from 100 to 200
)
```

### Change Output Filename

By default, metrics are saved as `performance_metrics.json`. To change this:

```python
metrics_tracker.save_metrics(filename='my_custom_metrics.json')
```

## Files Added/Modified

### New Files
- `go_melt/metrics_tracker.py` - Complete metrics tracking module (standalone, ~430 lines)
- `go_melt/example_performance_metrics.json` - Example output format

### Modified Files
- `go_melt/go_melt.py` - Added 5 lines to integrate metrics tracker:
  - 1 import statement
  - 1 tracker initialization
  - 1 mesh data initialization
  - 1 per-step update
  - 2 finalization/save calls

## Benefits

1. **Minimal code changes**: Only 5 additional lines in the main simulation file
2. **Automatic collection**: No manual tracking needed
3. **Comprehensive data**: All requested metrics in one place
4. **Easy comparison**: JSON format makes it easy to compare across runs
5. **Post-warmup accuracy**: Excludes JIT compilation overhead for accurate performance metrics
6. **Machine-readable**: JSON format allows easy parsing for analysis scripts

## Note on Cell Size

The tracker assumes cubic cells (as you mentioned you'll ensure). It calculates the cell size from Level 3 (finest mesh) using the x-direction as reference:

```
cell_size = (x_max - x_min) / num_elements_x
```

This is converted from mm to μm in the output.
