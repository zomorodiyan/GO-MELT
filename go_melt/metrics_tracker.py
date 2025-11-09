"""
Performance Metrics Tracker for GO-MELT Simulations

This module tracks and stores comprehensive performance metrics during simulation runs,
including mesh statistics, memory usage, execution time, and computational efficiency.

Metrics are calculated starting after 100 time-steps (warmup period) and saved alongside
simulation results.
"""

import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
import jax
import subprocess
import os


class MetricsTracker:
    """
    Tracks performance metrics for GO-MELT simulations.
    
    Collects:
    - Mesh/grid statistics (node counts, cell counts, domain size, cell size)
    - Memory usage (GPU memory)
    - Execution time and computational efficiency
    - Time-step information
    """
    
    def __init__(self, 
                 solver_input: dict,
                 save_path: str,
                 warmup_steps: int = 100):
        """
        Initialize the metrics tracker.
        
        Args:
            solver_input: The input configuration dictionary
            save_path: Path where metrics will be saved
            warmup_steps: Number of steps before starting metrics calculation (default: 100)
        """
        self.solver_input = solver_input
        self.save_path = Path(save_path)
        self.warmup_steps = warmup_steps
        
        # Timing metrics
        self.start_time = None
        self.warmup_complete_time = None
        self.end_time = None
        
        # Step tracking
        self.total_steps = 0
        self.steps_after_warmup = 0
        self.time_per_step_after_warmup = []
        
        # Memory tracking
        self.initial_gpu_memory = None
        self.min_gpu_memory = float('inf')
        self.max_gpu_memory = 0
        
        # Mesh statistics (computed once)
        self.mesh_stats = {}
        
        # Simulation parameters
        self.timestep_L3 = solver_input.get("nonmesh", {}).get("timestep_L3", None)
        
    def initialize(self, Levels: dict):
        """
        Initialize metrics collection at the start of simulation.
        
        Args:
            Levels: Dictionary containing all level mesh data
        """
        self.start_time = time.time()
        
        # Calculate mesh statistics
        self._calculate_mesh_stats(Levels)
        
        # Initialize GPU memory tracking
        self._update_memory_stats()
        self.initial_gpu_memory = self.max_gpu_memory
        
    def _calculate_mesh_stats(self, Levels: dict):
        """Calculate and store mesh/grid statistics."""
        # Node counts per level
        self.mesh_stats['level_1_node_count'] = int(Levels[1].get('nn', 0))
        self.mesh_stats['level_2_node_count'] = int(Levels[2].get('nn', 0))
        self.mesh_stats['level_3_node_count'] = int(Levels[3].get('nn', 0))
        
        # Heat source quadrature points (8 Gauss points per Level 3 element)
        # Each Level 3 hexahedral element uses 8 quadrature points for heat source integration
        if 'ne' in Levels[3]:
            num_elements_L3 = int(Levels[3]['ne'])
        else:
            level3_config = self.solver_input.get('Level3', {})
            elements = level3_config.get('elements', [0, 0, 0])
            num_elements_L3 = int(elements[0] * elements[1] * elements[2])
        
        self.mesh_stats['level_4_node_count_heat_source'] = num_elements_L3 * 8
        
        # Total computational node count (sum of Levels 1, 2, 3 - excludes Level 0 background mesh)
        # Level 0 is substrate state tracking, not part of thermal computation
        self.mesh_stats['total_node_count'] = (
            self.mesh_stats['level_1_node_count'] + 
            self.mesh_stats['level_2_node_count'] + 
            self.mesh_stats['level_3_node_count']
        )
        
        # Also store Level 0 for reference (background/substrate mesh)
        self.mesh_stats['level_0_node_count'] = int(Levels[0].get('nn', 0))
        
        # Domain size in mm (from Level 1 bounds)
        level1_config = self.solver_input.get('Level1', {})
        bounds = level1_config.get('bounds', {})
        
        x_bounds = bounds.get('x', [0, 0])
        y_bounds = bounds.get('y', [0, 0])
        z_bounds = bounds.get('z', [0, 0])
        
        self.mesh_stats['domain_size_mm'] = {
            'x_len_mm': float(x_bounds[1] - x_bounds[0]),
            'y_len_mm': float(y_bounds[1] - y_bounds[0]),
            'z_len_mm': float(z_bounds[1] - z_bounds[0])
        }
        
        # Cell size in micrometers (from Level 3, which has finest resolution)
        level3_config = self.solver_input.get('Level3', {})
        elements = level3_config.get('elements', [1, 1, 1])
        bounds_l3 = level3_config.get('bounds', {})
        
        x_bounds_l3 = bounds_l3.get('x', [0, 1])
        y_bounds_l3 = bounds_l3.get('y', [0, 1])
        z_bounds_l3 = bounds_l3.get('z', [0, 1])
        
        # Calculate cell edge length (assuming cubic cells as per user requirement)
        # Use x-direction as reference
        cell_size_mm = (x_bounds_l3[1] - x_bounds_l3[0]) / elements[0]
        self.mesh_stats['cell_size_um'] = float(cell_size_mm * 1000)  # Convert mm to micrometers
        
        # Store timestep
        self.mesh_stats['timestep_L3_seconds'] = float(self.timestep_L3) if self.timestep_L3 else None
        
    def _update_memory_stats(self):
        """Update GPU memory statistics using nvidia-smi."""
        try:
            # Get GPU ID from environment variable
            gpu_id = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
            if ',' in gpu_id:
                gpu_id = gpu_id.split(',')[0]  # Use first GPU if multiple
            
            # Query GPU memory using nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits', 
                 f'--id={gpu_id}'],
                capture_output=True,
                text=True,
                timeout=1
            )
            
            if result.returncode == 0:
                memory_mb = float(result.stdout.strip())
                
                # Update min/max
                self.min_gpu_memory = min(self.min_gpu_memory, memory_mb)
                self.max_gpu_memory = max(self.max_gpu_memory, memory_mb)
                return
        except Exception as e:
            pass
        
        # Fallback: Try to get GPU memory stats from JAX
        try:
            devices = jax.devices()
            if devices and devices[0].platform == 'gpu':
                # Get memory stats for first GPU
                memory_stats = devices[0].memory_stats()
                if memory_stats:
                    # Try different memory keys that JAX might report
                    bytes_in_use = memory_stats.get('bytes_in_use', 
                                    memory_stats.get('peak_bytes_in_use',
                                    memory_stats.get('bytes_limit', 0)))
                    
                    # Convert to MB
                    memory_mb = bytes_in_use / (1024 ** 2)
                    
                    # Only update if we got a meaningful value
                    if memory_mb > 0:
                        self.min_gpu_memory = min(self.min_gpu_memory, memory_mb)
                        self.max_gpu_memory = max(self.max_gpu_memory, memory_mb)
        except Exception as e:
            # If memory tracking fails, continue without it
            pass
    
    def update_step(self, step: int, step_start_time: float):
        """
        Update metrics after each time step.
        
        Args:
            step: Current time step number
            step_start_time: Time when this step started (from time.time())
        """
        self.total_steps = step
        
        # Update memory stats
        self._update_memory_stats()
        
        # After warmup period, track detailed timing
        if step > self.warmup_steps:
            if self.warmup_complete_time is None:
                self.warmup_complete_time = time.time()
                print(f"\n[Metrics] Warmup complete ({self.warmup_steps} steps). Starting performance tracking...\n")
            
            step_duration = time.time() - step_start_time
            self.time_per_step_after_warmup.append(step_duration)
            self.steps_after_warmup += 1
    
    def finalize(self):
        """Finalize metrics collection and compute summary statistics."""
        self.end_time = time.time()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get comprehensive metrics dictionary.
        
        Returns:
            Dictionary containing all collected metrics
        """
        metrics = {}
        
        # Mesh statistics
        metrics['mesh_statistics'] = self.mesh_stats.copy()
        
        # Memory usage (in MB)
        memory_usage_mb = self.max_gpu_memory - self.min_gpu_memory
        metrics['memory_usage'] = {
            'max_gpu_memory_mb': float(self.max_gpu_memory),
            'min_gpu_memory_mb': float(self.min_gpu_memory),
            'memory_usage_mb': float(memory_usage_mb),
            'note': 'Maximum GPU memory - Minimum GPU memory during run (after warmup)'
        }
        
        # Execution time
        if self.start_time and self.end_time:
            total_time = self.end_time - self.start_time
            metrics['execution_time'] = {
                'total_execution_time_seconds': float(total_time),
                'total_execution_time_hours': float(total_time / 3600),
                'total_steps': int(self.total_steps)
            }
            
            # Time after warmup
            if self.warmup_complete_time:
                time_after_warmup = self.end_time - self.warmup_complete_time
                metrics['execution_time']['time_after_warmup_seconds'] = float(time_after_warmup)
                metrics['execution_time']['steps_after_warmup'] = int(self.steps_after_warmup)
            
        # Computational efficiency (calculated after warmup)
        if self.time_per_step_after_warmup and len(self.time_per_step_after_warmup) > 0:
            avg_time_per_step = sum(self.time_per_step_after_warmup) / len(self.time_per_step_after_warmup)
            
            metrics['computational_efficiency'] = {
                'avg_computation_time_per_timestep_seconds': float(avg_time_per_step),
                'avg_computation_time_per_timestep_ms': float(avg_time_per_step * 1000),
                'warmup_steps_excluded': int(self.warmup_steps)
            }
            
            # Computation time per simulated time unit
            if self.timestep_L3:
                # Each timestep represents timestep_L3 seconds of simulation
                sim_time_per_step = self.timestep_L3
                
                # Compute time per simulated millisecond and second
                comp_time_per_sim_ms = avg_time_per_step / (sim_time_per_step * 1000)
                comp_time_per_sim_s = avg_time_per_step / sim_time_per_step
                
                # Choose appropriate unit (prefer ms if computation time is < 1 second per sim-second)
                if comp_time_per_sim_s < 1.0:
                    metrics['computational_efficiency']['computation_time_per_sim_ms_seconds'] = float(comp_time_per_sim_ms)
                    metrics['computational_efficiency']['preferred_unit'] = 'seconds per simulated millisecond'
                else:
                    metrics['computational_efficiency']['computation_time_per_sim_s_seconds'] = float(comp_time_per_sim_s)
                    metrics['computational_efficiency']['preferred_unit'] = 'seconds per simulated second'
                
                # Include both for completeness
                metrics['computational_efficiency']['computation_time_per_sim_ms_seconds'] = float(comp_time_per_sim_ms)
                metrics['computational_efficiency']['computation_time_per_sim_s_seconds'] = float(comp_time_per_sim_s)
                
                # Speedup factor (how much faster/slower than real-time)
                speedup_factor = sim_time_per_step / avg_time_per_step
                metrics['computational_efficiency']['speedup_factor'] = float(speedup_factor)
                metrics['computational_efficiency']['speedup_note'] = (
                    'Positive values > 1 mean faster than real-time, < 1 means slower than real-time'
                )
        
        # Add metadata
        metrics['metadata'] = {
            'warmup_steps': int(self.warmup_steps),
            'note': 'All computational efficiency metrics calculated after warmup period',
            'generated_at': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        return metrics
    
    def save_metrics(self, filename: str = 'performance_metrics.json'):
        """
        Save metrics to a JSON file.
        
        Args:
            filename: Name of the file to save (default: 'performance_metrics.json')
        """
        metrics = self.get_metrics()
        
        # Ensure save directory exists
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Save to JSON file
        output_path = self.save_path / filename
        with open(output_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\n[Metrics] Performance metrics saved to: {output_path}")
        
        # Print summary
        self._print_summary(metrics)
    
    def _print_summary(self, metrics: Dict[str, Any]):
        """Print a summary of key metrics."""
        print("\n" + "="*60)
        print("PERFORMANCE METRICS SUMMARY")
        print("="*60)
        
        # Mesh info
        mesh = metrics.get('mesh_statistics', {})
        print(f"\nMesh Statistics:")
        print(f"  Total nodes: {mesh.get('total_node_count', 'N/A'):,}")
        print(f"  Level 1 nodes: {mesh.get('level_1_node_count', 'N/A'):,}")
        print(f"  Level 2 nodes: {mesh.get('level_2_node_count', 'N/A'):,}")
        print(f"  Level 3 nodes: {mesh.get('level_3_node_count', 'N/A'):,}")
        print(f"  Level 4 nodes (heat source): {mesh.get('level_4_node_count_heat_source', 'N/A'):,}")
        
        domain = mesh.get('domain_size_mm', {})
        print(f"\nDomain Size: {domain.get('x_len_mm', 0):.2f} x {domain.get('y_len_mm', 0):.2f} x {domain.get('z_len_mm', 0):.2f} mm")
        print(f"Cell Size: {mesh.get('cell_size_um', 0):.2f} μm")
        
        # Memory
        memory = metrics.get('memory_usage', {})
        print(f"\nMemory Usage:")
        print(f"  GPU memory used: {memory.get('memory_usage_mb', 0):.2f} MB")
        
        # Execution time
        exec_time = metrics.get('execution_time', {})
        print(f"\nExecution Time:")
        print(f"  Total time: {exec_time.get('total_execution_time_hours', 0):.4f} hours")
        print(f"  Total steps: {exec_time.get('total_steps', 0):,}")
        
        # Efficiency
        efficiency = metrics.get('computational_efficiency', {})
        if efficiency:
            print(f"\nComputational Efficiency (after {efficiency.get('warmup_steps_excluded', 0)} warmup steps):")
            print(f"  Avg time per step: {efficiency.get('avg_computation_time_per_timestep_ms', 0):.2f} ms")
            
            if 'computation_time_per_sim_s_seconds' in efficiency:
                comp_per_sim_s = efficiency['computation_time_per_sim_s_seconds']
                if comp_per_sim_s < 1.0:
                    print(f"  Computation time: {comp_per_sim_s*1000:.2f} ms per simulated second")
                else:
                    print(f"  Computation time: {comp_per_sim_s:.2f} s per simulated second")
            
            if 'speedup_factor' in efficiency:
                speedup = efficiency['speedup_factor']
                if speedup > 1:
                    print(f"  Speed: {speedup:.2f}x faster than real-time")
                else:
                    print(f"  Speed: {1/speedup:.2f}x slower than real-time")
        
        print("\n" + "="*60 + "\n")
