#!/usr/bin/env python3
"""
Process performance metrics from all simulation results and create a summary CSV.

This script scans the results directory for performance_metrics.json files,
extracts key metrics, calculates snapshot file sizes, and outputs a CSV file.

Usage:
    python process_performance_metrics.py
"""

import json
import csv
import os
from pathlib import Path
from typing import Dict, List, Any


def get_snapshot_file_size(result_dir: Path) -> float:
    """
    Calculate average file size of VTR snapshot files in MB.
    
    Args:
        result_dir: Path to the result directory
        
    Returns:
        Average file size in MB, or 0 if no files found
    """
    vtr_files = list(result_dir.glob("Level*.vtr"))
    
    if not vtr_files:
        return 0.0
    
    total_size = sum(f.stat().st_size for f in vtr_files)
    avg_size_mb = total_size / len(vtr_files) / (1024 ** 2)
    
    return avg_size_mb


def process_metrics_file(metrics_path: Path) -> Dict[str, Any]:
    """
    Extract relevant metrics from a performance_metrics.json file.
    
    Args:
        metrics_path: Path to the performance_metrics.json file
        
    Returns:
        Dictionary with extracted metrics
    """
    with open(metrics_path, 'r') as f:
        data = json.load(f)
    
    result_dir = metrics_path.parent
    case_name = result_dir.name
    
    # Extract mesh statistics
    mesh_stats = data.get('mesh_statistics', {})
    # Total nodes from Level 0 (background mesh) - does not include heat source quadrature points
    total_nodes = mesh_stats.get('total_node_count', 0)
    cell_size_um = mesh_stats.get('cell_size_um', 0)
    timestep_l3 = mesh_stats.get('timestep_L3_seconds', 0)
    
    # Extract memory usage
    memory = data.get('memory_usage', {})
    memory_usage_mb = memory.get('memory_usage_mb', 0)
    
    # Extract execution time
    exec_time = data.get('execution_time', {})
    total_time_hours = exec_time.get('total_execution_time_hours', 0)
    total_time_seconds = exec_time.get('total_execution_time_seconds', 0)
    
    # Extract computational efficiency
    efficiency = data.get('computational_efficiency', {})
    avg_time_per_step_ms = efficiency.get('avg_computation_time_per_timestep_ms', 0)
    avg_time_per_step_s = efficiency.get('avg_computation_time_per_timestep_seconds', 0)
    speedup_factor = efficiency.get('speedup_factor', 0)
    
    # Get snapshot file size
    snapshot_size_mb = get_snapshot_file_size(result_dir)
    
    return {
        'case_name': case_name,
        'total_nodes': total_nodes,
        'L3_cell_size_um': cell_size_um,
        'memory_usage_mb': memory_usage_mb,
        'total_simulation_time_hours': total_time_hours,
        'total_simulation_time_seconds': total_time_seconds,
        'timestep_L3_seconds': timestep_l3,
        'avg_computation_time_per_timestep_ms': avg_time_per_step_ms,
        'avg_computation_time_per_timestep_seconds': avg_time_per_step_s,
        'speedup_factor': speedup_factor,
        'snapshot_file_size_mb': snapshot_size_mb
    }


def main():
    """Main function to process all performance metrics and create CSV."""
    # Define paths
    results_dir = Path(__file__).parent / "results"
    output_csv = Path(__file__).parent / "performance_summary.csv"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        return
    
    # Find all performance_metrics.json files
    metrics_files = list(results_dir.glob("*/performance_metrics.json"))
    
    if not metrics_files:
        print(f"No performance_metrics.json files found in {results_dir}")
        return
    
    print(f"Found {len(metrics_files)} performance metrics files")
    
    # Process each file
    all_metrics = []
    for metrics_path in sorted(metrics_files):
        try:
            metrics = process_metrics_file(metrics_path)
            all_metrics.append(metrics)
            print(f"  Processed: {metrics['case_name']}")
        except Exception as e:
            print(f"  Error processing {metrics_path}: {e}")
    
    if not all_metrics:
        print("No metrics were successfully processed")
        return
    
    # Write to CSV
    fieldnames = [
        'case_name',
        'total_nodes',
        'L3_cell_size_um',
        'memory_usage_mb',
        'total_simulation_time_hours',
        'total_simulation_time_seconds',
        'timestep_L3_seconds',
        'avg_computation_time_per_timestep_ms',
        'avg_computation_time_per_timestep_seconds',
        'speedup_factor',
        'snapshot_file_size_mb'
    ]
    
    with open(output_csv, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_metrics)
    
    print(f"\nSummary CSV created: {output_csv}")
    print(f"Total cases processed: {len(all_metrics)}")
    
    # Print summary statistics
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    for metric in all_metrics:
        print(f"\n{metric['case_name']}:")
        print(f"  Nodes: {metric['total_nodes']:,}")
        print(f"  Cell size: {metric['L3_cell_size_um']:.2f} μm")
        print(f"  Memory: {metric['memory_usage_mb']:.2f} MB")
        print(f"  Time: {metric['total_simulation_time_hours']:.4f} hours")
        print(f"  Avg step time: {metric['avg_computation_time_per_timestep_ms']:.2f} ms")
        print(f"  Snapshot size: {metric['snapshot_file_size_mb']:.2f} MB")


if __name__ == "__main__":
    main()
