import os
import re
import matplotlib.pyplot as plt

def parse_statistics_file(filepath):
    """Parse statistics_PMPG.txt file and extract depth, overlap depth, and width."""
    data = {
        'depth': [],
        'overlap_depth': [],
        'width': [],
        'track_ids': []
    }
    
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        
        # Extract Valley depths (limit to 45)
        valley_match = re.search(r'Valley\s+(\d+):\s+([\d.]+)\s+μm', line)
        if valley_match:
            track_id = int(valley_match.group(1))
            if track_id <= 45:
                depth = float(valley_match.group(2))
                if track_id not in data['track_ids']:
                    data['track_ids'].append(track_id)
                data['depth'].append(depth)
        
        # Extract Peak depths (limit to 44)
        peak_match = re.search(r'Peak\s+(\d+):\s+([\d.]+)\s+μm', line)
        if peak_match:
            peak_id = int(peak_match.group(1))
            if peak_id <= 44:
                overlap_depth = float(peak_match.group(2))
                data['overlap_depth'].append(overlap_depth)
        
        # Extract Track widths (limit to 45)
        track_match = re.search(r'Track\s+(\d+):\s+([\d.]+)\s+μm', line)
        if track_match:
            track_id = int(track_match.group(1))
            if track_id <= 45:
                width = float(track_match.group(2))
                data['width'].append(width)
    
    return data

def plot_statistics(all_data):
    """Create plots for all statistics."""
    # AMBench reference values
    ambench_values = {
        '5x5 mm, 0.75 ms Edge': {'depth': 170.26, 'width': 131.50, 'overlap_depth': 114.25},
        '5x5 mm, 0.75 ms Mid': {'depth': 175.59, 'width': 120.95, 'overlap_depth': 115.85},
        '1x5 mm, 0.75 ms': {'depth': 176.62, 'width': 133.68, 'overlap_depth': None},
        '5x5 mm, 5.0 ms Edge': {'depth': 149.74, 'width': 85.16, 'overlap_depth': 102.29},
        '5x5 mm, 5.0 ms Mid': {'depth': 159.89, 'width': 96.67, 'overlap_depth': 99.51},
        '1x5 mm, 5.0 ms': {'depth': 156.43, 'width': 97.77, 'overlap_depth': 120.06},
    }
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    metrics = ['depth', 'overlap_depth', 'width']
    titles = ['Depth', 'Overlap Depth', 'Width']
    ylabels = ['Depth (μm)', 'Overlap Depth (μm)', 'Width (μm)']
    
    # Define order of labels to control legend arrangement
    label_order = [
        '1x5 mm, 0.75 ms',
        '1x5 mm, 5.0 ms',
        '5x5 mm, 0.75 ms Edge',
        '5x5 mm, 0.75 ms Mid',
        '5x5 mm, 5.0 ms Edge',
        '5x5 mm, 5.0 ms Mid',
    ]
    
    for idx, (metric, title, ylabel) in enumerate(zip(metrics, titles, ylabels)):
        ax = axes[idx]
        
        # Plot in specified order
        for label in label_order:
            if label in all_data and all_data[label][metric]:
                data = all_data[label]
                # Use track IDs for x-axis
                x_values = list(range(1, len(data[metric]) + 1))
                line = ax.plot(x_values, data[metric], marker='o', label=label, linewidth=2.5, markersize=6, alpha=0.7)
                
                # Add AMBench reference line with same color (no label yet)
                if label in ambench_values and ambench_values[label][metric] is not None:
                    ambench_val = ambench_values[label][metric]
                    color = line[0].get_color()
                    ax.axhline(y=ambench_val, color=color, linestyle='--', linewidth=2, alpha=0.6)
                
                # Add dummy line for AMBench Reference after 2nd item (3rd position)
                if label == label_order[1]:
                    ax.plot([], [], color='gray', linestyle='--', linewidth=2, alpha=0.6, label='AMBench Reference')
        
        # Only add x-axis label for bottom plot
        if idx == 2:
            ax.set_xlabel('Laser Track Number', fontsize=16)
        
        # Don't add y-axis label for any plot
        ax.set_title(title, fontsize=18, fontweight='bold')
        
        # Increase tick label sizes
        ax.tick_params(axis='both', which='major', labelsize=14)
        
        # Adjust legend position for bottom plot
        if idx == 2:  # Width plot (bottom)
            ax.legend(fontsize=13, ncol=3, loc='upper right', bbox_to_anchor=(1.0, 0.8))
        else:
            ax.legend(fontsize=13, ncol=3)
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/home/mzomoro1/bin/GO-MELT/statistics_plot.png', dpi=300, bbox_inches='tight')
    print("Plot saved to statistics_plot.png")

def main():
    base_dir = "/home/mzomoro1/bin/GO-MELT/"
    
    # Define file paths
    file_paths = {
        '1x5 mm, 0.75 ms': os.path.join(base_dir, 'finalRectangleFast', 'PMPG', 'statistics_PMPG.txt'),
        '1x5 mm, 5.0 ms': os.path.join(base_dir, 'finalRectangleSlow', 'PMPG', 'statistics_PMPG.txt'),
        '5x5 mm, 0.75 ms Edge': os.path.join(base_dir, 'finalSquareFast', 'PMPG', 'edge', 'statistics_PMPG.txt'),
        '5x5 mm, 0.75 ms Mid': os.path.join(base_dir, 'finalSquareFast', 'PMPG', 'mid', 'statistics_PMPG.txt'),
        '5x5 mm, 5.0 ms Edge': os.path.join(base_dir, 'finalSquareSlow', 'PMPG', 'edge', 'statistics_PMPG.txt'),
        '5x5 mm, 5.0 ms Mid': os.path.join(base_dir, 'finalSquareSlow', 'PMPG', 'mid', 'statistics_PMPG.txt'),
    }
    
    all_data = {}
    
    # Parse all files
    for label, filepath in file_paths.items():
        if os.path.exists(filepath):
            print(f"Processing {label}...")
            all_data[label] = parse_statistics_file(filepath)
            print(f"  Valleys: {len(all_data[label]['depth'])}, Peaks: {len(all_data[label]['overlap_depth'])}, Tracks: {len(all_data[label]['width'])}")
        else:
            print(f"Warning: {filepath} not found")
    
    if all_data:
        plot_statistics(all_data)
    else:
        print("No data found to plot")

if __name__ == "__main__":
    main()
