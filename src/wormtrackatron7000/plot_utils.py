# plot_utils.py

import os
import csv
import matplotlib.pyplot as plt
import glob

def plot_worm_counts(csv_path):
    """Create plots for a single CSV file showing original and adjusted counts."""
    frames = []
    worms_inside = []
    worms_outside = []
    worms_inside_adjusted = []
    worms_outside_adjusted = []

    print(f"Processing: {os.path.basename(csv_path)}")
    
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['Frame'])
            inside = int(row['Worms_Inside'])
            outside = int(row['Worms_Outside'])
            inside_offset = int(row.get('Inside_Offset', 0))
            outside_offset = int(row.get('Outside_Offset', 0))
            
            frames.append(frame)
            worms_inside.append(inside)
            worms_outside.append(outside)
            worms_inside_adjusted.append(inside + inside_offset)
            worms_outside_adjusted.append(outside + outside_offset)

    if not frames:
        print(f"No data found in {csv_path}")
        return

    csv_dir = os.path.dirname(csv_path)
    csv_name = os.path.basename(csv_path)
    base_name = os.path.splitext(csv_name)[0]

    # Plot original values
    plt.figure(figsize=(10,6))
    plt.plot(frames, worms_inside, label="Worms Inside", color='green')
    plt.plot(frames, worms_outside, label="Worms Outside", color='red')

    plt.title("Original Worm Counts Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Number of Worms")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 14)
    plt.xlim(min(frames), max(frames))

    output_plot = os.path.join(csv_dir, f"{base_name}_plot_original.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()

    # Plot adjusted values
    plt.figure(figsize=(10,6))
    plt.plot(frames, worms_inside_adjusted, label="Worms Inside (Adjusted)", color='green')
    plt.plot(frames, worms_outside_adjusted, label="Worms Outside (Adjusted)", color='red')

    plt.title("Adjusted Worm Counts Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Number of Worms")
    plt.legend()
    plt.grid(True)
    plt.ylim(0, 14)
    plt.xlim(min(frames), max(frames))

    output_plot = os.path.join(csv_dir, f"{base_name}_plot_adjusted.png")
    plt.savefig(output_plot, dpi=300)
    plt.close()
    
    print(f"Generated plots for {os.path.basename(csv_path)}")

def generate_all_plots(folder_path):
    """Generate plots for all CSV files in the given folder and subfolders."""
    csv_files = glob.glob(os.path.join(folder_path, '**', '*_worms_count.csv'), recursive=True)
    
    if not csv_files:
        print("No CSV files found.")
        return
    
    print(f"Found {len(csv_files)} CSV files to process.")
    for csv_file in csv_files:
        plot_worm_counts(csv_file)
    
    print("All plots generated successfully.")