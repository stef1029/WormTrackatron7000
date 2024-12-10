import csv
import matplotlib.pyplot as plt

def plot_worm_counts(csv_path, output_plot="worm_counts_over_time.png"):
    frames = []
    worms_inside = []
    worms_outside = []
    
    # Read the CSV file
    with open(csv_path, mode='r', newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            frame = int(row['Frame'])
            inside = int(row['Worms_Inside'])
            outside = int(row['Worms_Outside'])
            
            frames.append(frame)
            worms_inside.append(inside)
            worms_outside.append(outside)
    
    # Create the plot
    plt.figure(figsize=(10,6))
    plt.plot(frames, worms_inside, label="Worms Inside", color='green')
    plt.plot(frames, worms_outside, label="Worms Outside", color='red')
    
    plt.title("Worm Counts Over Time")
    plt.xlabel("Frame Number")
    plt.ylabel("Number of Worms")
    plt.legend()
    plt.grid(True)
    
    # Set the y-axis limit as requested
    plt.ylim(0, 14)
    
    # Save the plot
    plt.savefig(output_plot, dpi=300)
    print(f"Plot saved to {output_plot}")

    # Uncomment the following line if you want to display the plot interactively:
    # plt.show()


# Example usage:
if __name__ == "__main__":
    csv_file_path = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving\A006 - 20241205_173236_worms_count.csv"  # Replace with the path to your CSV file
    plot_worm_counts(csv_file_path, output_plot="worm_counts_plot.png")
