# WORMTRACKATRON7000

**Worm Tracker** is a Python-based tool for processing and analyzing worm-tracking videos in `.wmv` format. It facilitates segmenting worms inside a defined polygonal region of interest (ROI) versus those outside it, saving results to CSV, generating labeled output videos (optional), producing trace maps, and reviewing and adjusting worm counts post-processing.

## Features

1. **Polygon ROI Selection:**  
   For each video or group of videos (organized by their folder structure), you can select or reuse previously saved polygons to define the area of interest.

2. **Video Processing & Counting:**  
   Using image processing and thresholding, the software detects and counts worms inside and outside the polygon. It stores the counts in a CSV file for each video.

3. **Parallel Processing:**  
   Multiple videos are processed in parallel using `multiprocessing`, speeding up analysis for larger datasets.

4. **Trace Maps & Visualization:**  
   Generates a color-coded "trace map" image showing where worms were detected most frequently inside and outside the polygon over time.

5. **Review Mode:**  
   Allows you to revisit completed videos, review worm counts, and apply offset corrections to inside/outside counts across one or multiple videos (if grouped by folder).

6. **Plot Generation:**  
   Automatically generates plots showing original and offset-adjusted worm counts over time for each processed CSV file.

## Requirements

- Python 3.8+
- Packages:
  - `opencv-python` (OpenCV)
  - `numpy`
  - `pandas`
  - `matplotlib`
  - `PyQt6`

You can install the dependencies using:
```bash
pip install opencv-python numpy pandas matplotlib PyQt6  
```

## Usage
### Running the analysis
```bash
wormtrack analyse
```
**Process:**
- The tool first asks for configuration parameters (threshold, blur kernel, etc.).
- You select if you want to group videos by folder, whether to reuse previously saved polygons, and if you want to save labeled videos.
- Videos found in the current working directory and its subdirectories (`.wmv` files) are processed. Worm counts are saved to CSV files named `[video_basename]_worms_count.csv`, and optional labeled videos are saved as `[video_basename]_labeled.avi`.
- A polygon ROI file (`[video_basename]_polygon.npy`) and a trace map file (`[video_basename]_trace.png`) are generated.  

### Running the review
```bash
wormtrack review
```
**Process:**
- Displays a simplified configuration GUI for review mode.
- Allows you to select which processed videos to review. Videos without trace files will be disabled for selection.
- For selected videos, you can adjust worm counts by applying offsets to inside/outside counts.
- If videos are grouped by folder, you can apply the same offsets to all videos in that folder. 
   
The CSV files are updated with the offset corrections after confirmation.
  
### Generating plots
```bash
wormtrack plot
```
**Process:**
- Scans for all CSV files in the current directory and subdirectories.
- Generates `*_plot_original.png` and `*_plot_adjusted.png` graphs for each CSV file, showing worm counts over frames.

## File structure
- `main.py`
The main entry point for the application.  
Handles command-line arguments and launches the appropriate mode (`analyse`, `review`, or `plot`).    
  
- `worm_tracker.py`
Contains the `WormTracker` class, which handles video loading, polygon ROI selection, worm counting, and trace map creation.  
  
- `worm_tracker_utils.py`
Utility functions to:
    - Organize videos by folder.
    - Handle polygon selection and loading/saving.
    - Process videos in parallel.
    - Update CSV files with offsets.
    - Manage the review workflow.

- `gui.py`
Contains GUI classes and functions for:  
    - Configuration windows (`ModernConfigWindow`, `ReviewConfigWindow`).
    - Polygon selection windows (`ModernPolygonSelector`).
    - Video selection for review (`VideoReviewSelector`).
    - Offset adjustment window (`OffsetAdjustmentWindow`).

- `plot_utils.py`
Functions for generating plots from CSV files.

## Input/ output Files
- **Input:**
`.wmv` video files located in the current directory and subdirectories.
- **Output:**
    - `[video_basename]_polygon.npy`: Polygon ROI coordinates.
    - `[video_basename]_worms_count.csv`: CSV file with frame-by-frame worm counts.
    - `[video_basename]_labeled.avi`: Optional labeled video with worm outlines and counts.
    - `[video_basename]_trace.png`: Color-coded trace map image.
    - `[video_basename]_plot_original.png` and `[video_basename]_plot_adjusted.png`: Plots showing original and adjusted worm counts over time.

## Notes
- Ensure that the working directory is the folder containing your `.wmv` videos before running `wormtrack`.
- The GUI-based approach may require a display environment (i.e., not headless) when run locally.
- This code was tested with `.wmv` format videos. Other formats may work if OpenCV supports them.

## License
This project is distributed under the MIT License. See `LICENSE` file for more details.