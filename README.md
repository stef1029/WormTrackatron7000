# Worm Tracker

Worm Tracker is a sophisticated Python application designed for tracking and analyzing worm movement in video recordings. The software uses computer vision techniques to detect worms, classify their positions relative to defined regions of interest (ROIs), and generate comprehensive movement analysis data and visualizations.

## Features

- Video Analysis
  - Automated worm detection and tracking
  - User-defined polygon regions of interest (ROI)
  - Frame-by-frame analysis of worm positions
  - Support for batch processing multiple videos
  - Options to group videos by folder for shared ROI settings

- Data Generation
  - CSV output with frame-by-frame worm counts
  - Heat map generation showing worm movement patterns
  - Optional labeled video output with tracking visualization
  - Adjustable tracking parameters for different experimental conditions

- Review and Adjustment
  - Interactive review interface for processed videos
  - Manual offset adjustments for worm counts
  - Batch adjustment capabilities for grouped videos
  - Automated plot generation for visualization

## Installation

### Prerequisites

- Python 3.8 or higher
- OpenCV (cv2)
- PyQt6
- NumPy
- Pandas
- Matplotlib

Install the required dependencies using pip:

```bash
pip install opencv-python-headless PyQt6 numpy pandas matplotlib
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/worm-tracker.git
cd worm-tracker
```

2. Ensure all files are in the correct directory structure:
```
worm-tracker/
├── __init__.py
├── gui.py
├── plot_utils.py
├── worm_tracker.py
├── worm_tracker_utils.py
└── wt.py
```

## Usage

The application provides three main modes of operation: analysis, review, and plotting. Each mode is accessed through different command-line arguments.

### Analysis Mode

Use this mode for initial video processing:

```bash
python wt.py analyse
```

This will:
1. Launch a configuration GUI for setting processing parameters
2. Allow selection of previous ROI polygons or creation of new ones
3. Process all videos in the current directory and subdirectories
4. Generate CSV files with tracking data and optional visualization outputs

### Review Mode

Use this mode to review and adjust processed videos:

```bash
python wt.py review
```

This will:
1. Show a selection interface for choosing videos to review
2. Allow manual adjustment of worm counts
3. Update CSV files with adjusted counts

### Plot Mode

Use this mode to generate visualization plots:

```bash
python wt.py plot
```

This will create plots for all processed videos in the current directory and subdirectories.

## Configuration Options

### Processing Parameters

- **Threshold Value**: Brightness threshold for worm detection (0-255)
- **Blur Kernel**: Size of Gaussian blur kernel for noise reduction
- **Color Inversion**: Option to invert video colors for different lighting conditions
- **Save Labeled Videos**: Option to save videos with tracking visualization
- **Group Videos**: Option to apply same ROI to all videos in a folder

### Review Options

- **Group Videos**: Apply same count adjustments to all videos in a folder
- **Manual Offsets**: Adjust inside and outside worm counts independently

## File Outputs

The application generates several output files for each processed video:

- `*_worms_count.csv`: Frame-by-frame worm count data
  - Columns: Frame, Worms_Inside, Worms_Outside, Inside_Offset, Outside_Offset
- `*_polygon.npy`: Saved ROI polygon coordinates
- `*_trace.png`: Heat map visualization of worm movement patterns
- `*_labeled.avi`: (Optional) Video with tracking visualization
- `*_plot_original.png`: Plot of original worm counts
- `*_plot_adjusted.png`: Plot of adjusted worm counts

## Advanced Usage

### Folder Organization

The application supports recursive processing of videos in subdirectories. When using the group videos option, all videos in the same folder will share:
- ROI polygon coordinates
- Processing parameters
- Review adjustments

### Batch Processing

For large datasets, the application uses parallel processing to analyze multiple videos simultaneously. The number of parallel processes is automatically determined based on available CPU cores.

### Error Handling

The application includes robust error handling for:
- Missing or corrupted video files
- Invalid ROI selections
- Processing parameter validation
- File I/O operations

## Troubleshooting

Common issues and solutions:

1. **Video Loading Fails**
   - Ensure video files are in WMV format
   - Check file permissions
   - Verify video file integrity

2. **Poor Tracking Results**
   - Adjust threshold value for different lighting conditions
   - Modify blur kernel size for different video qualities
   - Try inverting colors for better contrast

3. **Performance Issues**
   - Reduce video resolution if processing is slow
   - Ensure sufficient disk space for output files
   - Close other resource-intensive applications

## Contributing

Please read CONTRIBUTING.md for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments

- OpenCV team for computer vision libraries
- PyQt team for GUI framework
- Scientific community for feedback and testing

## Citation

If you use this software in your research, please cite:

```
@software{worm_tracker,
  author = {Your Name},
  title = {Worm Tracker: A Tool for Automated Worm Movement Analysis},
  year = {2025},
  url = {https://github.com/yourusername/worm-tracker}
}
```