"""
Main module and entry point for the WORMTRACKATRON7000 application.

This module provides the command-line interface for the worm tracking application,
with three main modes of operation:
1. 'analyse' - Process videos to track and count worms
2. 'review' - Review processed videos and adjust counts
3. 'plot' - Generate visualization plots from processed data

The module coordinates the video processing workflow by calling the appropriate
functions from other modules based on user input and configuration.
"""

import sys
import os
import glob
from wormtrackatron7000.worm_tracker import WormTracker
from wormtrackatron7000.worm_tracker_utils import (
    organize_videos_by_folder,
    get_existing_polygons,
    handle_group_polygons,
    handle_individual_polygons,
    process_videos_parallel,
    review_videos
)
from wormtrackatron7000.gui import (
    show_config_gui,
    show_polygon_selection_gui,
    show_review_selection_gui,
    show_review_config_gui
)

from wormtrackatron7000.plot_utils import generate_all_plots

def analyze_videos():
    """
    Function to handle initial video analysis.
    
    This function implements the main workflow for video analysis:
    1. Display configuration GUI to get user settings
    2. Find videos in the current directory and subdirectories
    3. Check for existing polygon ROIs and let user choose which to reuse
    4. Handle polygon selection for videos (grouped or individual)
    5. Process videos in parallel
    
    The function uses a graphical user interface for configuration and polygon
    selection, then processes videos according to the specified settings.
    """
    config = show_config_gui()
    if config is None:
        print("Configuration cancelled. Exiting...")
        sys.exit(0)

    video_folder = os.getcwd()
    video_list = glob.glob(os.path.join(video_folder, '**', '*.wmv'), recursive=True)
    # video_list = [video_list[0]]

    if not video_list:
        print("No .wmv files found in the directory or its subdirectories.")
        sys.exit(0)

    # Get existing polygon information
    polygon_files = get_existing_polygons(video_list)
    use_previous = show_polygon_selection_gui(video_list, polygon_files)
    
    if use_previous is None:
        print("Polygon selection cancelled. Exiting...")
        sys.exit(0)

    # Handle polygon selection based on grouping option
    if config['group_videos']:
        folders = organize_videos_by_folder(video_list)
        video_coords = handle_group_polygons(folders, use_previous, polygon_files)
    else:
        video_coords = handle_individual_polygons(video_list, use_previous, polygon_files)

    # Process videos in parallel
    process_videos_parallel(video_list, video_coords, config)
    print("All videos processed successfully.")

def review_mode():
    """
    Function to handle review mode.
    
    This function implements the workflow for reviewing processed videos:
    1. Find videos in the current directory and subdirectories
    2. Display review configuration GUI to get user settings
    3. Let user select which videos to review
    4. Display offset adjustment windows for selected videos
    5. Update CSVs with adjusted worm counts
    
    The function allows users to manually adjust worm counts by applying
    offset values, either individually or grouped by folder.
    """
    video_folder = os.getcwd()
    video_list = glob.glob(os.path.join(video_folder, '**', '*.wmv'), recursive=True)
    
    if not video_list:
        print("No .wmv files found in the directory or its subdirectories.")
        sys.exit(0)
    
    # Show simplified config GUI for review mode
    config = show_review_config_gui()
    if config is None:
        print("Configuration cancelled. Exiting...")
        sys.exit(0)
        
    # Show review selection GUI
    selected_videos = show_review_selection_gui(video_list)
    
    if selected_videos is None:
        print("Review selection cancelled. Exiting...")
        sys.exit(0)
        
    # Filter videos that were selected for review
    videos_to_review = [
        video_path for video_path, selected in selected_videos.items()
        if selected
    ]
    
    if not videos_to_review:
        print("No videos selected for review. Exiting...")
        sys.exit(0)
        
    print("Selected videos for review:")
    for video in videos_to_review:
        print(f"- {os.path.basename(video)}")
        
    print("\nStarting review process...")
    review_videos(videos_to_review, config['group_videos'])
    print("Review process complete!")

def plot_mode():
    """
    Function to handle plot generation.
    
    This function searches for all CSV files with worm count data in the
    current directory and its subdirectories, then generates visualization
    plots for each file.
    
    The plots show the number of worms inside and outside the ROI over time,
    both with and without any offset adjustments applied.
    """
    video_folder = os.getcwd()
    print("Starting plot generation...")
    generate_all_plots(video_folder)

def main():
    """
    Main function to run the worm tracking analysis.
    
    This function parses command-line arguments to determine which mode to run:
    - 'analyse': Process videos to track and count worms
    - 'review': Review processed videos and adjust counts
    - 'plot': Generate visualization plots
    
    Usage: python wt.py [analyse|review|plot]
    """
    if len(sys.argv) != 2 or sys.argv[1] not in ['analyse', 'review', 'plot']:
        print("Usage: python wt.py [analyse|review|plot]")
        sys.exit(1)
        
    if sys.argv[1] == 'analyse':
        analyze_videos()
    elif sys.argv[1] == 'review':
        review_mode()
    else:  # plot mode
        plot_mode()

if __name__ == "__main__":
    main()