# main.py

import sys
import os
import glob
from worm_tracker import WormTracker
from worm_tracker_utils import (
    organize_videos_by_folder,
    get_existing_polygons,
    handle_group_polygons,
    handle_individual_polygons,
    process_videos_parallel,
    review_videos
)
from gui import (
    show_config_gui,
    show_polygon_selection_gui,
    show_review_selection_gui,
    show_review_config_gui
)

from plot_utils import generate_all_plots

def analyze_videos():
    """Function to handle initial video analysis"""
    config = show_config_gui()
    if config is None:
        print("Configuration cancelled. Exiting...")
        sys.exit(0)

    video_folder = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving - Copy"
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
    """Function to handle review mode"""
    video_folder = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving - Copy"
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
    """Function to handle plot generation"""
    video_folder = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving - Copy"
    print("Starting plot generation...")
    generate_all_plots(video_folder)

def main():
    """Main function to run the worm tracking analysis"""
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