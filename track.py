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
    process_videos_parallel
)
from gui import (
    show_config_gui,
    show_polygon_selection_gui,
    show_review_selection_gui
)

def main():
    """Main function to run the worm tracking analysis"""
    # Check for review mode
    if len(sys.argv) > 1 and sys.argv[1] == "review":
        video_folder = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving - Copy"
        video_list = glob.glob(os.path.join(video_folder, '**', '*.wmv'), recursive=True)
        
        if not video_list:
            print("No .wmv files found in the directory or its subdirectories.")
            sys.exit(0)
            
        selected_videos = show_review_selection_gui(video_list)
        
        if selected_videos is None:
            print("Review selection cancelled. Exiting...")
            sys.exit(0)
            
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
            
        # TODO: Implement review functionality
        print("\nReview functionality coming soon...")
        return

    # Regular processing mode
    config = show_config_gui()
    if config is None:
        print("Configuration cancelled. Exiting...")
        sys.exit(0)

    video_folder = r"V:\Isabel videos\TrackingVideos_FoodLeaving\TrackingVideos_FoodLeaving - Copy"
    video_list = glob.glob(os.path.join(video_folder, '**', '*.wmv'), recursive=True)

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

if __name__ == "__main__":
    main()