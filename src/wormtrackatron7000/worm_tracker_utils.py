# worm_tracker_utils.py

import os
import glob
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
from functools import partial
import pandas as pd
from PyQt6.QtWidgets import QApplication

from worm_tracker import WormTracker
from gui import show_polygon_selection_gui, show_config_gui, show_review_selection_gui, OffsetAdjustmentWindow

def organize_videos_by_folder(video_list: List[str]) -> Dict[str, List[str]]:
    """Group videos by their containing folder.
    
    Args:
        video_list: List of video file paths
        
    Returns:
        Dictionary mapping folder paths to lists of video paths
    """
    folders = {}
    for video_path in video_list:
        folder = os.path.dirname(video_path)
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(video_path)
    return folders

def get_existing_polygons(video_list: List[str]) -> Dict[str, Optional[str]]:
    """Check for existing polygon files for each video.
    
    Args:
        video_list: List of video file paths
        
    Returns:
        Dictionary mapping video paths to polygon file paths (or None if not found)
    """
    polygon_files = {}
    for v in video_list:
        base_name = os.path.splitext(os.path.basename(v))[0]
        poly_file = os.path.join(os.path.dirname(v), f"{base_name}_polygon.npy")
        polygon_files[v] = poly_file if os.path.exists(poly_file) else None
    return polygon_files

def handle_group_polygons(
    folders: Dict[str, List[str]], 
    use_previous: Dict[str, bool], 
    polygon_files: Dict[str, Optional[str]]
) -> Dict[str, List[Tuple[int, int]]]:
    """Handle polygon selection and assignment for grouped videos.
    
    Args:
        folders: Dictionary mapping folders to video paths
        use_previous: Dictionary mapping video paths to boolean indicating whether to use previous polygon
        polygon_files: Dictionary mapping video paths to polygon file paths
        
    Returns:
        Dictionary mapping video paths to polygon coordinates
    """
    video_coords = {}
    
    for folder, folder_videos in folders.items():
        group_poly_file = os.path.join(folder, 'group_polygon.npy')
        folder_coords = None

        # Check for reusable polygon
        reuse_found = False
        for video_path in folder_videos:
            if use_previous.get(video_path, False) and polygon_files[video_path] is not None:
                folder_coords = np.load(polygon_files[video_path], allow_pickle=True).tolist()
                reuse_found = True
                print(f"Reusing polygon from {os.path.basename(video_path)} for folder {os.path.basename(folder)}")
                break

        # Create new polygon if needed
        if not reuse_found:
            first_video = folder_videos[0]
            print(f"\nDrawing polygon for folder: {os.path.basename(folder)}")
            print(f"Using video: {os.path.basename(first_video)}")
            
            temp_tracker = WormTracker(video_path=first_video,
                                       polygon_coords=None,
                                       save_video=False,
                                       create_trace=False)
            temp_tracker.select_roi_polygon()
            folder_coords = temp_tracker.roi_points
            temp_tracker.release()

            np.save(group_poly_file, folder_coords)
            print(f"Saved group polygon for folder {os.path.basename(folder)}")

        # Apply coordinates to all videos in folder
        for video_path in folder_videos:
            video_coords[video_path] = folder_coords
            
    return video_coords

def handle_individual_polygons(
    video_list: List[str], 
    use_previous: Dict[str, bool], 
    polygon_files: Dict[str, Optional[str]]
) -> Dict[str, List[Tuple[int, int]]]:
    """Handle polygon selection and assignment for individual videos.
    
    Args:
        video_list: List of video file paths
        use_previous: Dictionary indicating whether to use previous polygon for each video
        polygon_files: Dictionary mapping video paths to polygon file paths
        
    Returns:
        Dictionary mapping video paths to polygon coordinates
    """
    video_coords = {}
    for video_path in video_list:
        coords = None
        if use_previous[video_path] and polygon_files[video_path] is not None:
            coords = np.load(polygon_files[video_path], allow_pickle=True).tolist()
        else:
            temp_tracker = WormTracker(video_path=video_path,
                                       polygon_coords=None,
                                       save_video=False,
                                       create_trace=False)
            temp_tracker.select_roi_polygon()
            coords = temp_tracker.roi_points
            temp_tracker.release()
        video_coords[video_path] = coords
    return video_coords

def process_single_video(args: Tuple[str, List[Tuple[int, int]], Dict[str, Any]]) -> str:
    """Process a single video - to be run in parallel.
    
    Args:
        args: Tuple containing (video_path, polygon_coordinates, configuration_dictionary)
        
    Returns:
        Status message string
    """
    video_path, coords, config_dict = args
    
    tracker = WormTracker(
        video_path, 
        polygon_coords=coords,
        save_video=config_dict['save_video'],
        create_trace=True  # Always True now
    )
    
    tracker.set_parameters(
        threshold_value=config_dict['threshold'],
        blur_kernel_size=config_dict['blur_kernel'],
        invert_colors=config_dict['invert_colors']
    )
    
    if tracker.save_video:
        tracker.initialize_video_writer()
    
    tracker.process_video()
    tracker.release()
    return f"Completed processing {video_path}"

def process_videos_parallel(
    video_list: List[str], 
    video_coords: Dict[str, List[Tuple[int, int]]], 
    config: Dict[str, Any]
) -> None:
    """Process videos in parallel using multiprocessing.
    
    Args:
        video_list: List of video file paths
        video_coords: Dictionary mapping video paths to polygon coordinates
        config: Configuration dictionary
    """
    num_processes = min(mp.cpu_count(), len(video_list))
    pool = mp.Pool(processes=num_processes)
    
    processing_args = [
        (video_path, video_coords[video_path], dict(config)) 
        for video_path in video_list
    ]
    
    results = []
    for args in processing_args:
        result = pool.apply_async(process_single_video, (args,))
        results.append(result)

    for result in results:
        print(result.get())

    pool.close()
    pool.join()

def update_csv_with_offsets(csv_path, offsets):
    """Update CSV file with offset values"""
    df = pd.read_csv(csv_path)
    
    # Add offset columns if they don't exist
    if 'Inside_Offset' not in df.columns:
        df['Inside_Offset'] = 0
    if 'Outside_Offset' not in df.columns:
        df['Outside_Offset'] = 0
    
    # Set offset values
    df['Inside_Offset'] = offsets['inside_offset']
    df['Outside_Offset'] = offsets['outside_offset']
    
    # Save updated CSV
    df.to_csv(csv_path, index=False)

def organize_review_videos(selected_videos, group_videos=False):
    """Organize videos for review, either individually or by folder.
    
    Args:
        selected_videos: List of video paths to review
        group_videos: If True, group videos by folder
        
    Returns:
        List of (video_path, trace_path, csv_paths) tuples to review
    If grouped, video_path and trace_path are from one representative video,
    but csv_paths contains all CSVs in that group that need updating.
    """
    if not group_videos:
        # Return individual videos
        review_items = []
        for video_path in selected_videos:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            trace_path = os.path.join(os.path.dirname(video_path), f"{base_name}_trace.png")
            csv_path = os.path.join(os.path.dirname(video_path), f"{base_name}_worms_count.csv")
            if os.path.exists(trace_path) and os.path.exists(csv_path):
                review_items.append((video_path, trace_path, [csv_path]))
        return review_items
    
    # Group videos by folder
    folders = {}
    for video_path in selected_videos:
        folder = os.path.dirname(video_path)
        if folder not in folders:
            folders[folder] = []
        folders[folder].append(video_path)
    
    # For each folder, select one representative video and collect all CSVs
    review_items = []
    for folder, folder_videos in folders.items():
        # Use the first video as representative
        rep_video = folder_videos[0]
        rep_base_name = os.path.splitext(os.path.basename(rep_video))[0]
        rep_trace = os.path.join(folder, f"{rep_base_name}_trace.png")
        
        # Collect all CSV files for the group
        csv_paths = []
        for video in folder_videos:
            base_name = os.path.splitext(os.path.basename(video))[0]
            csv_path = os.path.join(folder, f"{base_name}_worms_count.csv")
            if os.path.exists(csv_path):
                csv_paths.append(csv_path)
        
        if os.path.exists(rep_trace) and csv_paths:
            review_items.append((rep_video, rep_trace, csv_paths))
    
    return review_items

def review_videos(selected_videos, group_videos=False):
    """Process videos for review, with optional grouping.
    
    Args:
        selected_videos: List of video paths to review
        group_videos: If True, apply same offsets to videos in same folder
    """
    app = QApplication.instance() or QApplication([])
    
    # Organize videos for review
    review_items = organize_review_videos(selected_videos, group_videos)
    
    for video_path, trace_path, csv_paths in review_items:
        # Show adjustment window
        if group_videos:
            folder_name = os.path.basename(os.path.dirname(video_path))
            print(f"\nReviewing folder: {folder_name}")
            print(f"Using representative video: {os.path.basename(video_path)}")
            print(f"This will affect {len(csv_paths)} videos in the folder")
        
        window = OffsetAdjustmentWindow(video_path, trace_path)
        window.show()
        app.exec()
        
        # Handle result
        if window.result is not None:
            if group_videos:
                print(f"Applying offsets to all videos in folder {folder_name}:")
            else:
                print(f"Applying offsets to {os.path.basename(video_path)}:")
                
            print(f"Inside offset: {window.result['inside_offset']}")
            print(f"Outside offset: {window.result['outside_offset']}")
            
            # Update all relevant CSV files
            for csv_path in csv_paths:
                update_csv_with_offsets(csv_path, window.result)
        else:
            if group_videos:
                print(f"Skipped folder {folder_name}, no changes made")
            else:
                print(f"Skipped {os.path.basename(video_path)}, no changes made")