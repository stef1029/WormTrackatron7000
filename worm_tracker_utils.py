# worm_tracker_utils.py

import os
import glob
import numpy as np
import multiprocessing as mp
from typing import Dict, List, Optional, Tuple, Any
from functools import partial

from worm_tracker import WormTracker
from gui import show_polygon_selection_gui, show_config_gui, show_review_selection_gui

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
            
            temp_tracker = WormTracker(first_video, None, False, False, False)
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
            temp_tracker = WormTracker(video_path, None, False, False, False)
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
        plot_results=config_dict['plot_results'],
        show_plot=config_dict['show_plot'],
        create_trace=config_dict['create_trace']
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