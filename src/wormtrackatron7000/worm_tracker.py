"""
Core worm tracking module for the WORMTRACKATRON7000 application.

This module contains the WormTracker class which handles video processing, worm detection,
polygon region of interest (ROI) selection, and generation of analysis outputs including
count data, trace maps, and labeled videos.
"""

import cv2
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys


class WormTracker:
    """
    Main class for worm tracking and video analysis.
    
    This class processes video files to track and count worms, distinguishing
    between those inside and outside a user-defined polygon region of interest (ROI).
    It can create labeled output videos, trace maps showing worm paths, and CSV data
    of worm counts.
    
    Attributes:
        video_path (str): Path to the video file being processed.
        polygon_coords (list or None): List of (x,y) coordinates defining the ROI polygon.
        save_video (bool): Whether to save a labeled output video.
        create_trace (bool): Whether to create a trace map of worm paths.
        use_contour_detection (bool): Whether to use full contour-based detection instead
            of centroid-based detection.
        cap (cv2.VideoCapture): OpenCV video capture object.
        output_dir (str): Directory where output files will be saved.
        threshold_value (int): Threshold value for binary image processing.
        blur_kernel_size (tuple): Size of the Gaussian blur kernel.
        invert_colors (bool): Whether to invert colors during processing.
        width (int): Width of the video frames.
        height (int): Height of the video frames.
        fps (float): Frames per second of the video.
        csv_path (str): Path where the CSV output will be saved.
        video_output_path (str): Path where the labeled video will be saved.
        trace_output_path (str): Path where the trace map will be saved.
        polygon_file (str): Path where the polygon coordinates will be saved.
        out (cv2.VideoWriter or None): Video writer for labeled output.
        trace_map (numpy.ndarray): Array for building the trace map.
        roi_points (list): List of (x,y) coordinates defining the ROI polygon.
        roi_final (numpy.ndarray): Array of ROI polygon coordinates for OpenCV functions.
    """
    
    def __init__(self, video_path, polygon_coords=None, save_video=True, create_trace=True, use_contour_detection=False):
        """
        Initialize the WormTracker with video path and options.
        
        Args:
            video_path (str): Path to the video file to process.
            polygon_coords (list, optional): List of (x,y) coordinates defining the ROI polygon.
                If None, user will be prompted to define a polygon. Defaults to None.
            save_video (bool, optional): Whether to save a labeled output video. Defaults to True.
            create_trace (bool, optional): Whether to create a trace map. Defaults to True.
            use_contour_detection (bool, optional): Whether to use full contour-based detection
                instead of centroid-based detection. Defaults to False.
        
        Raises:
            ValueError: If the video file cannot be opened.
        """
        self.video_path = video_path
        self.polygon_coords = polygon_coords
        self.save_video = save_video
        self.create_trace = True  # New parameter
        self.use_contour_detection = use_contour_detection  # New parameter for detection method

        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file at {video_path}")

        self.output_dir = os.path.dirname(video_path)
        base_name = os.path.splitext(os.path.basename(video_path))[0]

        self.threshold_value = 100
        self.blur_kernel_size = (5, 5)
        self.invert_colors = False

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        self.csv_path = os.path.join(self.output_dir, f"{base_name}_worms_count.csv")
        self.video_output_path = os.path.join(self.output_dir, f"{base_name}_labeled.avi")
        self.trace_output_path = os.path.join(self.output_dir, f"{base_name}_trace.png")
        self.polygon_file = os.path.join(self.output_dir, f"{base_name}_polygon.npy")

        self.out = None
        
        # Initialize trace map
        if self.create_trace:
            self.trace_map = np.zeros((self.height, self.width, 3), dtype=np.uint8)

        self.roi_points = []
        self.roi_final = None
        if self.polygon_coords is not None:
            self.roi_points = self.polygon_coords
            self.roi_final = np.array(self.roi_points, dtype=np.int32)

    def set_parameters(self, threshold_value=None, blur_kernel_size=None, invert_colors=None, use_contour_detection=None):
        """
        Set image processing parameters for worm detection.
        
        Args:
            threshold_value (int, optional): Threshold value for binary image processing (0-255).
            blur_kernel_size (tuple, optional): Size of the Gaussian blur kernel as (width, height).
            invert_colors (bool, optional): Whether to invert colors during processing.
            use_contour_detection (bool, optional): Whether to use full contour-based detection.
        """
        if threshold_value is not None:
            self.threshold_value = threshold_value
        if blur_kernel_size is not None:
            self.blur_kernel_size = blur_kernel_size
        if invert_colors is not None:
            self.invert_colors = invert_colors
        if use_contour_detection is not None:
            self.use_contour_detection = use_contour_detection

    def point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon.
        
        Args:
            point (tuple): (x, y) coordinates of the point to check.
            polygon (numpy.ndarray): Array of polygon vertices.
            
        Returns:
            bool: True if the point is inside or on the polygon, False otherwise.
        """
        test = cv2.pointPolygonTest(polygon, point, False)
        return test >= 0
        
    def contour_in_polygon(self, contour, polygon):
        """
        Check if a contour is at least partially inside a polygon.
        
        For contour-based detection:
        - Returns True if ANY point of the contour is inside the polygon
        - Returns False only if ALL points are outside the polygon
        
        Args:
            contour (numpy.ndarray): Contour to check.
            polygon (numpy.ndarray): Array of polygon vertices.
            
        Returns:
            bool: True if any part of the contour is inside the polygon, False otherwise.
        """
        # Simplify the contour to reduce computation
        epsilon = 0.01 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        # Check each point in the contour
        for point in approx_contour:
            # Extract the x,y coordinates correctly from the point
            x, y = point.ravel()  # Flattens the array to get coordinates
            
            # Check if this point is inside the polygon
            if cv2.pointPolygonTest(polygon, (float(x), float(y)), False) >= 0:
                # If any point is inside, the contour is considered inside
                return True
        
        # All points are outside
        return False

    def normalize_background(self, frame):
        """
        Normalize the background of a frame to improve worm detection.
        
        This method divides the frame by a blurred version of itself to
        reduce the effect of uneven lighting and improve contrast.
        
        Args:
            frame (numpy.ndarray): Input frame to normalize.
            
        Returns:
            numpy.ndarray: Normalized frame.
        """
        background = cv2.GaussianBlur(frame, (51, 51), 0)
        normalized_frame = cv2.divide(frame, background, scale=255)
        return normalized_frame

    def select_roi_polygon(self, video_index=0, video_total=1, scale_factor=0.5):
        """
        Allow the user to select a polygon region of interest on the first frame.
        
        If polygon coordinates are already known (self.roi_final is not None),
        this method does nothing. Otherwise, it displays the first frame and
        lets the user draw a polygon by clicking points.
        
        Args:
            video_index (int, optional): Index of current video when processing multiple.
                Defaults to 0.
            video_total (int, optional): Total number of videos being processed.
                Defaults to 1.
            scale_factor (float, optional): Scale factor for display purposes.
                Defaults to 0.5.
                
        Raises:
            ValueError: If the first frame cannot be read or if not enough points
                are selected to form a polygon.
        """
        if self.roi_final is not None:
            # Polygon known, skip
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            return

        ret, frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read the first frame for ROI selection.")

        display_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        original_display = display_frame.copy()
        frame_copy = display_frame.copy()

        window_title = f"Select ROI (Video {video_index+1}/{video_total}): Click points, Enter=done, z=reset, Esc=quit"
        scaled_points = []

        def mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                scaled_points.append((x, y))
                cv2.circle(frame_copy, (x, y), 3, (0, 0, 255), -1)
                if len(scaled_points) > 1:
                    cv2.line(frame_copy, scaled_points[-2], scaled_points[-1], (0,0,255), 1)
                cv2.imshow("Select ROI", frame_copy)

        cv2.namedWindow("Select ROI")
        cv2.setMouseCallback("Select ROI", mouse_callback)

        while True:
            cv2.setWindowTitle("Select ROI", window_title)
            cv2.imshow("Select ROI", frame_copy)
            key = cv2.waitKey(1)
            if key == 13:  # Enter
                if len(scaled_points) >= 3:
                    break
                else:
                    print("Need at least 3 points to form a polygon.")
            elif key == ord('z'):
                scaled_points.clear()
                frame_copy = original_display.copy()
                cv2.imshow("Select ROI", frame_copy)
            elif key == 27:  # Esc
                print("User requested exit.")
                cv2.destroyAllWindows()
                sys.exit(0)

        cv2.destroyWindow("Select ROI")

        if len(scaled_points) < 3:
            raise ValueError("Not enough points to form a polygon.")

        self.roi_points = [(int(pt[0]/scale_factor), int(pt[1]/scale_factor)) for pt in scaled_points]
        self.roi_final = np.array(self.roi_points, dtype=np.int32)

        np.save(self.polygon_file, self.roi_points)
        print(f"Polygon saved for {self.video_path} at {self.polygon_file}")

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def initialize_video_writer(self):
        """
        Initialize video writer for labeled output video.
        
        This method sets up a video writer to save the processed video with worms
        labeled as inside or outside the polygon.
        """
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(
                self.video_output_path, 
                fourcc, 
                self.fps, 
                (self.width, self.height)
            )

    def process_video(self):
        """
        Process the video to track and count worms.
        
        This is the main processing method that:
        1. Reads each frame of the video
        2. Detects worms using image processing techniques
        3. Determines if each worm is inside or outside the polygon
        4. Counts worms in each category
        5. Optionally creates a labeled output video
        6. Builds a frequency map for the trace visualization
        7. Saves worm counts to a CSV file
        8. Creates a trace map showing where worms were most frequently detected
        
        The method uses either centroid-based detection (default) or full contour-based
        detection based on the use_contour_detection attribute.
        """
        with open(self.csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame", "Worms_Inside", "Worms_Outside"])

            frame_count = 0
            
            # Store original first frame for final trace image
            ret, first_frame = self.cap.read()
            if not ret:
                return
                
            # Create separate frequency maps for inside and outside dots
            if self.create_trace:
                self.trace_map = np.zeros_like(first_frame, dtype=np.uint8)
                # Use float32 for frequency counting, separate for inside/outside
                self.frequency_map_inside = np.zeros((self.height, self.width), dtype=np.float32)
                self.frequency_map_outside = np.zeros((self.height, self.width), dtype=np.float32)
            
            # Reset video to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            while True:
                ret, frame = self.cap.read()
                if not ret:
                    break
                frame_count += 1

                gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frame = self.normalize_background(gray_frame)
                blurred_frame = cv2.GaussianBlur(gray_frame, self.blur_kernel_size, 0)

                thresh_type = cv2.THRESH_BINARY_INV if self.invert_colors else cv2.THRESH_BINARY
                _, thresholded_frame = cv2.threshold(blurred_frame, self.threshold_value, 255, thresh_type)

                thresholded_frame = 255 - thresholded_frame
                contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                worm_count_inside = 0
                worm_count_outside = 0
                worm_id = 0

                if self.save_video:
                    labeled_frame = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)
                else:
                    labeled_frame = None

                # Create temporary masks for this frame's dots
                if self.create_trace:
                    temp_mask_inside = np.zeros((self.height, self.width), dtype=np.uint8)
                    temp_mask_outside = np.zeros((self.height, self.width), dtype=np.uint8)

                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > 175:
                        worm_id += 1
                        M = cv2.moments(contour)
                        if M['m00'] != 0:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])

                            # Determine if worm is inside polygon based on detection method
                            if self.use_contour_detection:
                                is_inside = self.contour_in_polygon(contour, self.roi_final)
                            else:
                                is_inside = self.point_in_polygon((cx, cy), self.roi_final)
                                
                            if is_inside:
                                worm_count_inside += 1
                                # Draw on inside mask with larger radius
                                if self.create_trace:
                                    cv2.circle(temp_mask_inside, (cx, cy), 10, 255, -1)
                            else:
                                worm_count_outside += 1
                                # Draw on outside mask with larger radius
                                if self.create_trace:
                                    cv2.circle(temp_mask_outside, (cx, cy), 10, 255, -1)

                            if self.save_video:
                                contour_color = (0, 255, 0) if is_inside else (0, 0, 255)
                                cv2.drawContours(labeled_frame, [contour], -1, contour_color, 2)
                                cv2.putText(labeled_frame, str(worm_id), (cx, cy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 1)

                # Update frequency maps with temporary masks
                if self.create_trace:
                    self.frequency_map_inside += (temp_mask_inside > 0).astype(np.float32)
                    self.frequency_map_outside += (temp_mask_outside > 0).astype(np.float32)

                # Write frame data
                csv_writer.writerow([frame_count, worm_count_inside, worm_count_outside])

                if self.save_video:
                    cv2.putText(labeled_frame, f"Inside: {worm_count_inside}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    cv2.putText(labeled_frame, f"Outside: {worm_count_outside}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    cv2.polylines(labeled_frame, [self.roi_final], True, (255,0,0), 2)
                    self.out.write(labeled_frame)

            if self.create_trace:
                # Normalize frequencies separately
                max_freq = max(np.max(self.frequency_map_inside), np.max(self.frequency_map_outside))
                
                # Create color maps based on frequencies
                heat_map = np.zeros_like(self.trace_map)
                
                # Convert frequencies to color intensities
                inside_normalized = (self.frequency_map_inside / max_freq * 255).astype(np.uint8)
                outside_normalized = (self.frequency_map_outside / max_freq * 255).astype(np.uint8)
                
                # Apply colors (green for inside, red for outside)
                heat_map[:, :, 0] = outside_normalized  # Blue channel
                heat_map[:, :, 1] = inside_normalized   # Green channel
                heat_map[:, :, 2] = outside_normalized  # Red channel
                
                # Create combined mask
                mask = ((inside_normalized > 0) | (outside_normalized > 0)).astype(np.uint8)
                mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                
                # Blend with original frame
                alpha = 0.7
                final_trace = cv2.addWeighted(
                    first_frame,
                    1.0,
                    heat_map,
                    alpha,
                    0
                )
                
                # Add colorbar legend - one for each color
                legend_height = 30
                legend_width = 256
                padding = 10
                
                # Add space for two colorbars
                final_trace = cv2.copyMakeBorder(
                    final_trace,
                    0, (legend_height + padding) * 2 + padding,
                    0, 0,
                    cv2.BORDER_CONSTANT,
                    value=[255, 255, 255]
                )
                
                # Create and add inside (green) colorbar
                inside_colorbar = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
                for i in range(legend_width):
                    inside_colorbar[:, i] = [0, i, 0]  # green gradient
                    
                # Create and add outside (red) colorbar
                outside_colorbar = np.zeros((legend_height, legend_width, 3), dtype=np.uint8)
                for i in range(legend_width):
                    outside_colorbar[:, i] = [0, 0, i]  # red gradient
                
                # Add colorbars to image
                y_offset_inside = final_trace.shape[0] - (legend_height + padding) * 2
                y_offset_outside = final_trace.shape[0] - legend_height - padding
                x_offset = padding
                
                final_trace[y_offset_inside:y_offset_inside+legend_height, 
                        x_offset:x_offset+legend_width] = inside_colorbar
                final_trace[y_offset_outside:y_offset_outside+legend_height, 
                        x_offset:x_offset+legend_width] = outside_colorbar
                
                # Add text labels
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(final_trace, 'Inside (Green) - Low', 
                        (x_offset, y_offset_inside + legend_height + 15), 
                        font, 0.5, (0, 0, 0), 1)
                cv2.putText(final_trace, 'High', 
                        (x_offset + legend_width - 30, y_offset_inside + legend_height + 15), 
                        font, 0.5, (0, 0, 0), 1)
                        
                cv2.putText(final_trace, 'Outside (Red) - Low', 
                        (x_offset, y_offset_outside + legend_height + 15), 
                        font, 0.5, (0, 0, 0), 1)
                cv2.putText(final_trace, 'High', 
                        (x_offset + legend_width - 30, y_offset_outside + legend_height + 15), 
                        font, 0.5, (0, 0, 0), 1)
                
                # Draw the ROI polygon
                cv2.polylines(final_trace, [self.roi_final], True, (255,0,0), 2)
                
                cv2.imwrite(self.trace_output_path, final_trace)
                print(f"Trace map saved to {self.trace_output_path}")

    def release(self):
        """
        Release resources used by the WormTracker.
        
        This method releases the video capture and writer objects to free resources.
        It should be called when processing is complete or when the tracker is no longer needed.
        """
        self.cap.release()
        if self.out is not None:
            self.out.release()