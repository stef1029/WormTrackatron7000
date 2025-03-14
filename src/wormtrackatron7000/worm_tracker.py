import cv2
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys


class WormTracker:
    def __init__(self, video_path, polygon_coords=None, save_video=True, create_trace=True):
        self.video_path = video_path
        self.polygon_coords = polygon_coords
        self.save_video = save_video
        self.create_trace = True  # New parameter

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

    def set_parameters(self, threshold_value=None, blur_kernel_size=None, invert_colors=None):
        if threshold_value is not None:
            self.threshold_value = threshold_value
        if blur_kernel_size is not None:
            self.blur_kernel_size = blur_kernel_size
        if invert_colors is not None:
            self.invert_colors = invert_colors

    def normalize_background(self, frame):
        background = cv2.GaussianBlur(frame, (51, 51), 0)
        normalized_frame = cv2.divide(frame, background, scale=255)
        return normalized_frame

    def select_roi_polygon(self, video_index=0, video_total=1, scale_factor=0.5):
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

    def point_in_polygon(self, point, polygon):
        test = cv2.pointPolygonTest(polygon, point, False)
        return test >= 0

    def initialize_video_writer(self):
        """Initialize video writer for labeled output"""
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            self.out = cv2.VideoWriter(
                self.video_output_path, 
                fourcc, 
                self.fps, 
                (self.width, self.height)
            )

    def process_video(self):
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
        self.cap.release()
        if self.out is not None:
            self.out.release()