import cv2
import os
import numpy as np
import csv
import matplotlib.pyplot as plt
import sys


class WormTracker:
    def __init__(self, video_path, polygon_coords=None, save_video=True, plot_results=False, show_plot=False, create_trace=False):
        self.video_path = video_path
        self.polygon_coords = polygon_coords
        self.save_video = save_video
        self.plot_results = plot_results
        self.show_plot = show_plot
        self.create_trace = create_trace  # New parameter

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

    def process_video(self):
        with open(self.csv_path, mode='w', newline='') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(["Frame", "Worms_Inside", "Worms_Outside"])

            frame_count = 0
            
            # Store original first frame for final trace image
            ret, first_frame = self.cap.read()
            if not ret:
                return
                
            # Create trace overlay with transparency
            if self.create_trace:
                self.trace_map = np.zeros_like(first_frame, dtype=np.uint8)
            
            # Reset video to start
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

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
                                contour_color = (0, 255, 0)  # Green for inside
                            else:
                                worm_count_outside += 1
                                contour_color = (0, 0, 255)  # Red for outside

                            # Update trace map with semi-transparent dots
                            if self.create_trace:
                                cv2.circle(self.trace_map, (cx, cy), 3, contour_color, -1)

                            if self.save_video:
                                cv2.drawContours(labeled_frame, [contour], -1, contour_color, 2)
                                cv2.putText(labeled_frame, str(worm_id), (cx, cy - 5),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, contour_color, 1)

                csv_writer.writerow([frame_count, worm_count_inside, worm_count_outside])

                if self.save_video:
                    cv2.putText(labeled_frame, f"Inside: {worm_count_inside}", (10,30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
                    cv2.putText(labeled_frame, f"Outside: {worm_count_outside}", (10,70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
                    cv2.polylines(labeled_frame, [self.roi_final], True, (255,0,0), 2)
                    self.out.write(labeled_frame)

            print(f"Processing complete for {self.video_path}.\nCSV saved to {self.csv_path}")
            if self.save_video:
                print(f"Labeled video saved to {self.video_output_path}")

            # Create final trace image by blending with first frame
            if self.create_trace:
                # Create a mask where traces are drawn
                trace_mask = cv2.cvtColor(cv2.threshold(cv2.cvtColor(self.trace_map, cv2.COLOR_BGR2GRAY), 
                                                    1, 255, cv2.THRESH_BINARY)[1], cv2.COLOR_GRAY2BGR)
                
                # Blend the original frame with traces
                alpha = 0.7  # Adjust this value to change trace visibility (0.0-1.0)
                final_trace = cv2.addWeighted(first_frame, 1.0, self.trace_map, alpha, 0)
                
                # Draw the ROI polygon
                cv2.polylines(final_trace, [self.roi_final], True, (255,0,0), 2)
                
                cv2.imwrite(self.trace_output_path, final_trace)
                print(f"Trace map saved to {self.trace_output_path}")

            if self.plot_results:
                self._plot_worm_counts(self.csv_path, self.show_plot)

    def _plot_worm_counts(self, csv_path, show_plot=False):
        frames = []
        worms_inside = []
        worms_outside = []

        with open(csv_path, mode='r', newline='') as file:
            reader = csv.DictReader(file)
            for row in reader:
                frame = int(row['Frame'])
                inside = int(row['Worms_Inside'])
                outside = int(row['Worms_Outside'])
                frames.append(frame)
                worms_inside.append(inside)
                worms_outside.append(outside)

        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.basename(csv_path)
        base_name = os.path.splitext(csv_name)[0]
        output_plot = os.path.join(csv_dir, f"{base_name}_plot.png")

        plt.figure(figsize=(10,6))
        plt.plot(frames, worms_inside, label="Worms Inside", color='green')
        plt.plot(frames, worms_outside, label="Worms Outside", color='red')

        plt.title("Worm Counts Over Time")
        plt.xlabel("Frame Number")
        plt.ylabel("Number of Worms")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 14)

        plt.savefig(output_plot, dpi=300)
        print(f"Plot saved to {output_plot}")

        if show_plot:
            plt.show()
        plt.close()

    def release(self):
        self.cap.release()
        if self.out is not None:
            self.out.release()