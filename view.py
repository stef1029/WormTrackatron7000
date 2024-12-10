import cv2
import os
import numpy as np

class WormTracker:
    def __init__(self, video_path, output_folder):
        """
        Initialize the WormTracker with the path to a video file
        and the folder to save the processed output video.
        """
        self.video_path = video_path
        self.output_folder = output_folder
        self.cap = cv2.VideoCapture(video_path)

        if not self.cap.isOpened():
            raise ValueError(f"Error: Could not open video file at {video_path}")

        # Ensure the output folder exists
        os.makedirs(output_folder, exist_ok=True)

        self.threshold_value = 100
        self.blur_kernel_size = (5, 5)
        self.invert_colors = False

        # Prepare video writer
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.output_video_path = os.path.join(self.output_folder, "labeled_output.avi")
        self.out = cv2.VideoWriter(self.output_video_path, fourcc, self.fps, (self.width, self.height))

    def set_parameters(self, threshold_value=None, blur_kernel_size=None, invert_colors=None):
        """
        Set the parameters for image processing.
        """
        if threshold_value is not None:
            self.threshold_value = threshold_value
        if blur_kernel_size is not None:
            self.blur_kernel_size = blur_kernel_size
        if invert_colors is not None:
            self.invert_colors = invert_colors

    def normalize_background(self, frame):
        """
        Normalize the background of the frame to remove vignetting.
        """
        background = cv2.GaussianBlur(frame, (51, 51), 0)
        normalized_frame = cv2.divide(frame, background, scale=255)
        return normalized_frame

    def process_video(self):
        """
        Process the entire video frame by frame, detect worms in each frame,
        draw their contours, and write the labeled frames into a new output video.
        """
        frame_count = 0
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frame_count += 1

            # Convert to grayscale
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Normalize the background
            gray_frame = self.normalize_background(gray_frame)

            # Apply Gaussian blur
            blurred_frame = cv2.GaussianBlur(gray_frame, self.blur_kernel_size, 0)

            # Apply thresholding
            thresh_type = cv2.THRESH_BINARY_INV if self.invert_colors else cv2.THRESH_BINARY
            _, thresholded_frame = cv2.threshold(blurred_frame, self.threshold_value, 255, thresh_type)

            # Invert if worms appear as black blobs on white; we want white blobs on black
            thresholded_frame = 255 - thresholded_frame

            # Find contours (worms)
            contours, _ = cv2.findContours(thresholded_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Convert thresholded frame to BGR for drawing
            labeled_frame = cv2.cvtColor(thresholded_frame, cv2.COLOR_GRAY2BGR)

            worm_count = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                # Adjust this area filter as needed to ignore noise
                if area > 300:
                    worm_count += 1
                    # Draw the contour in green
                    cv2.drawContours(labeled_frame, [contour], -1, (0, 255, 0), 2)

                    # Calculate contour centroid to label worm number
                    M = cv2.moments(contour)
                    if M['m00'] != 0:
                        cx = int(M['m10'] / M['m00'])
                        cy = int(M['m01'] / M['m00'])
                        cv2.putText(labeled_frame, str(worm_count), (cx, cy - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # Put the total worm count at the top of the frame
            cv2.putText(labeled_frame, f"Worm Count: {worm_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            # Write the labeled frame to the output video
            self.out.write(labeled_frame)

        print(f"Processing complete. Labeled video saved to {self.output_video_path}")

    def release(self):
        """
        Release the video capture and video writer objects.
        """
        self.cap.release()
        self.out.release()


# Example usage:
if __name__ == "__main__":
    video_path = r"/cephfs2/srogers/Isabel videos/TrackingVideos_FoodLeaving/TrackingVideos_FoodLeaving/A005 - 20241205_173220.wmv"  # Replace with your video file path
    output_folder = "output_frames"    # Replace with your desired output folder

    tracker = WormTracker(video_path, output_folder)

    try:
        # Adjust parameters as needed
        tracker.set_parameters(threshold_value=220, blur_kernel_size=(7, 7), invert_colors=False)
        
        # Process the entire video and create a labeled output
        tracker.process_video()
    finally:
        tracker.release()
