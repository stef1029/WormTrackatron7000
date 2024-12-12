import sys
import os
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                           QHBoxLayout, QLabel, QCheckBox, QPushButton, QFrame,
                           QSpinBox, QScrollArea, QGroupBox, QLineEdit)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QFont, QPalette, QColor

class ModernConfigWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.result = None
        self.initUI()
        
    def initUI(self):
        self.setWindowTitle('Worm Tracker Configuration')
        self.setMinimumSize(500, 600)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QGroupBox {
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin-top: 1em;
                font-weight: bold;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 3px 0 3px;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QCheckBox {
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
            }
            QSpinBox {
                padding: 5px;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                min-width: 80px;  /* Made wider */
                font-size: 14px;  /* Larger font */
            }
            QLabel {
                color: #424242;
                font-size: 14px;  /* Larger font */
                min-width: 100px;  /* Minimum width for labels */
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Working Directory Group
        dir_group = QGroupBox("Working Directory")
        dir_layout = QVBoxLayout()
        dir_label = QLabel(os.getcwd())
        dir_label.setWordWrap(True)
        dir_layout.addWidget(dir_label)
        dir_group.setLayout(dir_layout)
        layout.addWidget(dir_group)

        # Processing Options Group
        options_group = QGroupBox("Processing Options")
        options_layout = QVBoxLayout()
        
        self.save_video = QCheckBox("Save labeled videos")
        self.plot_results = QCheckBox("Generate plots")
        self.show_plot = QCheckBox("Show plots during processing")
        self.group_videos = QCheckBox("Group videos by folder (share ROI)")
        self.create_trace = QCheckBox("Create movement trace map")  # New option
        
        for cb in [self.save_video, self.plot_results, self.show_plot, 
                  self.group_videos, self.create_trace]:  # Added create_trace
            options_layout.addWidget(cb)
            
        options_group.setLayout(options_layout)
        layout.addWidget(options_group)

        # Image Processing Parameters Group
        params_group = QGroupBox("Image Processing Parameters")
        params_layout = QVBoxLayout()

        # Threshold - now with expanded layout
        threshold_layout = QHBoxLayout()
        threshold_label = QLabel("Threshold:")
        self.threshold_spin = QSpinBox()
        self.threshold_spin.setRange(0, 255)
        self.threshold_spin.setValue(220)
        self.threshold_spin.setFixedWidth(100)  # Fixed width for consistency
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(self.threshold_spin)
        threshold_layout.addStretch()  # Pushes widgets to the left
        params_layout.addLayout(threshold_layout)

        # Add some vertical spacing
        params_layout.addSpacing(10)

        # Blur kernel - now with expanded layout
        blur_layout = QHBoxLayout()
        blur_label = QLabel("Blur kernel:")
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(1, 31)
        self.blur_spin.setValue(7)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setFixedWidth(100)  # Fixed width for consistency
        blur_layout.addWidget(blur_label)
        blur_layout.addWidget(self.blur_spin)
        blur_layout.addStretch()  # Pushes widgets to the left
        params_layout.addLayout(blur_layout)

        # Add some vertical spacing
        params_layout.addSpacing(10)

        # Invert colors with proper alignment
        invert_layout = QHBoxLayout()
        self.invert_colors = QCheckBox("Invert colors")
        invert_layout.addWidget(self.invert_colors)
        invert_layout.addStretch()
        params_layout.addLayout(invert_layout)
        
        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        # Start button
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.on_confirm)
        self.start_button.setFixedHeight(50)
        layout.addWidget(self.start_button)

        layout.addStretch()

    def on_confirm(self):
        if self.threshold_spin.value() < 0 or self.threshold_spin.value() > 255 or self.blur_spin.value() < 1:
            return
            
        blur = self.blur_spin.value()
        self.result = {
            'save_video': self.save_video.isChecked(),
            'plot_results': self.plot_results.isChecked(),
            'show_plot': self.show_plot.isChecked(),
            'group_videos': self.group_videos.isChecked(),
            'create_trace': self.create_trace.isChecked(),  # New option
            'threshold': self.threshold_spin.value(),
            'blur_kernel': (blur, blur),
            'invert_colors': self.invert_colors.isChecked()
        }
        self.close()

class ModernPolygonSelector(QMainWindow):
    def __init__(self, video_list, polygon_files):
        super().__init__()
        self.video_list = video_list
        self.polygon_files = polygon_files
        self.checkboxes = {}
        self.result = None  # Changed to None default to detect cancellation
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select Videos to Use Previous Polygons')
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#exitButton {
                background-color: #f44336;
            }
            QPushButton#exitButton:hover {
                background-color: #d32f2f;
            }
            QCheckBox {
                spacing: 8px;
                padding: 8px;
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin: 2px;
            }
            QCheckBox:hover {
                background-color: #f8f8f8;
            }
            QLabel {
                color: #424242;
                font-weight: bold;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Instructions
        instructions = QLabel("Select which videos to use previous polygon coords for:\n(Unselected videos will be re-labeled)")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Scroll area for video list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 8px;
            }
        """)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        for video_path in self.video_list:
            if self.polygon_files[video_path] is not None:
                checkbox = QCheckBox(os.path.basename(video_path))
                checkbox.setChecked(True)
                self.checkboxes[video_path] = checkbox
                scroll_layout.addWidget(checkbox)
            else:
                label = QLabel(f"{os.path.basename(video_path)} (no previous polygon)")
                label.setStyleSheet("""
                    padding: 8px;
                    background-color: #f5f5f5;
                    border-radius: 4px;
                """)
                scroll_layout.addWidget(label)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Button container
        button_container = QHBoxLayout()
        
        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.setObjectName("exitButton")  # For specific styling
        exit_button.setFixedHeight(50)
        exit_button.clicked.connect(self.on_exit)
        button_container.addWidget(exit_button)

        # Confirm button
        confirm_button = QPushButton("Confirm Selection")
        confirm_button.setFixedHeight(50)
        confirm_button.clicked.connect(self.on_confirm)
        button_container.addWidget(confirm_button)

        layout.addLayout(button_container)

    def on_confirm(self):
        self.result = {}
        for video_path in self.video_list:
            if video_path in self.checkboxes:
                self.result[video_path] = self.checkboxes[video_path].isChecked()
            else:
                self.result[video_path] = False
        self.close()

    def on_exit(self):
        self.result = None  # Explicitly set to None on exit
        self.close()

    def closeEvent(self, event):
        """Handle window close button event"""
        if self.result is None:  # If close button clicked without confirming
            self.result = None
        event.accept()

class VideoReviewSelector(QMainWindow):
    def __init__(self, video_list):
        super().__init__()
        self.video_list = video_list
        self.checkboxes = {}
        self.result = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Select Videos for Review')
        self.setMinimumSize(600, 400)
        self.setStyleSheet("""
            QMainWindow {
                background-color: #f5f5f5;
            }
            QWidget {
                background-color: #f5f5f5;
            }
            QPushButton {
                background-color: #2196F3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 10px 20px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton#exitButton {
                background-color: #f44336;
            }
            QPushButton#exitButton:hover {
                background-color: #d32f2f;
            }
            QCheckBox {
                spacing: 8px;
                padding: 8px;
                background-color: white;
                border: 1px solid #e0e0e0;
                border-radius: 4px;
                margin: 2px;
            }
            QCheckBox:hover {
                background-color: #f8f8f8;
            }
            QLabel {
                color: #424242;
                font-weight: bold;
            }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Instructions
        instructions = QLabel("Select videos to review:")
        instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(instructions)

        # Select/Deselect All buttons
        buttons_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        select_all_btn.clicked.connect(self.select_all)
        deselect_all_btn.clicked.connect(self.deselect_all)
        buttons_layout.addWidget(select_all_btn)
        buttons_layout.addWidget(deselect_all_btn)
        layout.addLayout(buttons_layout)

        # Scroll area for video list
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("""
            QScrollArea {
                border: none;
                background-color: white;
                border-radius: 8px;
            }
        """)
        
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout(scroll_widget)
        scroll_layout.setSpacing(5)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        for video_path in self.video_list:
            checkbox = QCheckBox(os.path.basename(video_path))
            checkbox.setChecked(True)  # Default to checked
            self.checkboxes[video_path] = checkbox
            scroll_layout.addWidget(checkbox)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)

        # Button container
        button_container = QHBoxLayout()
        
        # Exit button
        exit_button = QPushButton("Exit")
        exit_button.setObjectName("exitButton")
        exit_button.setFixedHeight(50)
        exit_button.clicked.connect(self.on_exit)
        button_container.addWidget(exit_button)

        # Confirm button
        confirm_button = QPushButton("Start Review")
        confirm_button.setFixedHeight(50)
        confirm_button.clicked.connect(self.on_confirm)
        button_container.addWidget(confirm_button)

        layout.addLayout(button_container)

    def select_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(True)

    def deselect_all(self):
        for checkbox in self.checkboxes.values():
            checkbox.setChecked(False)

    def on_confirm(self):
        self.result = {
            video_path: checkbox.isChecked()
            for video_path, checkbox in self.checkboxes.items()
        }
        self.close()

    def on_exit(self):
        self.result = None
        self.close()

def show_review_selection_gui(video_list):
    app = QApplication.instance() or QApplication(sys.argv)
    window = VideoReviewSelector(video_list)
    window.show()
    app.exec()
    return window.result

def show_polygon_selection_gui(video_list, polygon_files):
    app = QApplication.instance() or QApplication(sys.argv)
    window = ModernPolygonSelector(video_list, polygon_files)
    window.show()
    app.exec()
    return window.result

def show_config_gui():
    app = QApplication.instance() or QApplication(sys.argv)
    window = ModernConfigWindow()
    window.show()
    app.exec()
    return window.result

# def show_polygon_selection_gui(video_list, polygon_files):
#     app = QApplication.instance() or QApplication(sys.argv)
#     window = ModernPolygonSelector(video_list, polygon_files)
#     window.show()
#     app.exec()
#     return window.result