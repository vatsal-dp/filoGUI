# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:48:25 2025
@author: ataasso
"""


from skimage.io import imread
from scipy.stats import mode
import matplotlib.pyplot as plt
from skimage.morphology import disk
from skimage.morphology import thin, dilation, opening #  binary_openin, and disk taken out
import time
import scipy.io as sio
import pickle
import os
import subprocess
import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFileDialog, QMessageBox, QGroupBox, QFrame, QScrollArea
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import torch
torch.backends.mkldnn.enabled = False
import numpy as np

class ModelsView(QWidget):
    def __init__(self):
        super().__init__()

        self.models = {
            "Coni_7": os.path.join("FiloGUI_models", "Coni_7.pth"),
            "Phore_2": os.path.join("FiloGUI_models", "Phore_2.pth"),
            "FilaBranch_2": os.path.join("FiloGUI_models", "FilaBranch_2.pth"),
            "FilaCross_2": os.path.join("FiloGUI_models", "FilaCross_2.pth"),
            "FilaSeptum_4": os.path.join("FiloGUI_models", "FilaSeptum_4.pth"),
            "FilaTip_6": os.path.join("FiloGUI_models", "FilaTip_6.pth"),
            "Retrain_omni_5": os.path.join("FiloGUI_models", "Retrain_omni_5.pth")
        }

        self.selected_model = None
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Select a model:")
        self.label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px 0;")
        self.layout.addWidget(self.label)

        # Create horizontal layout for buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Left align the buttons

        self.buttons = {}
        for name in self.models:
            btn = QPushButton(name)
            btn.setFixedWidth(120)  # Slightly smaller width for horizontal layout
            btn.setFixedHeight(40)   # Set consistent height
            btn.setStyleSheet("""
            QPushButton {
                text-align: center;
                padding: 8px 12px;
                background-color: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                margin: 2px;
                color: black;
                font-size: 12px;
            }
            QPushButton:hover {
                background-color: #f5f5f5;
                border-color: #999;
            }
            QPushButton:checked {
                background-color: #0078d4;
                color: white;
                border-color: #0078d4;
            }
            """)
            btn.setCheckable(True)
            btn.clicked.connect(lambda checked, n=name: self.select_model(n))
            self.buttons[name] = btn
            buttons_layout.addWidget(btn)
            
            # Add small spacing between buttons
            buttons_layout.addSpacing(5)

        # Add stretch at the end to push buttons to the left
        buttons_layout.addStretch()
        
        # Add the horizontal buttons layout to the main layout
        self.layout.addLayout(buttons_layout)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()

        self.status = QLabel("No model selected.")
        self.status.setStyleSheet("font-size: 12px; color: #666; margin-top: 10px;")
        self.status.setAlignment(Qt.AlignmentFlag.AlignLeft)  # Left align the status text
        self.layout.addWidget(self.status)

    def select_model(self, model_name):
        # Uncheck all other buttons
        for name, button in self.buttons.items():
            button.setChecked(name == model_name)
        
        self.selected_model = model_name
        self.status.setText(f"Selected: {model_name}")

    def add_model(self, model_name, path):
        self.models[model_name] = path
        # For simplicity, just update the dict and select it
        self.select_model(model_name)

class ImageView(QWidget):
    def __init__(self, models_view):
        super().__init__()
        self.models_view = models_view

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # === Model Name Label (above image) ===
        self.model_name_label = QLabel("")
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_name_label.setStyleSheet("font-weight: bold; font-size: 14px; margin: 10px 0;")
        self.layout.addWidget(self.model_name_label)

        # === Single Centered Image Display ===
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # self.image_label.setMinimumSize(600, 600)  # Set minimum size for better display
        # self.image_label.setStyleSheet("border: 1px solid #ddd; background-color: #f9f9f9;")
        self.layout.addWidget(self.image_label)

        # === Navigation Controls (Horizontal) ===
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setFixedWidth(100)
        self.prev_btn.clicked.connect(self.show_prev_overlay)
        nav_layout.addWidget(self.prev_btn)

        nav_layout.addSpacing(20)  # Space between buttons

        self.next_btn = QPushButton("➡️ Next")
        self.next_btn.setFixedWidth(100)
        self.next_btn.clicked.connect(self.show_next_overlay)
        nav_layout.addWidget(self.next_btn)

        self.layout.addLayout(nav_layout)

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        # Overlay navigation properties
        self.overlay_index = 0
        self.overlay_pixmaps = []

    def show_next_overlay(self):
        if not self.overlay_pixmaps:
            return
        self.overlay_index = (self.overlay_index + 1) % len(self.overlay_pixmaps)
        self.image_label.setPixmap(self.overlay_pixmaps[self.overlay_index].scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.model_name_label.setText(self.overlay_model_names[self.overlay_index])

    def show_prev_overlay(self):
        if not self.overlay_pixmaps:
            return
        self.overlay_index = (self.overlay_index - 1) % len(self.overlay_pixmaps)
        self.image_label.setPixmap(self.overlay_pixmaps[self.overlay_index].scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.model_name_label.setText(self.overlay_model_names[self.overlay_index])

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)"
        )

        print(f"🔍 DEBUG - Loading image: {file_path}")
        try:
            # Quick check of the loaded image
            import tifffile
            temp_img = tifffile.imread(file_path)
            print(f"🔍 DEBUG - Loaded image shape: {temp_img.shape}, dtype: {temp_img.dtype}")
            print(f"🔍 DEBUG - Loaded image min/max: {temp_img.min()}/{temp_img.max()}")
        except Exception as e:
            print(f"🔍 DEBUG - Error reading image: {e}")
        if file_path:
            self.current_image_path = file_path
            print(f"✅ Image selected: {file_path}")
            
            # Clear the display and show just a placeholder
            self.image_label.clear()
            self.image_label.setText("Image loaded. Run segmentation to see overlay.")
            self.model_name_label.setText("")

    def compare_segmentations(self):
        from PySide6.QtCore import QTimer
        import tifffile
        import io
        import numpy as np
        import matplotlib.pyplot as plt
        from PIL import Image
    
        if not hasattr(self, 'current_image_path') or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
    
        model_keys = ["FilaBranch_2", "FilaTip_6", "Retrain_omni_5","Coni_7","Phore_2","FilaCross_2","FilaSeptum"]
        image_path = self.current_image_path
        image_folder = os.path.dirname(image_path)
        image_filename = os.path.basename(image_path)
        image_basename = os.path.splitext(image_filename)[0]
    
        self.compare_results = {}
    
        for model_key in model_keys:
            model_path = self.models_view.models.get(model_key)
            save_dir = os.path.join(image_folder, f"compare_{model_key}")
            os.makedirs(save_dir, exist_ok=True)
    
            if "omni" in model_key.lower():
                command = (
                    f'omnipose --dir "{image_folder}" '
                    f'--pretrained_model "{model_path}" --use_gpu --nchan 1 --nclasses 2 '
                    f'--verbose --save_tif --savedir "{save_dir}" '
                    f'--img_filter "{image_basename}" --no_npy'
                )
            else:
                command = (
                    f'cellpose --dir "{image_folder}" '
                    f'--pretrained_model "{model_path}" --verbose --chan 0 --chan2 0 --use_gpu '
                    f'--cellprob_threshold 0.0 --flow_threshold 0.4 '
                    f'--save_tif --savedir "{save_dir}" '
                    f'--img_filter "{image_basename}"'
                )
    
            subprocess.run(command, shell=True)
            self.compare_results[model_key] = save_dir
    
        try:
            # Load original image
            image = tifffile.imread(image_path)
            if image.ndim == 3:
                image = image[0]
    
            # Normalize image to [0, 255] and convert to RGB
            img_min, img_max = np.percentile(image, (1, 99))
            img_norm = np.clip((image - img_min) / (img_max - img_min), 0, 1)
            img_gray = (img_norm * 255).astype(np.uint8)
            image_rgb = np.stack([img_gray] * 3, axis=-1)
    
            self.overlay_pixmaps = []
            self.overlay_model_names = []
    
            for model_key in model_keys:
                mask_path = os.path.join(self.compare_results[model_key], image_basename + "_cp_masks.tif")
                if not os.path.exists(mask_path):
                    QMessageBox.warning(self, "Missing Output", f"Mask not found for {model_key}")
                    return
    
                mask = tifffile.imread(mask_path)
                if mask.ndim == 3:
                    mask = mask[0]
    
                # Prepare colored mask with jet colormap
                cmap = plt.get_cmap('jet')
                if mask.max() == 0:
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                else:
                    colored_mask = cmap(mask.astype(np.float32) / mask.max())[:, :, :3]
                    colored_mask = (colored_mask * 255).astype(np.uint8)
    
                # Alpha mask where mask > 0
                alpha = (mask > 0)[..., np.newaxis]
    
                # Blend grayscale and mask
                blended = image_rgb * (1 - alpha) + colored_mask * alpha
                blended = blended.astype(np.uint8)
    
                overlay_pil = Image.fromarray(blended)
                buf = io.BytesIO()
                overlay_pil.save(buf, format="PNG")
                buf.seek(0)
    
                pixmap = QPixmap()
                pixmap.loadFromData(buf.read(), "PNG")
                self.overlay_pixmaps.append(pixmap)
                self.overlay_model_names.append(model_key)
    
            if self.overlay_pixmaps:
                self.overlay_index = 0
                self.image_label.setPixmap(self.overlay_pixmaps[0].scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
                self.model_name_label.setText(self.overlay_model_names[0])
    
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
    
                QMessageBox.information(self, "Done", "Compared segmentations are displayed.")
    
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def run_single_image_segmentation(self):
        if not hasattr(self, 'current_image_path') or not os.path.exists(self.current_image_path):
            QMessageBox.warning(self, "No Image", "Please load an image first.")
            return
    
        model_key = self.models_view.selected_model
        if model_key is None:
            QMessageBox.warning(self, "Model Not Selected", "Please select a model.")
            return
    
        model_path = self.models_view.models.get(model_key)
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", f"Model path invalid:\n{model_path}")
            return
    
        image_folder = os.path.dirname(self.current_image_path)
        image_filename = os.path.basename(self.current_image_path)
        image_basename = os.path.splitext(image_filename)[0]  # removes extension
    
        save_dir = os.path.join(image_folder, model_key)
        os.makedirs(save_dir, exist_ok=True)

        import cv2, tifffile

        # 🔧 Load and resize image to 0.5x
        orig_img = tifffile.imread(self.current_image_path)
        if orig_img.ndim == 3:
            orig_img = orig_img[0]

        new_h, new_w = int(orig_img.shape[0] * 1), int(orig_img.shape[1] * 1)
        resized_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 🔧 Save temporary downscaled image to feed into the model
        temp_path = os.path.join(image_folder, image_basename + "_scaled.tif")
        tifffile.imwrite(temp_path, resized_img)

        # Update path so CLI sees scaled version
        image_basename = os.path.splitext(os.path.basename(temp_path))[0]
    
        # Build command
        if "omni" in model_key.lower():
            command = (
                f'omnipose --dir "{image_folder}" '
                f'--pretrained_model "{model_path}" --use_gpu --nchan 1 --nclasses 2 '
                f'--verbose --save_tif --savedir "{save_dir}" '
                f'--img_filter "{image_basename}" --no_npy'
            )
        else:
            command = (
                f'cellpose --dir "{image_folder}" '
                f'--pretrained_model "{model_path}" --verbose --chan 0 --chan2 0 --use_gpu '
                f'--cellprob_threshold 0.0 --flow_threshold 0.4 '
                f'--diameter 0 '
                f'--save_tif --savedir "{save_dir}" '
                f'--img_filter "{image_basename}"'
            )
    
        print(f"🔧 Running command:\n{command}")
    
        try:
            self.save_dir = save_dir
            self.image_dir = image_folder

            print(f"🔍 DEBUG - Model key: {model_key}")
            print(f"🔍 DEBUG - Image path: {self.current_image_path}")
            print(f"🔍 DEBUG - Save directory: {save_dir}")
            print(f"🔍 DEBUG - Expected mask file: {os.path.join(save_dir, image_basename + '_cp_masks.tif')}")

            subprocess.Popen(command, shell=True)
    
            QMessageBox.information(
                self,
                "Segmentation Started",
                f"Segmentation started on single image with model '{model_key}'"
            )
    
            from PySide6.QtCore import QTimer
            self.check_timer = QTimer(self)
            self.check_timer.timeout.connect(self.try_show_mask_from_save)
            self.check_timer.start(5000)
    
        except Exception as e:
            QMessageBox.critical(self, "Execution Error", str(e))

    def run_segmentation(self):
        pass  # Your original model.eval() segmentation code can go here if needed.

    # def run_cli_segmentation(self):
    #     import subprocess, os, cv2, tifffile, shutil
    #     from PySide6.QtCore import QTimer
    #     from PySide6.QtWidgets import QFileDialog, QMessageBox
    
    #     # STEP 1: Ask user to pick a folder
    #     folder_path = QFileDialog.getExistingDirectory(
    #         self,
    #         "Select Folder Containing Time Series Images",
    #         "",
    #         QFileDialog.Option.DontUseNativeDialog
    #     )

    #     # If the user cancels or picks nothing
    #     if not folder_path or folder_path.strip() == "":
    #         QMessageBox.warning(self, "No Folder Selected", "Please select a folder to proceed.")
    #         return
    
    #     self.image_dir = folder_path  # ✅ This is the selected folder
    
    #     model_key = self.models_view.selected_model
    #     if model_key is None:
    #         QMessageBox.warning(self, "Model Not Selected", "Please select a model before running segmentation.")
    #         return
    
    #     model_path = self.models_view.models.get(model_key)
    #     if not os.path.exists(model_path):
    #         QMessageBox.warning(self, "Invalid Model", f"The selected model path does not exist:\n{model_path}")
    #         return
    
    #     # STEP 2: Check for image files in the folder
    #     image_files = [
    #         f for f in os.listdir(self.image_dir)
    #         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
    #     ]
    #     if not image_files:
    #         QMessageBox.warning(self, "No Images Found", f"No valid image files found in:\n{self.image_dir}")
    #         return
    
    #     # STEP 3: Create output folder named after the model
    #     self.save_dir = os.path.join(self.image_dir, model_key)
    #     os.makedirs(self.save_dir, exist_ok=True)

    #     # STEP 4: PREPROCESS IMAGES (like single image function)
    #     print(f"🔧 Preprocessing {len(image_files)} images...")
    #     processed_count = 0

    #     for image_file in image_files:
    #         try:
    #             image_path = os.path.join(self.image_dir, image_file)
    #             image_basename = os.path.splitext(image_file)[0]
                
    #             # Load and process image (same as single image function)
    #             orig_img = tifffile.imread(image_path)
    #             if orig_img.ndim == 3:
    #                 orig_img = orig_img[0]  # Take first channel if 3D
                
    #             # Resize (even if 1x scale, this normalizes the data)
    #             new_h, new_w = int(orig_img.shape[0] * 1), int(orig_img.shape[1] * 1)
    #             resized_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
    #             # Save processed image
    #             temp_path = os.path.join(self.image_dir, image_basename + "_scaled.tif")
    #             tifffile.imwrite(temp_path, resized_img)
    #             processed_count += 1
                
    #         except Exception as e:
    #             print(f"⚠️ Failed to process {image_file}: {e}")

    #     if processed_count == 0:
    #         QMessageBox.warning(self, "Processing Failed", "Failed to preprocess any images.")
    #         return

    #     print(f"✅ Successfully processed {processed_count} images")

    #     # STEP 5: Create temporary directory with only processed images
    #     temp_dir = os.path.join(self.image_dir, "temp_processed")
    #     os.makedirs(temp_dir, exist_ok=True)

    #     # Move processed images to temp directory
    #     scaled_files = [f for f in os.listdir(self.image_dir) if f.endswith("_scaled.tif")]
    #     for scaled_file in scaled_files:
    #         src = os.path.join(self.image_dir, scaled_file)
    #         dst = os.path.join(temp_dir, scaled_file)
    #         shutil.move(src, dst)

    #     print(f"📁 Created temp directory with {len(scaled_files)} processed images")
        
    #     # STEP 6: Build command - use temp directory
    #     if "omni" in model_key.lower():
    #         command = (
    #             f'omnipose --dir "{temp_dir}" '
    #             f'--pretrained_model "{model_path}" --use_gpu --nchan 1 --nclasses 2 '
    #             f'--verbose --save_tif '
    #             f'--savedir "{self.save_dir}" --no_npy'
    #         )
    #     else:
    #         command = (
    #             f'cellpose --dir "{temp_dir}" '
    #             f'--pretrained_model "{model_path}" --verbose --use_gpu --chan 0 --chan2 0 '
    #             f'--cellprob_threshold 0.0 --flow_threshold 0.4 '
    #             f'--diameter 0 '
    #             f'--save_tif --savedir "{self.save_dir}"'
    #         )
    
    #     print(f"🔧 Running command:\n{command}")
    
    #     # STEP 5: Run the command
    #     try:
    #         subprocess.Popen(command, shell=True)
    #         QMessageBox.information(self, "Running", f"Segmentation started with model '{model_key}' on {len(image_files)} image(s).")
    #         self.check_timer = QTimer(self)
    #         self.check_timer.timeout.connect(self.try_show_mask_from_save)
    #         self.check_timer.start(5000)
    #     except Exception as e:
    #         QMessageBox.critical(self, "Error", str(e))

    def run_cli_segmentation(self):
        import subprocess, os, cv2, tifffile, shutil
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFileDialog, QMessageBox

        # STEP 1: Ask user to pick a folder
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Time Series Images",
            "",
            QFileDialog.Option.DontUseNativeDialog
        )

        # If the user cancels or picks nothing
        if not folder_path or folder_path.strip() == "":
            QMessageBox.warning(self, "No Folder Selected", "Please select a folder to proceed.")
            return

        self.image_dir = folder_path  # ✅ This is the selected folder

        model_key = self.models_view.selected_model
        if model_key is None:
            QMessageBox.warning(self, "Model Not Selected", "Please select a model before running segmentation.")
            return

        model_path = self.models_view.models.get(model_key)
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Invalid Model", f"The selected model path does not exist:\n{model_path}")
            return

        # STEP 2: Check for image files in the folder
        image_files = [
            f for f in os.listdir(self.image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'))
        ]
        if not image_files:
            QMessageBox.warning(self, "No Images Found", f"No valid image files found in:\n{self.image_dir}")
            return

        # Store the expected number of images for completion checking
        self.expected_mask_count = len(image_files)
        self.original_image_files = image_files.copy()  # Store original filenames

        # STEP 3: Create output folder named after the model
        self.save_dir = os.path.join(self.image_dir, model_key)
        os.makedirs(self.save_dir, exist_ok=True)

        # STEP 4: PREPROCESS IMAGES (like single image function)
        print(f"🔧 Preprocessing {len(image_files)} images...")
        processed_count = 0

        for image_file in image_files:
            try:
                image_path = os.path.join(self.image_dir, image_file)
                image_basename = os.path.splitext(image_file)[0]
                
                # Load and process image (same as single image function)
                orig_img = tifffile.imread(image_path)
                if orig_img.ndim == 3:
                    orig_img = orig_img[0]  # Take first channel if 3D
                
                # Resize (even if 1x scale, this normalizes the data)
                new_h, new_w = int(orig_img.shape[0] * 1), int(orig_img.shape[1] * 1)
                resized_img = cv2.resize(orig_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                
                # Save processed image
                temp_path = os.path.join(self.image_dir, image_basename + "_scaled.tif")
                tifffile.imwrite(temp_path, resized_img)
                processed_count += 1
                
            except Exception as e:
                print(f"⚠️ Failed to process {image_file}: {e}")

        if processed_count == 0:
            QMessageBox.warning(self, "Processing Failed", "Failed to preprocess any images.")
            return

        print(f"✅ Successfully processed {processed_count} images")

        # STEP 5: Create temporary directory with only processed images
        temp_dir = os.path.join(self.image_dir, "temp_processed")
        os.makedirs(temp_dir, exist_ok=True)

        # Move processed images to temp directory
        scaled_files = [f for f in os.listdir(self.image_dir) if f.endswith("_scaled.tif")]
        for scaled_file in scaled_files:
            src = os.path.join(self.image_dir, scaled_file)
            dst = os.path.join(temp_dir, scaled_file)
            shutil.move(src, dst)

        print(f"📁 Created temp directory with {len(scaled_files)} processed images")
        
        # STEP 6: Build command - use temp directory
        if "omni" in model_key.lower():
            command = (
                f'omnipose --dir "{temp_dir}" '
                f'--pretrained_model "{model_path}" --use_gpu --nchan 1 --nclasses 2 '
                f'--verbose --save_tif '
                f'--savedir "{self.save_dir}" --no_npy'
            )
        else:
            command = (
                f'cellpose --dir "{temp_dir}" '
                f'--pretrained_model "{model_path}" --verbose --use_gpu --chan 0 --chan2 0 '
                f'--cellprob_threshold 0.0 --flow_threshold 0.4 '
                f'--diameter 0 '
                f'--save_tif --savedir "{self.save_dir}"'
            )

        print(f"🔧 Running command:\n{command}")

        # STEP 7: Run the command
        try:
            subprocess.Popen(command, shell=True)
            QMessageBox.information(self, "Running", f"Segmentation started with model '{model_key}' on {len(image_files)} image(s).")
            
            # Initialize progress tracking
            self.check_timer = QTimer(self)
            self.check_timer.timeout.connect(self.check_folder_segmentation_progress)
            self.check_timer.start(3000)  # Check every 3 seconds
            
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))

    def check_folder_segmentation_progress(self):
        """Check progress of folder segmentation and load all completed overlays"""
        import os
        
        if not os.path.exists(self.save_dir):
            print("❌ Save folder not created yet.")
            return

        mask_files = [f for f in os.listdir(self.save_dir) if f.endswith("_cp_masks.tif")]
        completed_count = len(mask_files)
        
        print(f"🔍 Progress: {completed_count}/{self.expected_mask_count} masks completed")
        
        # If all masks are completed, stop checking and load all overlays
        if completed_count >= self.expected_mask_count:
            print("✅ All segmentation completed!")
            self.check_timer.stop()
            self.load_all_folder_overlays()
        elif completed_count > 0:
            # Show partial progress but keep checking
            print(f"⏳ Partial completion: {completed_count}/{self.expected_mask_count}")

    def load_all_folder_overlays(self):
        """Load all completed segmentation overlays for navigation"""
        import os, io
        import tifffile
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        print(f"🔍 Loading all overlays from: {self.save_dir}")
        
        if not os.path.exists(self.save_dir):
            QMessageBox.warning(self, "Error", "Save directory not found.")
            return

        mask_files = [f for f in os.listdir(self.save_dir) if f.endswith("_cp_masks.tif")]
        if not mask_files:
            QMessageBox.warning(self, "No Results", "No mask files found.")
            return

        # Clear previous overlays
        self.overlay_pixmaps = []
        self.overlay_model_names = []

        print(f"✅ Found {len(mask_files)} mask files")

        try:
            # Sort mask files to maintain consistent order
            mask_files.sort()
            
            # Process all mask files
            for mask_file in mask_files:
                base_name = mask_file.replace("_cp_masks.tif", ".tif")
                # Remove _scaled suffix if it exists to find original image
                if base_name.endswith("_scaled.tif"):
                    original_name = base_name.replace("_scaled.tif", ".tif")
                else:
                    original_name = base_name

                mask_path = os.path.join(self.save_dir, mask_file)
                
                # Try to find the original image
                image_path = None
                for potential_name in [base_name, original_name]:
                    potential_path = os.path.join(self.image_dir, potential_name)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if not image_path:
                    print(f"⚠️ Original image not found for: {mask_file}")
                    continue

                print(f"✅ Processing: {mask_file} with image: {os.path.basename(image_path)}")

                image = tifffile.imread(image_path)
                mask = tifffile.imread(mask_path)

                if image.ndim == 3:
                    image = image[0]
                if mask.ndim == 3:
                    mask = mask[0]

                # === Process original image for consistent display ===
                img_min, img_max = np.percentile(image, (1, 99))
                img_norm = np.clip((image - img_min) / (img_max - img_min), 0, 1)
                img_uint8 = (img_norm * 255).astype(np.uint8)

                # === Create overlay image ===
                image_rgb = np.stack([img_uint8] * 3, axis=-1)
            
                # Create colored mask with jet colormap
                cmap = plt.get_cmap('jet')
                if mask.max() == 0:
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                else:
                    colored_mask = cmap(mask.astype(np.float32) / mask.max())[:, :, :3]
                    colored_mask = (colored_mask * 255).astype(np.uint8)

                # Alpha mask where mask > 0
                alpha = (mask > 0)[..., np.newaxis]

                # Blend grayscale and mask
                blended = image_rgb * (1 - alpha) + colored_mask * alpha
                blended = blended.astype(np.uint8)

                overlay_pil = Image.fromarray(blended)

                buf = io.BytesIO()
                overlay_pil.save(buf, format="PNG")
                buf.seek(0)

                pixmap = QPixmap()
                pixmap.loadFromData(buf.read(), "PNG")
                
                # Add to overlay arrays
                self.overlay_pixmaps.append(pixmap)
                # Use the original filename without _scaled suffix for display
                display_name = original_name.replace(".tif", "")
                if hasattr(self, 'models_view') and self.models_view.selected_model:
                    display_name = f"{self.models_view.selected_model} - {display_name}"
                self.overlay_model_names.append(display_name)

            # Display the first overlay if we have any
            if self.overlay_pixmaps:
                self.overlay_index = 0
                self.image_label.setPixmap(
                    self.overlay_pixmaps[0].scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.model_name_label.setText(self.overlay_model_names[0])
                
                # Enable navigation buttons if we have multiple overlays
                if len(self.overlay_pixmaps) > 1:
                    self.prev_btn.setEnabled(True)
                    self.next_btn.setEnabled(True)
                    QMessageBox.information(
                        self, "Segmentation Complete", 
                        f"Processed {len(self.overlay_pixmaps)} images successfully!\nUse Previous/Next buttons to navigate through results."
                    )
                else:
                    self.prev_btn.setEnabled(False)
                    self.next_btn.setEnabled(False)
                    QMessageBox.information(self, "Segmentation Complete", "Single image processed successfully.")
            else:
                QMessageBox.warning(self, "No Results", "No valid overlays could be created.")

        except Exception as e:
            import traceback
            print("❌ Error loading overlays:", traceback.format_exc())
            QMessageBox.critical(self, "Display Error", str(e))

    def try_show_mask_from_save(self):
        import os, io
        import tifffile
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt

        print(f"🔍 Checking folder: {self.save_dir}")
        if not os.path.exists(self.save_dir):
            print("❌ Save folder not created yet.")
            return

        mask_files = [f for f in os.listdir(self.save_dir) if f.endswith("_cp_masks.tif")]
        if not mask_files:
            print("⏳ No masks found yet.")
            return

        # Stop the timer since we found masks
        self.check_timer.stop()
        
        # Clear previous overlays
        self.overlay_pixmaps = []
        self.overlay_model_names = []

        print(f"✅ Found {len(mask_files)} mask files")

        try:
            # Process all mask files
            for mask_file in mask_files:
                base_name = mask_file.replace("_cp_masks.tif", ".tif")
                # Also try without _scaled suffix if it exists
                if base_name.endswith("_scaled.tif"):
                    original_name = base_name.replace("_scaled.tif", ".tif")
                else:
                    original_name = base_name

                mask_path = os.path.join(self.save_dir, mask_file)
                
                # Try to find the original image
                image_path = None
                for potential_name in [base_name, original_name]:
                    potential_path = os.path.join(self.image_dir, potential_name)
                    if os.path.exists(potential_path):
                        image_path = potential_path
                        break
                
                if not image_path:
                    print(f"⚠️ Original image not found for: {mask_file}")
                    continue

                print(f"✅ Processing: {mask_file} with image: {os.path.basename(image_path)}")

                image = tifffile.imread(image_path)
                mask = tifffile.imread(mask_path)

                if image.ndim == 3:
                    image = image[0]
                if mask.ndim == 3:
                    mask = mask[0]

                # === Process original image for consistent display ===
                img_min, img_max = np.percentile(image, (1, 99))
                img_norm = np.clip((image - img_min) / (img_max - img_min), 0, 1)
                img_uint8 = (img_norm * 255).astype(np.uint8)

                # === Create overlay image ===
                image_rgb = np.stack([img_uint8] * 3, axis=-1)
            
                # Create colored mask with jet colormap
                cmap = plt.get_cmap('jet')
                if mask.max() == 0:
                    colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
                else:
                    colored_mask = cmap(mask.astype(np.float32) / mask.max())[:, :, :3]
                    colored_mask = (colored_mask * 255).astype(np.uint8)

                # Alpha mask where mask > 0
                alpha = (mask > 0)[..., np.newaxis]

                # Blend grayscale and mask
                blended = image_rgb * (1 - alpha) + colored_mask * alpha
                blended = blended.astype(np.uint8)

                overlay_pil = Image.fromarray(blended)

                buf = io.BytesIO()
                overlay_pil.save(buf, format="PNG")
                buf.seek(0)

                pixmap = QPixmap()
                pixmap.loadFromData(buf.read(), "PNG")
                
                # Add to overlay arrays
                self.overlay_pixmaps.append(pixmap)
                # Use the original filename without _scaled suffix for display
                display_name = original_name.replace(".tif", "")
                if hasattr(self, 'models_view') and self.models_view.selected_model:
                    display_name = f"{self.models_view.selected_model} - {display_name}"
                self.overlay_model_names.append(display_name)

            # Display the first overlay if we have any
            if self.overlay_pixmaps:
                self.overlay_index = 0
                self.image_label.setPixmap(
                    self.overlay_pixmaps[0].scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )
                self.model_name_label.setText(self.overlay_model_names[0])
                
                # Enable navigation buttons if we have multiple overlays
                if len(self.overlay_pixmaps) > 1:
                    self.prev_btn.setEnabled(True)
                    self.next_btn.setEnabled(True)
                    QMessageBox.information(
                        self, "Done", 
                        f"Segmentation complete! {len(self.overlay_pixmaps)} overlays ready. Use Previous/Next buttons to navigate."
                    )
                else:
                    self.prev_btn.setEnabled(False)
                    self.next_btn.setEnabled(False)
                    QMessageBox.information(self, "Done", "Segmentation overlay displayed.")
            else:
                QMessageBox.warning(self, "No Results", "No valid overlays could be created.")

        except Exception as e:
            import traceback
            print("❌ Error showing overlay:", traceback.format_exc())
            QMessageBox.critical(self, "Display Error", str(e))
    # def try_show_mask_from_save(self):
    #     import os, io
    #     import tifffile
    #     import numpy as np
    #     from PIL import Image
    #     import matplotlib.pyplot as plt
    
    #     print(f"🔍 Checking folder: {self.save_dir}")
    #     if not os.path.exists(self.save_dir):
    #         print("❌ Save folder not created yet.")
    #         return
    
    #     mask_files = [f for f in os.listdir(self.save_dir) if f.endswith("_cp_masks.tif")]
    #     if not mask_files:
    #         print("⏳ No masks found yet.")
    #         return
    
    #     mask_file = mask_files[0]
    #     base_name = mask_file.replace("_cp_masks.tif", ".tif")
    
    #     mask_path = os.path.join(self.save_dir, mask_file)
    #     image_path = os.path.join(self.image_dir, base_name)
    
    #     if not os.path.exists(image_path):
    #         print(f"⚠️ Original image not found: {image_path}")
    #         return
    
    #     print(f"✅ Found mask: {mask_path}")
    #     print(f"✅ Found image: {image_path}")
    #     self.check_timer.stop()
    
    #     try:
    #         image = tifffile.imread(image_path)
    #         mask = tifffile.imread(mask_path)

    #         print(f"🔍 DEBUG - Original image shape: {image.shape}, dtype: {image.dtype}")
    #         print(f"🔍 DEBUG - Original image min/max: {image.min()}/{image.max()}")
    #         print(f"🔍 DEBUG - Original mask shape: {mask.shape}, dtype: {mask.dtype}")
    #         print(f"🔍 DEBUG - Original mask min/max: {mask.min()}/{mask.max()}")
    #         print(f"🔍 DEBUG - Unique mask values: {np.unique(mask)}")
    
    #         if image.ndim == 3:
    #             image = image[0]
    #         if mask.ndim == 3:
    #             mask = mask[0]

    #         if image.ndim == 3:
    #             print(f"🔍 DEBUG - Image reduced to 2D: {image.shape}")
    #         if mask.ndim == 3:
    #             print(f"🔍 DEBUG - Mask reduced to 2D: {mask.shape}")
    
    #         # === Process original image for consistent display ===
    #         img_min, img_max = np.percentile(image, (1, 99))
    #         img_norm = np.clip((image - img_min) / (img_max - img_min), 0, 1)
    #         img_uint8 = (img_norm * 255).astype(np.uint8)

    #         print(f"🔍 DEBUG - Image after normalization shape: {img_uint8.shape}, dtype: {img_uint8.dtype}")
    #         print(f"🔍 DEBUG - Normalized image min/max: {img_uint8.min()}/{img_uint8.max()}")  

    #         print(f"🔍 DEBUG - Image for overlay shape: {image.shape}")
    #         print(f"🔍 DEBUG - Mask for overlay shape: {mask.shape}")
    #         print(f"🔍 DEBUG - Mask for overlay unique values: {np.unique(mask)}")
    #         print(f"🔍 DEBUG - Non-zero mask pixels count: {np.count_nonzero(mask)}")
    
    #         # === Create overlay image ===
    #         image_rgb = np.stack([img_uint8] * 3, axis=-1)
        
    #         # Create colored mask with jet colormap
    #         cmap = plt.get_cmap('jet')
    #         if mask.max() == 0:
    #             colored_mask = np.zeros((*mask.shape, 3), dtype=np.uint8)
    #         else:
    #             colored_mask = cmap(mask.astype(np.float32) / mask.max())[:, :, :3]
    #             colored_mask = (colored_mask * 255).astype(np.uint8)

    #         # Alpha mask where mask > 0
    #         alpha = (mask > 0)[..., np.newaxis]

    #         # Blend grayscale and mask
    #         blended = image_rgb * (1 - alpha) + colored_mask * alpha
    #         blended = blended.astype(np.uint8)

    #         overlay_pil = Image.fromarray(blended)
    
    #         buf = io.BytesIO()
    #         overlay_pil.save(buf, format="PNG")
    #         buf.seek(0)
    
    #         pixmap_mask = QPixmap()
    #         pixmap_mask.loadFromData(buf.read(), "PNG")
    #         self.image_label.setPixmap(pixmap_mask.scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            
    #         # Set the model name
    #         if hasattr(self, 'models_view') and self.models_view.selected_model:
    #             self.model_name_label.setText(self.models_view.selected_model)
    
    #         QMessageBox.information(self, "Done", "Segmentation overlay displayed.")
    
    #     except Exception as e:
    #         import traceback
    #         print("❌ Error showing overlay:", traceback.format_exc())
    #         QMessageBox.critical(self, "Display Error", str(e))


    # (Methods remain unchanged: select_image_folder, select_model, run_training)
class RetrainModelView(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_dir = ""
        self.model_path = ""

        # Horizontal layout for folder selection
        folder_layout = QHBoxLayout()
        folder_layout.addWidget(QLabel("📁 Select Training Image Directory:"))
        self.image_btn = QPushButton("Choose Image Folder")
        self.image_btn.setFixedWidth(200)
        self.image_btn.clicked.connect(self.select_image_folder)
        folder_layout.addWidget(self.image_btn)
        folder_layout.addStretch()
        self.layout.addLayout(folder_layout)

        # Horizontal layout for model selection
        model_layout = QHBoxLayout()
        model_layout.addWidget(QLabel("🧠 Select Pretrained Model (.pth):"))
        self.model_btn = QPushButton("Choose Model")
        self.model_btn.setFixedWidth(200)
        self.model_btn.clicked.connect(self.select_model)
        model_layout.addWidget(self.model_btn)
        model_layout.addStretch()
        self.layout.addLayout(model_layout)

        # Horizontal layout for mask filter
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("🧪 Enter Mask Filter (e.g. _cp_masks):"))
        self.mask_filter = QLabel("_cp_masks")
        mask_layout.addWidget(self.mask_filter)
        mask_layout.addStretch()
        self.layout.addLayout(mask_layout)

        # Horizontal layout for training button
        train_layout = QHBoxLayout()
        self.train_btn = QPushButton("Start Retraining")
        self.train_btn.setFixedWidth(200)
        self.train_btn.clicked.connect(self.run_training)
        train_layout.addWidget(self.train_btn)
        train_layout.addStretch()
        self.layout.addLayout(train_layout)

        self.status = QLabel("")
        self.layout.addWidget(self.status)

    def select_image_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if folder:
            self.image_dir = folder
            self.status.setText(f"✅ Selected folder: {folder}")

    def select_model(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select Model", "", "Models (*.pth)")
        if file_path:
            self.model_path = file_path
            self.status.setText(f"✅ Model selected: {file_path}")

    def run_training(self):
        if not self.image_dir or not self.model_path:
            QMessageBox.warning(self, "Missing Input", "Please select both image folder and model.")
            return
    
        model_name = os.path.basename(self.model_path)
    
        # if model_name == "Retrain_omni_4.pth":
        if "omni" in model_key.lower():
            command = [
                "omnipose", "--train", "--use_gpu",
                "--dir", self.image_dir,
                "--mask_filter", "_cp_masks",
                "--pretrained_model", self.model_path,
                "--n_epochs", "10",
                "--learning_rate", "0.1",
                "--diameter", "0",
                "--nchan", "1",
                "--nclasses", "2",
                "--verbose"
            ]
        else:


            command = [
                "cellpose", "--train",
                "--dir", self.image_dir,
                "--pretrained_model", self.model_path,
                "--chan", "0",
                "--learning_rate", "0.1",
                "--weight_decay", "0.0001",
                "--n_epochs", "5",
                "--mask_filter", "_cp_masks",
                "--use_gpu",
                "--min_train_masks", "1",
                "--verbose"
            ]


        import subprocess
        
        try:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate()
        
            # Print both to console
            print("========== STDOUT ==========")
            print(stdout.decode())
            print("========== STDERR ==========")
            print(stderr.decode())
        
            # Save logs to file (optional for debugging)
            with open("cellpose_training_stdout.txt", "w") as f:
                f.write(stdout.decode())
            with open("cellpose_training_stderr.txt", "w") as f:
                f.write(stderr.decode())
        
            saved_model = os.path.join(self.image_dir, "models", "cellpose_train_0.npy")
            if os.path.exists(saved_model):
                self.status.setText(f"✅ Model saved:\n{saved_model}")
            else:
                self.status.setText("⚠️ Training finished, but model was not saved.\nCheck logs.")
        
        except Exception as e:
            self.status.setText(f"❌ Error: {str(e)}")

class TrackingView(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)
        
        self.mask_dir = ""
        self.output_dir = ""
        
        # Title
        title = QLabel("Filament Tracking")
        title.setStyleSheet("font-size: 18px; font-weight: bold; margin: 10px 0;")
        self.layout.addWidget(title)
        
        # Horizontal layout for mask folder selection
        mask_layout = QHBoxLayout()
        mask_layout.addWidget(QLabel("Select Mask Folder:"))
        self.mask_btn = QPushButton("Choose Mask Folder")
        self.mask_btn.setFixedWidth(200)
        self.mask_btn.clicked.connect(self.select_mask_folder)
        mask_layout.addWidget(self.mask_btn)
        mask_layout.addStretch()
        self.layout.addLayout(mask_layout)
        
        # Horizontal layout for output folder selection
        output_layout = QHBoxLayout()
        output_layout.addWidget(QLabel("Select Output Folder:"))
        self.output_btn = QPushButton("Choose Output Folder")
        self.output_btn.setFixedWidth(200)
        self.output_btn.clicked.connect(self.select_output_folder)
        output_layout.addWidget(self.output_btn)
        output_layout.addStretch()
        self.layout.addLayout(output_layout)
        
        # Horizontal layout for tracking button
        track_layout = QHBoxLayout()
        self.track_btn = QPushButton("Start Tracking")
        self.track_btn.setFixedWidth(200)
        self.track_btn.clicked.connect(self.run_tracking)
        track_layout.addWidget(self.track_btn)
        track_layout.addStretch()
        self.layout.addLayout(track_layout)
        
        # Status label
        self.status = QLabel("Select mask and output folders to begin tracking.")
        self.status.setStyleSheet("font-size: 12px; color: #666; margin-top: 10px;")
        self.layout.addWidget(self.status)
        
        # Add stretch to push everything to the top
        self.layout.addStretch()

        # Image display area
        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setMinimumSize(600, 400)
        self.image_display.setStyleSheet("border: 1px solid #ddd; background-color: #f9f9f9;")
        self.image_display.setText("Tracking visualizations will appear here")
        self.layout.addWidget(self.image_display)

        # Navigation for multiple plots
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.prev_plot_btn = QPushButton("← Previous Plot")
        self.prev_plot_btn.setFixedWidth(120)
        self.prev_plot_btn.clicked.connect(self.show_prev_plot)
        nav_layout.addWidget(self.prev_plot_btn)

        nav_layout.addSpacing(20)

        self.next_plot_btn = QPushButton("Next Plot →")
        self.next_plot_btn.setFixedWidth(120)
        self.next_plot_btn.clicked.connect(self.show_next_plot)
        nav_layout.addWidget(self.next_plot_btn)

        self.layout.addLayout(nav_layout)

        # Initially disable navigation buttons
        self.prev_plot_btn.setEnabled(False)
        self.next_plot_btn.setEnabled(False)

        # Plot navigation properties
        self.current_plot_index = 0
        self.plot_pixmaps = []
        self.plot_titles = []
    
    def select_mask_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Mask Directory")
        if folder:
            self.mask_dir = folder
            self.status.setText(f"Mask folder selected: {os.path.basename(folder)}")
    
    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Directory") 
        if folder:
            self.output_dir = folder
            self.status.setText(f"Output folder selected: {os.path.basename(folder)}")
        

    def show_prev_plot(self):
        if self.plot_pixmaps:
            self.current_plot_index = (self.current_plot_index - 1) % len(self.plot_pixmaps)
            self.display_current_plot()

    def show_next_plot(self):
        if self.plot_pixmaps:
            self.current_plot_index = (self.current_plot_index + 1) % len(self.plot_pixmaps)
            self.display_current_plot()

    def display_current_plot(self):
        if self.plot_pixmaps:
            pixmap = self.plot_pixmaps[self.current_plot_index]
            title = self.plot_titles[self.current_plot_index]
            self.image_display.setPixmap(pixmap.scaled(600, 400, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.status.setText(f"Showing: {title} ({self.current_plot_index + 1}/{len(self.plot_pixmaps)})")
    
    # Instead of plt.show(), capture the plot
    def capture_plot(self,title):
        import io
        from PIL import Image
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Convert to QPixmap
        pil_image = Image.open(buf)
        buf2 = io.BytesIO()
        pil_image.save(buf2, format='PNG')
        buf2.seek(0)
        
        pixmap = QPixmap()
        pixmap.loadFromData(buf2.read(), "PNG")
        
        self.plot_pixmaps.append(pixmap)
        self.plot_titles.append(title)
        
        plt.close()  # Close the figure to free memory
        
    def run_tracking(self):

        if not self.mask_dir or not self.output_dir:
            QMessageBox.warning(self, "Missing Input", "Please select both mask folder and output folder.")
            return
        # Define paths 
        # pos = 'Pos13_1_B' # name of folder, could be made into a loop
        # path = '/Users\oargell\Documents\Toy_datasets\Pos13_1_B/' # absolute path to the folder with the masks
        # sav_path = '/Users\oargell\Desktop/' # absolute path to the folder where teh results will be saved
        pos = os.path.basename(self.mask_dir)
        path = self.mask_dir + "/"  # absolute path to the folder with the masks
        sav_path = self.output_dir + "/"  # absolute path to the folder where teh results

        #Helper Functions: directly in the script for ease of checking

        # 1. Binarization function
        def binar(IS1):
            IS1B = IS1.copy()
            IS1B[IS1 != 0] = 1
            return IS1B

        # 2. Small segmentation artifact removal funtion
        def remove_artif(I2A,disk_size):  

            I2AA=np.copy(I2A) #   plt.imshow(IS2)
            # Applying logical operation and morphological opening
            I2A1 = binar(I2A);#binar(I2A) plt.imshow(I2A1)     plt.imshow(I2A)

            # Create a disk-shaped structuring element with radius 3
            selem = disk(disk_size)
            # Perform morphological opening
            I2B = opening(I2A1, selem)
            
            # Morphological dilation   plt.imshow(I2B)
            I2C = dilation(I2B, disk(disk_size))  # Adjust the disk size as needed

            I3 = I2AA * I2C # plt.imshow(I3)

            # Extract unique objects
            objs = np.unique(I3)
            objs = objs[1:len(objs)]
            
            # Initialize an image of zeros with the same size as I2A
            I4 = np.uint16(np.zeros((I3.shape[0], I3.shape[1])))
            # Mapping the original image values where they match the unique objects
            AZ=1
            for obj in objs:
                I4[I2A == obj] = AZ
                AZ=AZ+1
            
            return I4

        # 3. Reindexing function
        def OAM_23121_tp3(M, cel, no_obj1, A):
            tp3 = np.array(M)  # Ensure M is a numpy array
            tp3[tp3 == cel] = no_obj1 + A
            return tp3


        # Load ART masks into a tensor
        path_dir = [f for f in sorted(os.listdir(path)) if f.endswith('_ART_masks.tif')] # specify the mask ending, here: '_ART_masks.tif'
        Masks_Tensor = [imread(os.path.join(path, f)).astype(np.uint16) for f in path_dir]# plt.imshow(Mask3[0])
            
        im_no1 = 0
        im_no = len(Masks_Tensor)
        mm = range(im_no) # time points to track
        disk_size = 6 # Defines filter threshold for excluding segmentation artifacts due to interpolation, 3 if the objects are ~ 500 pixels, 6 if object are ~ 2000 pixels

        """
        Load the first mask that begins the indexing for all the cells; IS1 is updated to the most recently processed tracked mask at the end of it0
        """    
        IS1 = np.copy(Masks_Tensor[im_no1]).astype('uint16') # start tracking at first time point # plt.imshow(IS1)
        IS1 = remove_artif(IS1, disk_size) # remove artifacts and start tracking at first time point # plt.imshow(IS1) #  
        masks = np.zeros((IS1.shape[0], IS1.shape[1], im_no)) # contains the re-labeled masks according to labels in the last tp mask
        masks[:,:,im_no1] = IS1.copy() # first time point defines indexing; IS1 is first segmentation output


        IblankG = np.zeros(IS1.shape, dtype="uint16") # this matrix will contain cells with segmentation gaps, IblankG is updated within the loops 
        tic = time.time()

        # main tracking loop cells fomr IS1 (previous segmentation) are tracked onto IS2 (Next segmentation)

        for it0 in mm: # notice IS1 will be updated at each loop iteration
            print(f'it0={it0}')
            # Load the segmentation mask for the next frame that will be re-indexed at the end of the loop
            IS2 = np.copy(Masks_Tensor[it0]).astype('uint16') # plt.imshow(IS2)  
            IS2 = remove_artif(IS2, disk_size) # remove small segmentation artifacts to focus tracking on real objects 
            
            IS2C = np.copy(IS2) # plt.imshow(IS2C) # <--- a copy of IS2, gets updated in it1 loop
            IS1B = binar(IS1) # biarization of the previous frame
            
            IS3 = IS1B.astype('uint16') * IS2 # previous binary segmentation superimposed on the next segmentaion; updated in it1
            tr_cells = np.unique(IS1[IS1 != 0]) # the tracked cells present in the present mask, IS1
            
            gap_cells = np.unique(IblankG[IblankG != 0]) # the tracked cells that had a gap in their segmentation; were not detected in IS1, gets updated in it1 loop
            cells_tr = np.concatenate((tr_cells, gap_cells)) # all the cells that have been tracked up to this tp for this position

            # Allocate space for the re-indexed IS2 according to tracking
            Iblank0 = np.zeros_like(IS1)
            
            # Go to the previously tracked cells and find corresponding index in current tp being processed, IS2 becomes Iblank0
            
            if cells_tr.sum() != 0: # required in case the mask does not have anymore cells to track
                for it1 in np.sort(cells_tr): # cells are processed according to the index
                    IS5 = (IS1 == it1).copy() # binary image of previous mask containing only cell # it1
                    IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS3 # find the overlap area between the binary image of cell # it1 and IS3 (previous binary image superimposed on the next segmentation)

                    if IS5.sum() == 0: # if the cell was missing in the original previous mask; use the previous tracked mask instead IS2C
                        IS5 = (IblankG == it1).copy()
                        IS6A = np.uint16(thin(IS5, max_num_iter=1)) * IS2C 
                        IblankG[IblankG == it1] = 0 # remove the cell from the segmentation mask, updated for next iteration 

                
                    if IS6A.sum() != 0: # Find the tracked cell's corresponding index in IS2, update IS3 and IS2C 
                        IS2ind = 0 if not IS6A[IS6A != 0].any() else mode(IS6A[IS6A != 0])[0]
                        Iblank0[IS2 == IS2ind] = it1
                        IS3[IS3 == IS2ind] = 0
                        IS2C[IS2 == IS2ind] = 0

                # Define cells with segmentation gap, update IblankG, the segmentation gap mask
                seg_gap = np.setdiff1d(tr_cells, np.unique(Iblank0)) # cells in the past mask (IS1), that were not found in IS2 

                if seg_gap.size > 0:
                    for itG in seg_gap:
                        IblankG[IS1 == itG] = itG

                # Define cells that were not relabelled in IS2; these are the new born cells or cells entering the frame
                Iblank0B = Iblank0.copy()
                Iblank0B[Iblank0 != 0] = 1
                ISB = IS2 * np.uint16(1 - Iblank0B)
                
                # Add the new cells to Iblank0, Iblank0 becomes Iblank, the final tracked segmentation with all cells added
                newcells = np.unique(ISB[ISB != 0])
                Iblank = Iblank0.copy()
                A = 1

                if newcells.size > 0:
                    for it2 in newcells:
                        Iblank[IS2 == it2] = np.max(cells_tr) + A # create new index that hasn't been present in tracking
                        A += 1

                masks[:, :, it0] = np.uint16(Iblank).copy() #<---convert tracked mask to uint16 and store
                IS1 = masks[:, :, it0].copy() # IS1, past mask, is updated for next iteration of it0

            else:
                masks[:, :, it0] = IS2.copy()
                IS1 = IS2.copy()

        toc = time.time()
        print(f'Elapsed time is {toc - tic} seconds.')


        """

        Addtional corrections and visualization to confirm the cells are OK
        
        """


        """
        Vizualize all cell tracks as single heatmap "All Ob"
        """
        obj = np.unique(masks)
        no_obj = int(np.max(obj))
        im_no = masks.shape[2]
        all_ob = np.zeros((no_obj, im_no))

        tic = time.time()

        for ccell in range(1, no_obj + 1):
            Maa = (masks == ccell)

            for i in range(im_no):
                pix = np.sum(Maa[:, :, i])
                all_ob[ccell-1, i] = pix
                
                

        plt.figure()
        plt.imshow(all_ob, aspect='auto', cmap='viridis', interpolation="nearest")
        plt.title("all_obj")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        # plt.show()
        self.capture_plot("All Objects Heatmap")

        """
        Ensure all cells numbered 1-n if necessary: the tracking loop assigns a unique index to all cells, but they might not be continous 
        """

        im_no = masks.shape[2]
        # Find all unique non-zero cell identifiers across all time points
        ccell2 = np.unique(masks[masks != 0])
        # Initialize Mask2 with zeros of the same shape as masks
        Mask2 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

        # Process each unique cell ID
        for itt3 in range(len(ccell2)):  # cells
            pix3 = np.where(masks == ccell2[itt3])
            Mask2[pix3] = itt3 + 1  # ID starts from 1

        """
        Vizualize all cell tracks as single binary heatmap "All Ob1"
        """

        # Get cell presence
        Mask3 = Mask2.copy()
        numbM = im_no
        obj = np.unique(Mask3)
        no_obj1 = int(obj.max())
        A = 1

        tp_im = np.zeros((no_obj1, im_no))

        for cel in range(1, no_obj1+1):
            Ma = (Mask3 == cel)

            for ih in range(numbM):
                if Ma[:, :, ih].sum() != 0:
                    tp_im[cel-1, ih] = 1


        plt.figure()
        plt.imshow(tp_im, aspect='auto', interpolation="nearest")
        plt.title("Cell Presence Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        # plt.show()
        self.capture_plot("Cell Presence Heatmap")


        """
        Split Interrupted tracks into tracks if necessary: use this if gaps of a couple of frames are not desired 
        """

        tic = time.time()
        for cel in range(1, no_obj1+1):
            print(cel)
            tp_im2 = np.diff(tp_im[cel-1, :])
            tp1 = np.where(tp_im2 == 1)[0]
            tp2 = np.where(tp_im2 == -1)[0]
            maxp = (Mask3[:, :, numbM - 1] == cel).sum()

            if len(tp1) == 1 and len(tp2) == 1 and maxp != 0:  # has one interruption
                for itx in range(tp1[0], numbM):
                    tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                    Mask3[:, :, itx] = tp3.copy()
                no_obj1 += A
            
            elif len(tp1) == 1 and len(tp2) == 1 and maxp == 0:  # has one interruption
                pass
            
            elif len(tp1) == len(tp2) + 1 and maxp != 0:
                tp2 = np.append(tp2, numbM-1)

                for itb in range(1, len(tp1)):  # starts at 2 because the first cell index remains unchanged
                    for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                        tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                        Mask3[:, :, itx] = tp3.copy()
                    no_obj1 += A
            
            elif len(tp2) == 0 or len(tp1) == 0:  # it's a normal cell, it's born and stays until the end
                pass
            
            elif len(tp1) == len(tp2):
                if tp1[0] > tp2[0]:
                    tp2 = np.append(tp2, numbM-1) #check this throughly
                    for itb in range(len(tp1)):
                        for itx in range(tp1[itb]+1, tp2[itb + 1] + 1):
                            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A) #+1 here
                            Mask3[:, :, itx] = tp3.copy()    
                        no_obj1 += A
                elif tp1[0] < tp2[0]:
                    for itb in range(1, len(tp1)): 
                        for itx in range(tp1[itb] + 1, tp2[itb] + 1):  # Inclusive range
                            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                            Mask3[:, :, itx] = tp3.copy()
                        no_obj1 += A
                elif len(tp2) > 1:
                    for itb in range(1, len(tp1)):
                        for itx in range(tp1[itb] + 1, tp2[itb] + 1):
                            tp3 = OAM_23121_tp3(Mask3[:, :, itx], cel, no_obj1, A)
                            Mask3[:, :, itx] = tp3.copy()    
                        no_obj1 += A
        toc = time.time()
        print(f'Elapsed time is {toc - tic} seconds.')


        """
        Vizualize all cell tracks as single binary heatmap "tp_im"
        """
        numbM = im_no
        obj = np.unique(Mask3)

        # Get cell presence 2
        tp_im = np.zeros((int(max(obj)), im_no))

        for cel in range(1, int(max(obj)) + 1):
            Ma = (Mask3 == cel)

            for ih in range(numbM):
                if Ma[:, :, ih].sum() != 0:
                    tp_im[cel-1, ih] = 1


        plt.figure()
        plt.imshow(tp_im, aspect='auto', interpolation="nearest")
        plt.title("Cell Presence Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        # plt.show()
        self.capture_plot("Cell Presence Heatmap After Splitting")


        """
        Exclude tracks that are not continous during the whole experiment if necessary

        """
        cell_artifacts = np.zeros(tp_im.shape[0])

        for it05 in range(tp_im.shape[0]):
            arti = np.where(np.diff(tp_im[it05, :]) == -1)[0]  # Find artifacts in the time series

            if arti.size > 0:
                cell_artifacts[it05] = it05 + 1  # Mark cells with artifacts

        goodcells = np.setdiff1d(np.arange(1, tp_im.shape[0] + 1), cell_artifacts[cell_artifacts != 0])  # Identify good cells


        im_no = Mask3.shape[2]
        Mask4 = np.zeros((masks.shape[0], masks.shape[1], masks.shape[2]))

        for itt3 in range(goodcells.size):
            pix3 = np.where(Mask3 == goodcells[itt3])
            Mask4[pix3] = itt3 + 1  # IDs start from 1


        """
        Vizualize all cell tracks as single binary heatmap "tp_im2"
        """

        Mask5 = Mask4.copy()
        numbM = im_no
        obj = np.unique(Mask4)
        no_obj1 = int(obj.max())
        A = 1

        tp_im2 = np.zeros((no_obj1, im_no))

        for cel in range(1, no_obj1+1):
            Ma = (Mask5 == cel)

            for ih in range(numbM):
                if Ma[:, :, ih].sum() != 0:
                    tp_im2[cel-1, ih] = 1

        plt.figure()
        plt.imshow(tp_im2, aspect='auto', interpolation="nearest")
        plt.title("Cell Presence Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        # plt.show()
        self.capture_plot("Cell Presence Heatmap After Removing Artifacts")


        """
        Sort cell tracks according to their lentgh if necessary 
        """

        cell_exists0 = np.zeros((2, tp_im2.shape[0]))
        for itt2 in range(tp_im2.shape[0]):
            # Find indices of non-zero elements
            non_zero_indices = np.where(tp_im2[itt2, :] != 0)[0]
            
            # If there are non-zero elements, get first and last
            if non_zero_indices.size > 0:
                first_non_zero = non_zero_indices[0]
                last_non_zero = non_zero_indices[-1]
            else:
                first_non_zero = -1  # Or any placeholder value for rows without non-zero elements
                last_non_zero = -1   # Or any placeholder value for rows without non-zero elements
            
            cell_exists0[:, itt2] = [first_non_zero, last_non_zero]

        sortOrder = sorted(range(cell_exists0.shape[1]), key=lambda i: cell_exists0[0, i])

        cell_exists = cell_exists0[:, sortOrder]
        art_cell_exists = cell_exists

        Mask6 = np.zeros_like(Mask5)
            
        for itt3 in range(len(sortOrder)):
            pix3 = np.where(Mask5 == sortOrder[itt3] + 1)  # here
            Mask6[pix3] = itt3 + 1# reassign

        """
        Vizualize all cell tracks as single binary heatmap "tp_im3"
        """
        Mask7 = Mask6.copy()
        numbM = im_no
        obj = np.unique(Mask6)
        no_obj1 = int(obj.max())
        A = 1

        tic = time.time()
        tp_im3 = np.zeros((no_obj1, im_no))
        for cel in range(1, no_obj1 + 1):
            tp_im3[cel - 1, :] = ((Mask7 == cel).sum(axis=(0, 1)) != 0).astype(int)
        toc = time.time()
        print(f'Elapsed time is {toc - tic} seconds.')

        plt.figure()
        plt.imshow(tp_im3, aspect='auto', interpolation="nearest")
        plt.title("Cell Presence Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        # plt.show()    
        self.capture_plot("Final Cell Presence Heatmap After Sorting")


        """
        Calculate object size if necessary
        """

        # 
        obj = np.unique(Mask7)
        no_obj = int(np.max(obj))
        im_no = Mask7.shape[2]
        all_ob = np.zeros((no_obj, im_no))

        tic = time.time()
        for ccell in range(1, no_obj + 1):
            Maa = (Mask7 == ccell)

            for i in range(im_no):
                pix = np.sum(Maa[:, :, i])
                all_ob[ccell-1, i] = pix
        toc = time.time()
        print(f'Elapsed time is {toc - tic} seconds.')


        """
        Variables for saving with the tracked masks
        """
        Final_unique_objects = np.unique(Mask7)
        Final_number_objects = int(np.max(obj))
        Final_number_frames = Mask7.shape[2]
        Final_Object_Size = all_ob
        Final_tracked_tensor = Mask7

        plt.figure()
        plt.imshow(Final_Object_Size , aspect='auto', cmap='viridis',interpolation="nearest")
        plt.title("Cell Sizes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")


        """
        Save variables and tracks for MATLAB
        """
        sio.savemat(os.path.join(sav_path, f'{pos}_ART_Tracks_MATLAB.mat'), {
            "Final_unique_objects": Final_unique_objects,
            "Final_number_objects": Final_number_objects,
            "Final_number_frames": Final_number_frames,
            "Final_Object_Size": Final_Object_Size,
            "Final_tracked_tensor": Final_tracked_tensor,
        }, do_compression=True)



        """
        Save tracks as tensor and other variables as pickle in Python
        """

        Tracks_file = {"Final_unique_objects": Final_unique_objects,
            "Final_number_objects": Final_number_objects,
            "Final_number_frames": Final_number_frames,
            "Final_Object_Size": Final_Object_Size}

        name_save=os.path.join(sav_path, f'{pos}_Tracks_vars_file.pklt')

        with open(name_save, 'wb') as file:
            pickle.dump(Tracks_file, file)

        name_save2=os.path.join(sav_path, f'{pos}_Tracks')
        np.save(name_save2, Final_tracked_tensor)


        # check tracks are saved oK by Loading data from the file
        # with open('/Users\oargell\Desktop/Pos13_1_B_Tracks_file.pklt', 'rb') as file:
        #     loaded_data = pickle.load(file)

        # loaded_tensor = np.load('/Users\oargell\Desktop/Pos13_1_B_Tracks.npy')

        # Use dynamic paths based on user selection
        tracks_file_path = os.path.join(self.output_dir, f'{pos}_Tracks_vars_file.pklt')
        tracks_tensor_path = os.path.join(self.output_dir, f'{pos}_Tracks.npy')

        # Check if files were created successfully before trying to load them
        if os.path.exists(tracks_file_path) and os.path.exists(tracks_tensor_path):
            try:
                with open(tracks_file_path, 'rb') as file:
                    loaded_data = pickle.load(file)
                
                loaded_tensor = np.load(tracks_tensor_path)
                print("Files loaded successfully:")
                print(loaded_data)
                print(f"Tensor shape: {loaded_tensor.shape}")
                
                # Show final plot
                plt.figure()
                plt.imshow(loaded_tensor[:,:,0], aspect='auto', cmap='viridis', interpolation="nearest")
                plt.title("First Frame - Tracked Objects")
                plt.xlabel("X")
                plt.ylabel("Y")
                self.capture_plot("First Frame - Tracked Objects")
                
            except Exception as e:
                print(f"Error loading saved files: {e}")
        else:
            print("Warning: Saved files not found, but tracking completed.")
            print(loaded_data)
            print(loaded_tensor)


        plt.figure()
        plt.imshow(loaded_tensor[:,:,0] , aspect='auto', cmap='viridis',interpolation="nearest")
        plt.title("Cell Sizes Over Time")
        plt.xlabel("Time")
        plt.ylabel("Cells")
        self.capture_plot("Sample Tracked Mask Frame")

        # After tracking completes
        if self.plot_pixmaps:
            self.current_plot_index = 0
            self.display_current_plot()
            self.prev_plot_btn.setEnabled(True)
            self.next_plot_btn.setEnabled(True)
            
            QMessageBox.information(self, "Tracking Complete", 
                                f"Tracking completed successfully!\n"
                                f"Generated {len(self.plot_pixmaps)} visualization plots.\n"
                                f"Results saved to: {self.output_dir}")




class PlaceholderView(QWidget):
    def __init__(self, name):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel(f"{name} view (Coming soon...)"))
        self.setLayout(layout)

class FileView(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        layout.addWidget(QLabel("File operations will go here."))
        self.setLayout(layout)

class AddModelView(QWidget):
    def __init__(self, models_view):
        super().__init__()
        self.models_view = models_view

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # Horizontal layout for model upload
        upload_layout = QHBoxLayout()
        upload_layout.addWidget(QLabel("Upload your own model (.pth):"))
        self.upload_btn = QPushButton("Browse and Add Model")
        self.upload_btn.setFixedWidth(200)
        self.upload_btn.clicked.connect(self.upload_model)
        upload_layout.addWidget(self.upload_btn)
        upload_layout.addStretch()
        self.layout.addLayout(upload_layout)

        self.status = QLabel("")
        self.layout.addWidget(self.status)

    def upload_model(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Model File", "", "PyTorch Models (*.pth)"
        )
        if file_path:
            model_name = os.path.basename(file_path)
            if model_name in self.models_view.models:
                self.status.setText("⚠️ Model already exists.")
                return
            self.models_view.add_model(model_name, file_path)
            self.status.setText(f"✅ Added and selected: {model_name}")
        else:
            self.status.setText("⚠️ No model selected.")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation GUI")
        self.setGeometry(100, 100, 1200, 700)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # === Top Navigation Bar with ALL buttons ===
        nav_group = QGroupBox("Controls")
        nav_layout = QHBoxLayout()
        nav_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)  # Center the entire layout

        btn_style = """
        QPushButton {
            padding: 8px 8px;
            background-color: #f0f0f0;
            border: 1px solid #aaa;
            border-radius: 3px;
            min-width: 30px;
            min-height: 20px;
            margin: 2px;
            color: black;
            font-weight: bold;
            font-size: 18px;  
            alignment: center;          
        }
        QPushButton:hover {
            background-color: #ddd;
        }
        QPushButton:pressed {
            background-color: #ccc;
        }
        QPushButton:checked {
            background-color: #0078d4;
            color: white;
            border-color: #0078d4;
        }
        """

        # Add flexible spacing before buttons to center them
        nav_layout.addStretch(1)

        # View navigation buttons with icons
        models_btn = QPushButton("⚙")
        models_btn.setStyleSheet(btn_style)
        models_btn.setCheckable(True)
        models_btn.setToolTip("Models")
        nav_layout.addWidget(models_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        # Add flexible spacing between each button
        nav_layout.addStretch(1)

        add_model_btn = QPushButton("＋")
        add_model_btn.setStyleSheet(btn_style)
        add_model_btn.setCheckable(True)
        add_model_btn.setToolTip("Add Model")
        nav_layout.addWidget(add_model_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        nav_layout.addStretch(1)

        retrain_btn = QPushButton("↻")
        retrain_btn.setStyleSheet(btn_style)
        retrain_btn.setCheckable(True)
        retrain_btn.setToolTip("Retrain")
        nav_layout.addWidget(retrain_btn, 0, Qt.AlignmentFlag.AlignCenter)

        nav_layout.addStretch(1)

        tracking_btn = QPushButton("🔗")
        tracking_btn.setStyleSheet(btn_style)
        tracking_btn.setCheckable(True)
        tracking_btn.setToolTip("Track Filaments")
        nav_layout.addWidget(tracking_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        nav_layout.addStretch(1)

        image_btn = QPushButton("◯")
        image_btn.setStyleSheet(btn_style)
        image_btn.setCheckable(True)
        image_btn.setToolTip("Image")
        nav_layout.addWidget(image_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # Larger stretch for visual separator
        nav_layout.addStretch(2)

        # Image processing buttons with icons
        self.load_btn = QPushButton("📁")
        self.load_btn.setStyleSheet(btn_style)
        self.load_btn.setToolTip("Load Image")
        nav_layout.addWidget(self.load_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        nav_layout.addStretch(1)

        self.segment_single_btn = QPushButton("▶")
        self.segment_single_btn.setStyleSheet(btn_style)
        self.segment_single_btn.setToolTip("Run Single Segmentation")
        nav_layout.addWidget(self.segment_single_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        nav_layout.addStretch(1)

        self.compare_btn = QPushButton("⊞")
        self.compare_btn.setStyleSheet(btn_style)
        self.compare_btn.setToolTip("Compare Segmentations")
        nav_layout.addWidget(self.compare_btn, 0, Qt.AlignmentFlag.AlignCenter)
        
        nav_layout.addStretch(1)

        self.folder_seg_btn = QPushButton("₂")
        self.folder_seg_btn.setStyleSheet(btn_style)
        self.folder_seg_btn.setToolTip("Run Folder Segmentation")
        nav_layout.addWidget(self.folder_seg_btn, 0, Qt.AlignmentFlag.AlignCenter)

        # Add flexible spacing after buttons to center them
        nav_layout.addStretch(1)
        nav_group.setLayout(nav_layout)
        nav_group.setFixedHeight(80)

        # === Main Content Area ===
        self.stack = QStackedWidget()

        # Initialize views
        self.models_view = ModelsView()
        self.stack.addWidget(self.models_view)

        self.add_model_view = AddModelView(self.models_view)
        self.stack.addWidget(self.add_model_view)

        self.image_view = ImageView(self.models_view)
        self.stack.addWidget(self.image_view)

        self.retrain_view = RetrainModelView()
        self.stack.addWidget(self.retrain_view)

        self.tracking_view = TrackingView()
        self.stack.addWidget(self.tracking_view)

        # Navigation button connections
        self.nav_buttons = [models_btn, add_model_btn, retrain_btn, tracking_btn, image_btn]
        models_btn.clicked.connect(lambda: self.switch_view(self.models_view, models_btn))
        add_model_btn.clicked.connect(lambda: self.switch_view(self.add_model_view, add_model_btn))
        image_btn.clicked.connect(lambda: self.switch_view(self.image_view, image_btn))
        retrain_btn.clicked.connect(lambda: self.switch_view(self.retrain_view, retrain_btn))
        tracking_btn.clicked.connect(lambda: self.switch_view(self.tracking_view, tracking_btn))

        # Connect image processing buttons to image_view methods
        self.load_btn.clicked.connect(self.image_view.load_image)
        self.segment_single_btn.clicked.connect(self.image_view.run_single_image_segmentation)
        self.compare_btn.clicked.connect(self.image_view.compare_segmentations)
        self.folder_seg_btn.clicked.connect(self.image_view.run_cli_segmentation)

        # Set default view
        models_btn.setChecked(True)

        # === Assemble Layout ===
        main_layout.addWidget(nav_group)
        main_layout.addWidget(self.stack)

    def switch_view(self, view, button):
        # Uncheck all navigation buttons
        for btn in self.nav_buttons:
            btn.setChecked(False)
        # Check the clicked button
        button.setChecked(True)
        # Switch to the view
        self.stack.setCurrentWidget(view)

if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()