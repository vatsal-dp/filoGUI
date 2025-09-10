# -*- coding: utf-8 -*-
"""
Created on Wed Apr 23 16:48:25 2025
@author: ataasso
"""
import os
import subprocess

import sys
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QStackedWidget, QFileDialog, QMessageBox
)
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
import torch
torch.backends.mkldnn.enabled = False
import numpy as np

class ImageView(QWidget):
    def __init__(self, models_view):
        super().__init__()
        self.models_view = models_view

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        # === Display Area for Original and Masked Images ===
        self.image_row = QHBoxLayout()
        
        # === Left Column for Model Name and Original Image ===
        self.left_image_column = QVBoxLayout()
        
        # Model Name Label (above original image)
        self.model_name_label = QLabel("")
        self.model_name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.model_name_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        self.left_image_column.addWidget(self.model_name_label)
        
        # Original Image
        self.image_label_original = QLabel()
        self.image_label_original.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.left_image_column.addWidget(self.image_label_original)
        
        self.image_row.addLayout(self.left_image_column)
        
        # === Right: Segmentation Result ===
        self.image_label_masked = QLabel()
        self.image_label_masked.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_row.addWidget(self.image_label_masked)
        
        self.layout.addLayout(self.image_row)


        # === Section: Single Image Tools ===
        self.single_image_label = QLabel("🖼️ Single Image Preview")
        self.layout.addWidget(self.single_image_label)

        # Load Image Button
        self.load_btn = QPushButton("Load Image")
        self.load_btn.setFixedWidth(200)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.load_btn)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)
        self.load_btn.clicked.connect(self.load_image)

        # Run Single Image Segmentation Button
        self.segment_single_btn = QPushButton("Run Single Image Segmentation")
        self.segment_single_btn.setFixedWidth(200)
        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.segment_single_btn)
        btn_layout2.addStretch()
        self.layout.addLayout(btn_layout2)
        self.segment_single_btn.clicked.connect(self.run_single_image_segmentation)

        # Compare Segmentations Button
        self.compare_btn = QPushButton("Compare Segmentations")
        self.compare_btn.setFixedWidth(200)
        btn_layout3 = QHBoxLayout()
        btn_layout3.addWidget(self.compare_btn)
        btn_layout3.addStretch()
        self.layout.addLayout(btn_layout3)
        self.compare_btn.clicked.connect(self.compare_segmentations)

        self.layout.addSpacing(10)

        # Overlay navigation
        self.overlay_index = 0
        self.overlay_pixmaps = []
        
        nav_layout = QHBoxLayout()
        self.prev_btn = QPushButton("⬅️ Previous")
        self.prev_btn.setFixedWidth(100)
        self.prev_btn.clicked.connect(self.show_prev_overlay)
        nav_layout.addWidget(self.prev_btn)

        self.next_btn = QPushButton("➡️ Next")
        self.next_btn.setFixedWidth(100)
        self.next_btn.clicked.connect(self.show_next_overlay)
        nav_layout.addWidget(self.next_btn)

        nav_layout.addStretch()
        self.layout.addLayout(nav_layout)

        self.prev_btn.setEnabled(False)
        self.next_btn.setEnabled(False)

        self.layout.addSpacing(20)

        # === Section: Folder-Based Batch Segmentation ===
        self.batch_label = QLabel("📂 Folder-Based Segmentation")
        self.layout.addWidget(self.batch_label)

        self.folder_seg_btn = QPushButton("Run Folder Segmentation")
        self.folder_seg_btn.setFixedWidth(200)
        btn_layout4 = QHBoxLayout()
        btn_layout4.addWidget(self.folder_seg_btn)
        btn_layout4.addStretch()
        self.layout.addLayout(btn_layout4)
        self.folder_seg_btn.clicked.connect(self.run_cli_segmentation)


    def show_next_overlay(self):
        if not self.overlay_pixmaps:
            return
        self.overlay_index = (self.overlay_index + 1) % len(self.overlay_pixmaps)
        self.image_label_masked.setPixmap(self.overlay_pixmaps[self.overlay_index].scaled(400, 400, Qt.KeepAspectRatio))
        self.model_name_label.setText(self.overlay_model_names[self.overlay_index])

    def show_prev_overlay(self):
        if not self.overlay_pixmaps:
            return
        self.overlay_index = (self.overlay_index - 1) % len(self.overlay_pixmaps)
        self.image_label_masked.setPixmap(self.overlay_pixmaps[self.overlay_index].scaled(400, 400, Qt.KeepAspectRatio))
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
    
    

    
    
    
    
    
        model_keys = ["FilaBranch", "FilaTip_6", "Retrain_omni_5","Coni_4","ConidioBUD","FilaCross_1","FilaSeptum_3"]
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
                self.image_label_masked.setPixmap(self.overlay_pixmaps[0].scaled(400, 400, Qt.KeepAspectRatio))
                self.model_name_label.setText(self.overlay_model_names[0])  # Show first model name
    
                self.prev_btn.setEnabled(True)
                self.next_btn.setEnabled(True)
    
                self.image_label_original.clear()
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
                f'--cellprob_threshold 0.0 --flow_threshold 0.4 '  # Much more sensitive thresholds
                f'--diameter 0 '  # Let cellpose estimate diameter
                f'--save_tif --savedir "{save_dir}" '
                f'--img_filter "{image_basename}"'
            )
            # command = (
            #     f'cellpose --dir "{image_folder}" '
            #     f'--pretrained_model "{model_path}" --verbose --chan 0 --chan2 0 --use_gpu '
            #     f'--cellprob_threshold 0.5 --flow_threshold 0.9  --save_tif --savedir "{save_dir}" '
            #     f'--img_filter "{image_basename}"'
            # )
    
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

    def run_cli_segmentation(self):
        import subprocess, os
        from PySide6.QtCore import QTimer
        from PySide6.QtWidgets import QFileDialog, QMessageBox
    
        # STEP 1: Ask user to pick a folder
        from PySide6.QtWidgets import QFileDialog
        
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Folder Containing Time Series Images",
            "",
            QFileDialog.Option(QFileDialog.DontUseNativeDialog)
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
    
        # STEP 3: Create output folder named after the model
        self.save_dir = os.path.join(self.image_dir, model_key)
        os.makedirs(self.save_dir, exist_ok=True)

        import cv2, tifffile

        scaled_dir = os.path.join(self.image_dir, "scaled_inputs")
        os.makedirs(scaled_dir, exist_ok=True)

        scaled_files = []
        for fname in image_files:
            img_path = os.path.join(self.image_dir, fname)
            img = tifffile.imread(img_path)
            if img.ndim == 3:
                img = img[0]
            new_h, new_w = int(img.shape[0] * 1), int(img.shape[1] * 1)
            img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

            out_path = os.path.join(scaled_dir, fname)
            tifffile.imwrite(out_path, img_resized)
            scaled_files.append(out_path)

        # ✅ Point CLI command to scaled_dir instead of original self.image_dir
        self.image_dir = scaled_dir

    
        # STEP 4: Build command
        if "omni" in model_key.lower():
            command = (
                f'omnipose --dir "{self.image_dir}" '
                f'--pretrained_model "{model_path}" --use_gpu --nchan 1 --nclasses 2 '
                f'--verbose --save_tif --save_dtype uint16 '
                f'--savedir "{self.save_dir}" '
                f'--look_one_level_down --no_npy'
            )

        else:
            command = (
                f'cellpose --dir "{self.image_dir}" '
                f'--pretrained_model "{model_path}" --verbose --use_gpu --chan 0 --chan2 0 '
                f'--cellprob_threshold 0.0 --flow_threshold 0.4 '  # More sensitive
                f'--diameter 0 '  # Let cellpose estimate
                f'--save_tif --savedir "{self.save_dir}"'
            )
            # command = (
            #     f'cellpose --dir "{self.image_dir}" '
            #     f'--pretrained_model "{model_path}" --verbose --use_gpu --chan 0 --chan2 0 --cellprob_threshold 0.5 --flow_threshold 0.9 '
            #     f'  --save_tif --savedir "{self.save_dir}"'
            # )
    
        # STEP 5: Run the command
        try:
            subprocess.Popen(command, shell=True)
            QMessageBox.information(self, "Running", f"Segmentation started with model '{model_key}' on {len(image_files)} image(s).")
            self.check_timer = QTimer(self)
            self.check_timer.timeout.connect(self.try_show_mask_from_save)
            self.check_timer.start(5000)
        except Exception as e:
            QMessageBox.critical(self, "Error", str(e))







    def try_show_mask_from_save(self):
        import os, io
        import tifffile
        import numpy as np
        from PIL import Image
        import matplotlib.pyplot as plt
    
        print(f"📁 Checking folder: {self.save_dir}")
        if not os.path.exists(self.save_dir):
            print("❌ Save folder not created yet.")
            return
    
        mask_files = [f for f in os.listdir(self.save_dir) if f.endswith("_cp_masks.tif")]
        if not mask_files:
            print("⌛ No masks found yet.")
            return
    
        mask_file = mask_files[0]
        base_name = mask_file.replace("_cp_masks.tif", ".tif")
    
        mask_path = os.path.join(self.save_dir, mask_file)
        image_path = os.path.join(self.image_dir, base_name)
    
        if not os.path.exists(image_path):
            print(f"⚠️ Original image not found: {image_path}")
            return
    
        print(f"✅ Found mask: {mask_path}")
        print(f"✅ Found image: {image_path}")
        self.check_timer.stop()
    
        try:
            image = tifffile.imread(image_path)
            mask = tifffile.imread(mask_path)

            print(f"🔍 DEBUG - Original image shape: {image.shape}, dtype: {image.dtype}")
            print(f"🔍 DEBUG - Original image min/max: {image.min()}/{image.max()}")
            print(f"🔍 DEBUG - Original mask shape: {mask.shape}, dtype: {mask.dtype}")
            print(f"🔍 DEBUG - Original mask min/max: {mask.min()}/{mask.max()}")
            print(f"🔍 DEBUG - Unique mask values: {np.unique(mask)}")
    
            if image.ndim == 3:
                image = image[0]
            if mask.ndim == 3:
                mask = mask[0]

            if image.ndim == 3:
                print(f"🔍 DEBUG - Image reduced to 2D: {image.shape}")
            if mask.ndim == 3:
                print(f"🔍 DEBUG - Mask reduced to 2D: {mask.shape}")
    
            # === Process original image for consistent display ===
            img_min, img_max = np.percentile(image, (1, 99))
            img_norm = np.clip((image - img_min) / (img_max - img_min), 0, 1)
            img_uint8 = (img_norm * 255).astype(np.uint8)

            print(f"🔍 DEBUG - Image after normalization shape: {img_uint8.shape}, dtype: {img_uint8.dtype}")
            print(f"🔍 DEBUG - Normalized image min/max: {img_uint8.min()}/{img_uint8.max()}")  
    
            original_pil = Image.fromarray(img_uint8)
            buf_orig = io.BytesIO()
            original_pil.save(buf_orig, format="PNG")
            buf_orig.seek(0)
    
            pixmap_orig = QPixmap()
            pixmap_orig.loadFromData(buf_orig.read(), "PNG")
            self.image_label_original.setPixmap(pixmap_orig.scaled(400, 400, Qt.KeepAspectRatio))

            print(f"🔍 DEBUG - Image for overlay shape: {image.shape}")
            print(f"🔍 DEBUG - Mask for overlay shape: {mask.shape}")
            print(f"🔍 DEBUG - Mask for overlay unique values: {np.unique(mask)}")
            print(f"🔍 DEBUG - Non-zero mask pixels count: {np.count_nonzero(mask)}")
    
            # === Process mask overlay image using matplotlib ===
            # fig, ax = plt.subplots(figsize=(5, 5), facecolor='black')
            # ax.set_facecolor('black')
            # ax.imshow(image, cmap='gray', interpolation='nearest')
            # ax.imshow(mask, alpha=1, cmap='jet', interpolation='nearest')
            # ax.axis('off')

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
            # plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
            # plt.close()
            overlay_pil.save(buf, format="PNG")
            buf.seek(0)
    
            pixmap_mask = QPixmap()
            pixmap_mask.loadFromData(buf.read(), "PNG")
            self.image_label_masked.setPixmap(pixmap_mask.scaled(400, 400, Qt.KeepAspectRatio))
    
            QMessageBox.information(self, "Done", "Segmentation overlay displayed.")
    
        except Exception as e:
            import traceback
            print("❌ Error showing overlay:", traceback.format_exc())
            QMessageBox.critical(self, "Display Error", str(e))



class RetrainModelView(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.image_dir = ""
        self.model_path = ""

        self.layout.addWidget(QLabel("📁 Select Training Image Directory"))

        self.image_btn = QPushButton("Choose Image Folder")
        self.image_btn.setFixedWidth(200)
        btn_layout1 = QHBoxLayout()
        btn_layout1.addWidget(self.image_btn)
        btn_layout1.addStretch()
        self.layout.addLayout(btn_layout1)
        self.image_btn.clicked.connect(self.select_image_folder)

        self.layout.addWidget(QLabel("🧠 Select Pretrained Model (.pth)"))

        self.model_btn = QPushButton("Choose Model")
        self.model_btn.setFixedWidth(200)
        btn_layout2 = QHBoxLayout()
        btn_layout2.addWidget(self.model_btn)
        btn_layout2.addStretch()
        self.layout.addLayout(btn_layout2)
        self.model_btn.clicked.connect(self.select_model)

        self.layout.addWidget(QLabel("🧪 Enter Mask Filter (e.g. _cp_masks)"))

        self.mask_filter = QLabel("_cp_masks")
        self.layout.addWidget(self.mask_filter)

        self.train_btn = QPushButton("Start Retraining")
        self.train_btn.setFixedWidth(200)
        btn_layout3 = QHBoxLayout()
        btn_layout3.addWidget(self.train_btn)
        btn_layout3.addStretch()
        self.layout.addLayout(btn_layout3)
        self.train_btn.clicked.connect(self.run_training)

        self.status = QLabel("")
        self.layout.addWidget(self.status)

    # (Methods remain unchanged: select_image_folder, select_model, run_training)


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

        self.label = QLabel("Upload your own model (.pth)")
        self.layout.addWidget(self.label)

        self.upload_btn = QPushButton("Browse and Add Model")
        self.upload_btn.setFixedWidth(200)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(self.upload_btn)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)

        self.status = QLabel("")
        self.layout.addWidget(self.status)

        self.upload_btn.clicked.connect(self.upload_model)

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


 



class ModelsView(QWidget):
    def __init__(self):
        super().__init__()




        self.models = {
            "Coni_4": os.path.join("FiloGUI_models", "Coni_4.pth"),
            "ConidioBUD": os.path.join("FiloGUI_models", "ConidioBUD.pth"),
            "FilaBranch": os.path.join("FiloGUI_models", "FilaBranch.pth"),
            "FilaCross_1": os.path.join("FiloGUI_models", "FilaCross_1.pth"),
            "FilaSeptum_3": os.path.join("FiloGUI_models", "FilaSeptum_3.pth"),
            "FilaTip_6": os.path.join("FiloGUI_models", "FilaTip_6.pth"),
            "Retrain_omni_5": os.path.join("FiloGUI_models", "Retrain_omni_5.pth")
        }


      

        self.selected_model = None
        self.layout = QVBoxLayout()
        self.setLayout(self.layout)

        self.label = QLabel("Select a model:")
        self.layout.addWidget(self.label)

        self.buttons = {}
        for name in self.models:
            self._add_model_button(name)

        # self.status = QLabel("No model selected.")
        # self.layout.addWidget(self.status)

    def _add_model_button(self, model_name):
        btn = QPushButton(model_name)
        btn.setFixedWidth(200)
        btn_layout = QHBoxLayout()
        btn_layout.addWidget(btn)
        btn_layout.addStretch()
        self.layout.addLayout(btn_layout)
        btn.clicked.connect(lambda checked, n=model_name: self.select_model(n))
        self.buttons[model_name] = btn

    def select_model(self, model_name):
        self.selected_model = model_name
        self.status.setText(f"Selected: {model_name}\nPath: {self.models[model_name]}")

    def add_model(self, model_name, path):
        self.models[model_name] = path
        self._add_model_button(model_name)
        self.select_model(model_name)



from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QStackedWidget, QGroupBox
)
from PySide6.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmentation GUI")
        self.setGeometry(100, 100, 900, 600)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # === Left Side Menu with GroupBox ===
        side_group = QGroupBox("Navigation")
        side_layout = QVBoxLayout()

        btn_style = """
        QPushButton {
            padding: 5px;
            background-color: #f0f0f0;
            border: 1px solid #aaa;
            border-radius: 3px;
        }
        QPushButton:hover {
            background-color: #ddd;
        }
        """

        models_btn = QPushButton("Models")
        models_btn.setFixedWidth(130)
        models_btn.setStyleSheet(btn_style)
        side_layout.addWidget(models_btn)

        add_model_btn = QPushButton("Add Model")
        add_model_btn.setFixedWidth(130)
        add_model_btn.setStyleSheet(btn_style)
        side_layout.addWidget(add_model_btn)

        retrain_btn = QPushButton("Retrain")
        retrain_btn.setFixedWidth(130)
        retrain_btn.setStyleSheet(btn_style)
        side_layout.addWidget(retrain_btn)

        image_btn = QPushButton("Image")
        image_btn.setFixedWidth(130)
        image_btn.setStyleSheet(btn_style)
        side_layout.addWidget(image_btn)

        side_layout.addStretch()
        side_group.setLayout(side_layout)
        side_group.setFixedWidth(150)

        # === Main Content Area ===
        self.stack = QStackedWidget()

        # Models View
        self.models_view = ModelsView()
        self.stack.addWidget(self.models_view)
        models_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.models_view))

        # Add Model View
        self.add_model_view = AddModelView(self.models_view)
        self.stack.addWidget(self.add_model_view)
        add_model_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.add_model_view))

        # Image View
        self.image_view = ImageView(self.models_view)
        self.stack.addWidget(self.image_view)
        image_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.image_view))

        # Retrain View
        self.retrain_view = RetrainModelView()
        self.stack.addWidget(self.retrain_view)
        retrain_btn.clicked.connect(lambda: self.stack.setCurrentWidget(self.retrain_view))

        # === Assemble Layout ===
        main_layout.addWidget(side_group)
        main_layout.addWidget(self.stack)



if __name__ == "__main__":
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    app.exec()