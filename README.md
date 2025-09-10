# 👬 FiloSegGUI — A Deep Learning GUI for Fungal Cell Segmentation

**FiloSegGUI** is a PySide6-powered graphical user interface for applying and retraining deep learning-based segmentation models (Omnipose and Cellpose) on fungal microscopy images.

### ✨ Key Features

* ✅ Single image and batch segmentation
* 🔁 Retraining with custom image/mask datasets
* 🔄 Easy toggling between pretrained models
* 🖼️ Overlay visualization of segmentation results

---

## 📁 Project Structure

```
FiloSegGUI/
├── FiloGUI_models/          # Pretrained .pth models go here
├── main.py                  # The GUI application
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # (Optional) Ignore cache/files
```

---

## 🔧 Installation Guide

### 📦 Step 1: Clone the Repository

```bash
git clone https://github.com/Arsalancr7/FiloSegGUI.git
cd FiloSegGUI
```

### 🐍 Step 2: Create a Conda Environment (Recommended)

Use **Python 3.8.5**, which is the most compatible version for Omnipose + GUI.

```bash
conda create -n filoseg python=3.10.12
conda activate filoseg
```

### 🔥 Step 3: Install PyTorch

If you have an **NVIDIA GPU**:

```bash
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
```

If you're using **CPU only**:

```bash
conda install pytorch torchvision cpuonly -c pytorch
```

### ✅ Step 4: Install Python Requirements

```bash
pip install -r requirements.txt
```

This installs:

* Omnipose (older stable commit)
* PySide6, tifffile, matplotlib
* numpy, pillow, natsort

---

## 🚀 Run the GUI

```bash
python main.py
```

---

## 📂 Pretrained Models

All pretrained `.pth` models must be stored in the `FiloGUI_models/` directory:

* `Coni_4.pth`
* `FilaTip_6.pth`
* `Retrain_omni_4.pth`
* *(and others)*

### Example Model Reference in Code

```python
"./FiloGUI_models/Coni_4.pth"
```

> 💡 **Tip**: Keep `.pth` files under 100MB or use [GitHub Releases](https://docs.github.com/en/repositories/releasing-projects-on-github/about-releases) or Google Drive links to host larger files.

---

## 🧪 Retraining Models

To retrain a model:

1. Open the **"Retrain"** tab.
2. Select a directory containing images and masks.
3. Select a pretrained `.pth` model.
4. Start training — the new model will be saved in the `models/` subdirectory.

---

## 🧠 Dependencies

* Python 3.8.5
* PyTorch 2.5.1
* Omnipose (commit [`04b09e7`](https://github.com/kevinjohncutler/omnipose/commit/04b09e7))
* PySide6 6.9.0
* matplotlib, tifffile, pillow, numpy, natsort

---

## 🛠 Troubleshooting

If you see:

```text
cellpose not found. Error: The 'cellpose' distribution was not found
```

Install Cellpose manually:

```bash
pip install cellpose==3.0.8
```

---

## 📄 License

MIT License © 2025 Arsalan Taassob

---

## 🤝 Acknowledgements

* [Cellpose](https://github.com/MouseLand/cellpose)
* [Omnipose](https://github.com/kevinjohncutler/omnipose)
