# CodeAlpha_object_tracker
# 🎯 Real-time Object Detection & Tracking
### CodeAlpha AI Internship — Task 4

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?style=flat-square)
![Streamlit](https://img.shields.io/badge/Streamlit-Live-red?style=flat-square&logo=streamlit)
![OpenCV](https://img.shields.io/badge/OpenCV-4.8-green?style=flat-square&logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A production-ready, real-time object detection and tracking system built with **YOLOv8** and a custom **Centroid Tracker** — deployed live on Streamlit Cloud.

---

## 🚀 Live Demo

🔗 **[Try it live →]([https://objecttracker-byshahab.streamlit.app/])**

---

## 📸 Preview

> Upload any video or connect your webcam — the app detects and tracks every object in real time with motion trails, confidence scores, and live FPS stats.

---

## ✨ Features

| Feature | Description |
|---|---|
| ⚡ **YOLOv8 Detection** | Real-time object detection on every frame |
| 🎯 **Centroid Tracker** | Smooth multi-object tracking with unique IDs |
| 🌊 **Motion Trails** | Visual path history for every tracked object |
| 📊 **Live HUD** | FPS counter, object count, frame index overlay |
| 🎛️ **Full Controls** | Confidence, IOU threshold, inference size, frame skip |
| 📥 **Dual Input** | Video file upload (.mp4, .avi, .mov, .mkv) + Webcam |
| 💾 **Save Output** | Download the fully tracked video |
| 🖥️ **Dual Display** | Streamlit UI + local OpenCV window (optional) |

---

## 🧠 How It Works

```
Video Frame
    │
    ▼
YOLOv8 Inference  ──────►  Bounding Boxes + Labels + Confidence
    │
    ▼
Centroid Tracker  ──────►  Assigns unique ID to each object
    │
    ▼
Draw Detections   ──────►  Boxes + Labels + Motion Trails + HUD
    │
    ▼
Streamlit Display ──────►  Live JPEG stream in browser
```

**NLP Stack:**
- **Detection:** YOLOv8n / YOLOv8s / YOLOv8m (selectable)
- **Tracking:** Custom vectorized Centroid Tracker (NumPy-based, no extra deps)
- **Encoding:** JPEG (quality 82) for 3x faster frame streaming vs PNG

---

## ⚙️ Installation

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/codealpha_object_tracker.git
cd codealpha_object_tracker
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
# If streamlit is in PATH
streamlit run object_tracker_app.py

# On Windows (if PATH issue)
python -m streamlit run object_tracker_app.py
```

---

## 📁 Project Structure

```
codealpha_object_tracker/
│
├── object_tracker_app.py   # Main Streamlit application
├── requirements.txt        # Python dependencies
├── packages.txt            # System dependencies (for Streamlit Cloud)
├── runtime.txt             # Python version lock
├── .python-version         # Python version (3.11)
└── README.md               # This file
```

---

## 🎛️ Configuration Options

| Setting | Range | Recommended | Effect |
|---|---|---|---|
| **Model Size** | n / s / m | `yolov8n` | Faster vs accurate |
| **Inference Size** | 320 – 640 px | `416` | Lower = faster FPS |
| **Confidence** | 0.1 – 0.95 | `0.45` | Detection sensitivity |
| **IOU Threshold** | 0.1 – 0.9 | `0.45` | Overlap suppression |
| **Frame Skip** | 1 – 4 | `1` or `2` | Skip frames for speed |
| **Trail Length** | 5 – 60 | `25` | Motion trail history |

### ⚡ FPS Optimization Tips

- Use `yolov8n.pt` (nano model) — fastest inference
- Set **Inference Size = 320** for maximum speed
- Enable **Frame Skip = 2** — reuses last detection, zero inference cost
- **Target: 50–60 FPS** on CPU with above settings

---

## 📦 Requirements

```txt
torch
torchvision
ultralytics==8.3.0
opencv-python-headless
streamlit>=1.28.0
numpy
```

> **Note:** For local use, replace `opencv-python-headless` with `opencv-python` to enable the OpenCV window feature.

---

## 🌐 Deployment (Streamlit Cloud)

This app is configured for Streamlit Cloud with:

- `runtime.txt` → Python 3.11
- `packages.txt` → System libraries (`libgl1`, `libglib2.0-0t64`)
- `requirements.txt` → Python packages

> ⚠️ **Webcam** and **OpenCV window** features are only available on local machines, not on Streamlit Cloud.

---

## 🧑‍💻 Intern Info

| Field | Detail |
|---|---|
| **Intern** | Shahab Ullah Khattak |
| **Organization** | CodeAlpha |
| **Task** | 4 — Object Detection & Tracking |
| **Tech Stack** | Python, YOLOv8, OpenCV, Streamlit, NumPy |

---

## 📄 License

This project is licensed under the MIT License.

---

<p align="center">Built with ❤️ during AI Internship at <b>CodeAlpha</b></p>
