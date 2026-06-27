# Real-Time Object Trajectory Segmentation using YOLOv8 Oriented Bounding Boxes (OBB)

A real-time computer vision pipeline for object trajectory tracking and automatic temporal segmentation using a custom **YOLOv8 Oriented Bounding Box (OBB)** model.

> **Note:** This repository contains only the inference pipeline. The training dataset and trained weights are **not included** because they are proprietary.

---

# Features

- Real-time object detection using YOLOv8 OBB
- Oriented Bounding Box (OBB) tracking
- Geometric center extraction
- EMA trajectory filtering
- Savitzky–Golay smoothing
- Automatic motion start detection
- Bend-angle based temporal segmentation
- Live visualization
- CSV, JSON, graph and video export

---

# Repository Structure

```text
.
├── mop_tracker.py
├── README.md
├── requirements.txt
├── .gitignore
├── sample_results/
└── best.pt (not included)
```

# Requirements

```bash
pip install ultralytics opencv-python numpy scipy matplotlib
```

# Model

The project uses a custom-trained **YOLOv8 OBB** model.

The trained weights (`best.pt`) are **not included**.

Place `best.pt` beside the script or set:

macOS / Linux

```bash
export MODEL_PATH=/path/to/best.pt
```

Windows

```cmd
set MODEL_PATH=C:\path\to\best.pt
```

# Running

```bash
python mop_tracker.py
```

# Keyboard Controls

| Key | Action |
|-----|--------|
| S | Save trajectory and start a new one |
| R | Reset current trajectory |
| Q | Save trajectory and quit |

# Outputs

- framewise_data.csv
- boundary_points.csv
- temporal_segments.csv
- trajectory_summary.csv
- all_trajectories_summary.csv
- segmentation_output.json
- trajectory_graph.png
- annotated_video.mp4
- raw_video.mp4

# Applications

- Robot manipulation
- Motion analysis
- Cloth trajectory analysis
- Object trajectory segmentation
- Computer vision research

# License

Add your preferred open-source license (MIT recommended).
