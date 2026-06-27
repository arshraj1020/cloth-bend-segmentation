"""
Real-Time YOLOv8 (OBB) Mop Center-Trajectory Tracker
====================================================
Tracks the CENTER point of a detected "mop" object using a custom YOLO model,
records its trajectory in real time, automatically splits it into temporal
segments at sharp direction changes, and exports CSV / JSON / graph outputs.

The anchor used for tracking is the geometric CENTER of the oriented
bounding box: the mean of its 4 corners, i.e. ((x1+x2)/2, (y1+y2)/2).

--------------------------------------------------------------------------
QUICK START
--------------------------------------------------------------------------
1) Install dependencies:
       pip install ultralytics opencv-python numpy scipy matplotlib

2) Place your trained model file (best.pt) next to this script,
   OR set its location via the MODEL_PATH environment variable
   (see the "USER SETTINGS" section below).

3) Run:
       python mop_tracker.py

--------------------------------------------------------------------------
CONTROLS (while the camera window is focused)
--------------------------------------------------------------------------
    s  =  save current trajectory  ->  start a new one
    r  =  reset current trajectory WITHOUT saving
    q  =  save current trajectory  ->  quit

--------------------------------------------------------------------------
OUTPUT FILES (per trajectory)
--------------------------------------------------------------------------
    framewise_data.csv             - one row per frame
    boundary_points.csv            - detected segment boundaries
    temporal_segments.csv          - segment start/end ranges
    trajectory_summary.csv         - single-row summary of the trajectory
    segmentation_output.json       - everything above in JSON form
    *_trajectory_graph.png         - final high-res trajectory plot
    *_annotated_video.mp4          - video with overlays
    *_raw_video.mp4                - raw camera video

A master CSV (all_trajectories_summary.csv) is also appended to in the
output root, with one row per saved trajectory across all runs.

==========================================================================
  HOW TO CONFIGURE THIS SCRIPT FOR YOUR SETUP
==========================================================================
Almost everything you need to change lives in the "USER SETTINGS" section
right below. The most common things to edit:

  * MODEL_PATH        -> path to your trained best.pt (or use env var)
  * TARGET_CLASS      -> the class name your model was trained on
  * CAMERA_INDEX      -> which webcam to use (0, 1, 2, ...)
  * OUTPUT_DIR_ENV    -> where results are written (or use env var)
  * CONF_THRESHOLD    -> detection confidence cutoff

You normally do NOT need to touch anything below the USER SETTINGS block.
==========================================================================
"""

print("SCRIPT STARTED", flush=True)

import os
import csv
import json
import math
from datetime import datetime

import cv2
import numpy as np
from scipy.signal import savgol_filter

import matplotlib
matplotlib.use("Agg")  # headless backend; we render plots to images, not a GUI
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ══════════════════════════════════════════════════════════════════════════════
# USER SETTINGS  ──  EDIT THIS SECTION TO MATCH YOUR SETUP
# ══════════════════════════════════════════════════════════════════════════════

# ── Camera & detection ───────────────────────────────────────────────────────
CAMERA_INDEX       = 0          # Webcam index. Try 1 or 2 if 0 doesn't work.
TARGET_CLASS       = "mop"      # MUST match a class name in your trained model.
CONF_THRESHOLD     = 0.5        # Min confidence. Lower = more (noisier) detections.
YOLO_IMGSZ         = 480        # Inference image size. Must be a multiple of 32.

# ── Display & saving toggles ─────────────────────────────────────────────────
SHOW_WINDOWS           = True   # Show the live camera window.
SHOW_CURRENT_POINT     = True   # Draw a cyan dot at the current center point.
SAVE_ANNOTATED_VIDEO   = True   # Save annotated + raw video files.
SHOW_LIVE_GRAPH_WINDOW = True   # Show a live, separate trajectory-graph window.

PLAYBACK_DELAY_MS = 1           # cv2.waitKey delay. Increase to slow playback.

# ── Model location ───────────────────────────────────────────────────────────
# Option A (recommended): just drop your "best.pt" next to this script.
# Option B: set the MODEL_PATH environment variable to an absolute path, e.g.
#       Windows : set MODEL_PATH=C:\models\best.pt
#       macOS/Linux : export MODEL_PATH=/home/you/models/best.pt
#
# The script tries the env var first, then a few sensible default locations.
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
CURRENT_DIR = os.getcwd()

MODEL_PATH_CANDIDATES = [
    os.environ.get("MODEL_PATH", ""),          # 1) environment variable (if set)
    os.path.join(SCRIPT_DIR,  "best.pt"),       # 2) same folder as this script
    os.path.join(CURRENT_DIR, "best.pt"),       # 3) current working directory
    "best.pt",                                  # 4) bare filename fallback
]

# ── Output location ──────────────────────────────────────────────────────────
# By default, results are written to a "mop_trajectory_output" folder inside
# this script's directory. To put them somewhere else, set the
# MOP_OUTPUT_DIR environment variable, e.g.
#       export MOP_OUTPUT_DIR=/path/to/results        (macOS/Linux)
#       set MOP_OUTPUT_DIR=D:\results                 (Windows)
BASE_OUTPUT_DIR = os.environ.get(
    "MOP_OUTPUT_DIR",
    os.path.join(SCRIPT_DIR, "mop_trajectory_output"),
)

# ── Camera capture settings ──────────────────────────────────────────────────
CAMERA_WIDTH  = 640
CAMERA_HEIGHT = 480
CAMERA_FPS    = 30

YOLO_EVERY_N_FRAMES = 1         # Run YOLO every N frames (1 = every frame).
MAX_HOLD_FRAMES     = 5         # Frames to keep last position when detection drops.

# ── Smoothing ────────────────────────────────────────────────────────────────
SAVGOL_WINDOW    = 11           # Savitzky-Golay window (odd number).
SAVGOL_POLYORDER = 2            # Savitzky-Golay polynomial order.
EMA_ALPHA        = 0.35         # Exponential-moving-average weight (0-1).

# ── Start-of-motion detection ────────────────────────────────────────────────
START_SPEED_PIXELS_PER_FRAME      = 0.35
START_MOTION_CONSECUTIVE_FRAMES   = 3
START_MOTION_DISTANCE_PIXELS      = 4.0
START_REST_BACKTRACK_FRAMES       = 1
FORCE_START_AFTER_VALID_POINTS    = 5

# ── Trajectory segmentation (split at sharp bends) ───────────────────────────
MIN_BEND_ANGLE_DEG              = 30.0   # Angle that counts as a direction change.
BEND_WINDOW_FRAMES              = 5      # Frames before/after used to measure a bend.
MIN_BOUNDARY_GAP_FRAMES         = 10     # Min frames between two boundaries.
FIRST_BOUNDARY_MIN_GAP_FRAMES   = 3      # Min frames before the first boundary.
MIN_SEGMENT_DISPLACEMENT_PIXELS = 6.0    # Ignore tiny jitters as bends.
MIN_SEGMENT_LENGTH_FRAMES       = 5      # Drop segments shorter than this.
MAX_BOUNDARIES                  = None   # Cap on number of boundaries (None = unlimited).

# ── Final saved graph (PNG) appearance ───────────────────────────────────────
FIGURE_SIZE          = (10, 7)
FIGURE_DPI           = 220
SAVE_DPI             = 350
GRAPH_PADDING_PIXELS = 100
MARKER_SIZE          = 90        # Scatter marker size (in points squared).

X_LABEL_TEXT = "X"
Y_LABEL_TEXT = "Y"
FOOTER_TEXT  = ""                # Optional footer caption on the saved graph.

AXIS_LABEL_FONTSIZE   = 13
AXIS_LABEL_FONTWEIGHT = "bold"
TICK_LABEL_FONTSIZE   = 11
LEGEND_FONTSIZE       = 11
LEGEND_TITLE_FONTSIZE = 12

# ── Live graph window appearance ─────────────────────────────────────────────
LIVE_GRAPH_UPDATE_EVERY_N_FRAMES = 5
LIVE_GRAPH_WIN_W = 900
LIVE_GRAPH_WIN_H = 620
LIVE_GRAPH_DPI   = 130

# ══════════════════════════════════════════════════════════════════════════════
# END OF USER SETTINGS — you normally don't need to edit below this line.
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# OUTPUT PATH MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════
# A unique, timestamped folder is created for each program run so results from
# different sessions never overwrite each other.

RUN_NAME     = "mop_realtime_" + datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_ROOT_DIR = os.path.join(BASE_OUTPUT_DIR, RUN_NAME)
os.makedirs(RUN_ROOT_DIR, exist_ok=True)

# Master CSV (appended across ALL runs) lives in the output root.
master_summary_csv = os.path.join(BASE_OUTPUT_DIR, "all_trajectories_summary.csv")

SAVE_AND_NEXT_KEY      = ord("s")
CURRENT_TRAJECTORY_ID  = 1
ACTIVE_CAMERA_INDEX    = CAMERA_INDEX

# These globals are (re)assigned per trajectory by set_trajectory_output_paths().
OUTPUT_DIR                   = ""
output_video_path            = ""
raw_video_path               = ""
framewise_csv_path           = ""
boundary_csv_path            = ""
segment_csv_path             = ""
trajectory_summary_csv_path  = ""
json_path                    = ""
final_image_path             = ""
preview_image_path           = ""


def set_trajectory_output_paths(trajectory_id):
    """Build all output file paths for a given trajectory id and make its folder."""
    global CURRENT_TRAJECTORY_ID, OUTPUT_DIR
    global output_video_path, raw_video_path, framewise_csv_path
    global boundary_csv_path, segment_csv_path
    global trajectory_summary_csv_path, json_path
    global final_image_path, preview_image_path

    CURRENT_TRAJECTORY_ID = int(trajectory_id)
    prefix = f"trajectory_{CURRENT_TRAJECTORY_ID:03d}"

    OUTPUT_DIR = os.path.join(RUN_ROOT_DIR, prefix)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    output_video_path           = os.path.join(OUTPUT_DIR, f"{prefix}_annotated_video.mp4")
    raw_video_path              = os.path.join(OUTPUT_DIR, f"{prefix}_raw_video.mp4")
    framewise_csv_path          = os.path.join(OUTPUT_DIR, "framewise_data.csv")
    boundary_csv_path           = os.path.join(OUTPUT_DIR, "boundary_points.csv")
    segment_csv_path            = os.path.join(OUTPUT_DIR, "temporal_segments.csv")
    trajectory_summary_csv_path = os.path.join(OUTPUT_DIR, "trajectory_summary.csv")
    json_path                   = os.path.join(OUTPUT_DIR, "segmentation_output.json")
    final_image_path            = os.path.join(OUTPUT_DIR, f"{prefix}_trajectory_graph.png")
    preview_image_path          = os.path.join(OUTPUT_DIR, f"{prefix}_trajectory_graph_preview.png")


set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)


# ══════════════════════════════════════════════════════════════════════════════
# GENERAL UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

def segment_label(seg_id):
    """Human-readable label for a segment id (e.g. 1 -> 'Segment_1')."""
    return f"Segment_{seg_id}"


def euclidean(p1, p2):
    """Straight-line distance between two (x, y) points."""
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def is_finite(p):
    """True if p is a valid (x, y) point with finite numeric coordinates."""
    if p is None:
        return False
    try:
        return np.isfinite(float(p[0])) and np.isfinite(float(p[1]))
    except Exception:
        return False


def is_nan(v):
    """True if v is NaN or cannot be interpreted as a number."""
    try:
        return bool(np.isnan(float(v)))
    except Exception:
        return True


def clip_bbox(bbox, w, h):
    """Clamp a bbox to image bounds and guarantee a minimum size."""
    x1, y1, x2, y2 = bbox
    x1 = int(max(0, min(w - 1, x1)))
    y1 = int(max(0, min(h - 1, y1)))
    x2 = int(max(0, min(w - 1, x2)))
    y2 = int(max(0, min(h - 1, y2)))
    if x2 <= x1: x2 = min(w - 1, x1 + 2)
    if y2 <= y1: y2 = min(h - 1, y1 + 2)
    return x1, y1, x2, y2


def bbox_metrics(bbox):
    """Return width/height/area/aspect-ratio for a bbox (logged per frame)."""
    x1, y1, x2, y2 = bbox
    bw = max(1.0, float(x2 - x1))
    bh = max(1.0, float(y2 - y1))
    return {"bbox_width": bw, "bbox_height": bh,
            "bbox_area": bw * bh, "aspect_ratio": bw / bh}


def center_anchor(bbox):
    """Return the geometric center (cx, cy) of an axis-aligned bbox.

    NOTE: For OBB models the true center is computed in the main loop as the
    mean of the 4 rotated corners; this helper is kept for completeness.
    """
    x1, y1, x2, y2 = bbox
    return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)


def best_detection(detections, last_anchor):
    """Pick the detection to track this frame.

    If we have no previous position, take the most confident detection.
    Otherwise take the one closest to where the object was last frame.
    """
    if not detections:
        return None
    if last_anchor is None:
        return max(detections, key=lambda d: d["confidence"])
    return min(detections, key=lambda d: euclidean(d["anchor"], last_anchor))


def angle_between(v1, v2):
    """Angle (degrees) between two 2D vectors; 0 if either is near-zero length."""
    n1, n2 = float(np.linalg.norm(v1)), float(np.linalg.norm(v2))
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    cos_v = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_v)))


# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

def savgol_smooth(points):
    """Savitzky-Golay smoothing applied to the most recent window of points.

    Returns the smoothed value of the LATEST point. Falls back gracefully when
    there aren't enough points yet to form a valid window.
    """
    if len(points) < 3:
        return points[-1]
    local = points[-SAVGOL_WINDOW:] if len(points) >= SAVGOL_WINDOW else points[:]
    n   = len(local)
    win = SAVGOL_WINDOW if SAVGOL_WINDOW <= n else (n if n % 2 == 1 else n - 1)
    if win % 2 == 0: win -= 1
    if win < 3:      return local[-1]
    poly = min(SAVGOL_POLYORDER, win - 1)
    xs   = np.array([p[0] for p in local], dtype=np.float32)
    ys   = np.array([p[1] for p in local], dtype=np.float32)
    try:
        sx = savgol_filter(xs, window_length=win, polyorder=poly, mode="interp")
        sy = savgol_filter(ys, window_length=win, polyorder=poly, mode="interp")
        return float(sx[-1]), float(sy[-1])
    except Exception:
        return local[-1]


def ema_filter(cur, prev, alpha):
    """Exponential moving average between current and previous point."""
    if prev is None:
        return cur
    return (alpha * cur[0] + (1 - alpha) * prev[0],
            alpha * cur[1] + (1 - alpha) * prev[1])


# ══════════════════════════════════════════════════════════════════════════════
# START DETECTION
# ══════════════════════════════════════════════════════════════════════════════

def detect_start(smooth_points, frame_numbers):
    """Find the frame where the mop first starts moving.

    Looks for a run of frames whose per-frame speed exceeds a threshold, then
    backtracks slightly to mark the rest-to-motion transition. Falls back to the
    first valid point once enough points have accumulated.
    """
    if len(smooth_points) < START_MOTION_CONSECUTIVE_FRAMES + 3:
        return None, None
    pts    = np.array(smooth_points, dtype=np.float32)
    speeds = np.linalg.norm(pts[1:] - pts[:-1], axis=1)
    run    = 0
    fi     = None
    for i, spd in enumerate(speeds):
        if spd >= START_SPEED_PIXELS_PER_FRAME:
            run += 1
            if run >= START_MOTION_CONSECUTIVE_FRAMES:
                fi = max(0, i - START_MOTION_CONSECUTIVE_FRAMES + 1)
                break
        else:
            run = 0
    if fi is None:
        if len(smooth_points) >= FORCE_START_AFTER_VALID_POINTS:
            return 0, int(frame_numbers[0])
        return None, None
    ri   = max(0, fi - START_REST_BACKTRACK_FRAMES)
    li   = min(len(pts) - 1, ri + START_MOTION_CONSECUTIVE_FRAMES + 3)
    disp = float(np.linalg.norm(pts[li] - pts[ri]))
    if disp < START_MOTION_DISTANCE_PIXELS:
        if len(smooth_points) >= FORCE_START_AFTER_VALID_POINTS:
            return 0, int(frame_numbers[0])
        return None, None
    return ri, int(frame_numbers[ri])


# ══════════════════════════════════════════════════════════════════════════════
# SEGMENTATION
# ══════════════════════════════════════════════════════════════════════════════

def detect_boundaries(smooth_points, frame_numbers, start_index):
    """Find candidate segment boundaries where the path bends sharply.

    For each point we compare the incoming and outgoing direction vectors; a
    large enough angle (and enough movement) marks a boundary candidate.
    Nearby candidates are merged, keeping the sharpest bend.
    """
    if start_index is None:
        return []
    pts_a  = smooth_points[start_index:]
    frms_a = frame_numbers[start_index:]
    if len(pts_a) < (2 * BEND_WINDOW_FRAMES + 1):
        return []
    pts_np     = np.array(pts_a, dtype=np.float32)
    candidates = []
    for i in range(BEND_WINDOW_FRAMES, len(pts_np) - BEND_WINDOW_FRAMES):
        v1 = pts_np[i]                      - pts_np[i - BEND_WINDOW_FRAMES]
        v2 = pts_np[i + BEND_WINDOW_FRAMES] - pts_np[i]
        d1 = float(np.linalg.norm(v1))
        d2 = float(np.linalg.norm(v2))
        if d1 < MIN_SEGMENT_DISPLACEMENT_PIXELS or d2 < MIN_SEGMENT_DISPLACEMENT_PIXELS:
            continue
        ang = angle_between(v1, v2)
        if ang >= MIN_BEND_ANGLE_DEG:
            candidates.append({
                "frame_number": int(frms_a[i]),
                "x":            float(pts_a[i][0]),
                "y":            float(pts_a[i][1]),
                "bend_angle":   float(ang),
                "score":        float(ang / max(MIN_BEND_ANGLE_DEG, 1e-6)),
            })
    if not candidates:
        return []

    # Merge candidates that are too close together; keep the sharpest one.
    selected = []
    for c in candidates:
        cf = int(c["frame_number"])
        if not selected:
            if cf - int(frms_a[0]) < FIRST_BOUNDARY_MIN_GAP_FRAMES:
                continue
            selected.append(c)
        else:
            lf = int(selected[-1]["frame_number"])
            if cf - lf < MIN_BOUNDARY_GAP_FRAMES:
                if c["bend_angle"] > selected[-1]["bend_angle"]:
                    selected[-1] = c
            else:
                selected.append(c)
        if MAX_BOUNDARIES is not None and len(selected) >= MAX_BOUNDARIES:
            break

    boundaries = []
    for i, c in enumerate(selected, start=1):
        boundaries.append({
            "boundary_id":      i,
            "frame_number":     int(c["frame_number"]),
            "x":                float(c["x"]),
            "y":                float(c["y"]),
            "direction_change": float(c["bend_angle"]),
            "boundary_score":   float(c["score"]),
            "cue_type":         "bend_angle_ge_30deg",
        })
    return boundaries


def clean_boundaries(boundaries, start_frame, total_frames):
    """Remove boundaries that would create too-short segments or sit too close."""
    if start_frame is None:
        return []
    cleaned    = []
    prev_start = int(start_frame)
    for b in sorted(boundaries, key=lambda x: x["frame_number"]):
        bf = int(b["frame_number"])
        if bf <= prev_start:
            continue
        if bf - prev_start + 1 < MIN_SEGMENT_LENGTH_FRAMES:
            continue
        if cleaned and bf - int(cleaned[-1]["frame_number"]) < MIN_BOUNDARY_GAP_FRAMES:
            continue
        cleaned.append(dict(b))
        prev_start = bf + 1
    # Drop a trailing boundary that leaves too short a final segment.
    while cleaned and total_frames - int(cleaned[-1]["frame_number"]) < MIN_SEGMENT_LENGTH_FRAMES:
        cleaned.pop()
    for i, b in enumerate(cleaned, start=1):
        b["boundary_id"] = i
    return cleaned


def make_segments(boundaries, start_frame, total_frames):
    """Turn a list of boundaries into contiguous (start, end) segment ranges."""
    if total_frames <= 0 or start_frame is None:
        return []
    cleaned = clean_boundaries(boundaries, start_frame, total_frames)
    segs    = []
    seg_st  = int(start_frame)
    for b in cleaned:
        bf = int(b["frame_number"])
        if bf < seg_st:
            continue
        sid = len(segs) + 1
        segs.append({"segment_id": sid, "label": segment_label(sid),
                     "start_frame": seg_st, "end_frame": bf,
                     "duration_frames": bf - seg_st + 1})
        seg_st = bf + 1
    if seg_st <= total_frames:
        sid = len(segs) + 1
        segs.append({"segment_id": sid, "label": segment_label(sid),
                     "start_frame": seg_st, "end_frame": total_frames,
                     "duration_frames": total_frames - seg_st + 1})
    return segs


def seg_for_frame(frame_no, boundaries, start_frame):
    """Return (segment_id, label) for a given frame number."""
    if start_frame is None or frame_no < start_frame:
        return 0, "Before_START"
    sid = 1
    for b in sorted(boundaries, key=lambda x: x["frame_number"]):
        if frame_no > b["frame_number"]:
            sid += 1
        else:
            break
    return sid, segment_label(sid)


# ══════════════════════════════════════════════════════════════════════════════
# CAMERA FEED OVERLAY
# ══════════════════════════════════════════════════════════════════════════════
# NOTE: OpenCV uses BGR color order, so e.g. (255, 0, 0) is BLUE, not red.

def draw_overlay(frame, smooth_points, boundaries, start_point, start_frame):
    """Draw the trajectory line, start dot, boundary dots, and current point."""
    # Blue trajectory line
    valid = [(int(p[0]), int(p[1])) for p in smooth_points if is_finite(p)]
    for i in range(1, len(valid)):
        cv2.line(frame, valid[i - 1], valid[i], (255, 0, 0), 3, cv2.LINE_AA)

    # Green start dot
    if start_point is not None and start_frame is not None:
        cv2.circle(frame, (int(start_point[0]), int(start_point[1])),
                   9, (0, 255, 0), -1, cv2.LINE_AA)

    # Red segment-boundary dots
    for b in boundaries:
        cv2.circle(frame, (int(b["x"]), int(b["y"])),
                   10, (0, 0, 255), -1, cv2.LINE_AA)

    # Cyan current CENTER dot
    if SHOW_CURRENT_POINT and smooth_points:
        p = smooth_points[-1]
        if is_finite(p):
            cv2.circle(frame, (int(p[0]), int(p[1])),
                       7, (255, 255, 0), -1, cv2.LINE_AA)  # cyan in BGR


def draw_bbox(frame, bbox, conf, corners=None):
    """Draw the oriented bounding box polygon + center crosshair on the live feed."""
    if bbox is None:
        return
    x1, y1, x2, y2 = bbox
    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

    if corners is not None:
        # Draw the actual rotated polygon (tight, follows the mop's orientation).
        pts = corners.astype(np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255),
                      thickness=2, lineType=cv2.LINE_AA)
    else:
        # Fallback: plain axis-aligned rectangle.
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2, cv2.LINE_AA)

    # Center crosshair (+)
    half = 12
    cv2.line(frame, (cx - half, cy), (cx + half, cy), (0, 200, 255), 2, cv2.LINE_AA)
    cv2.line(frame, (cx, cy - half), (cx, cy + half), (0, 200, 255), 2, cv2.LINE_AA)

    # Confidence label
    label = f"{TARGET_CLASS} {conf:.2f}"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), (0, 200, 255), -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def draw_status(frame, smooth_points, boundaries):
    """Draw the top status bar with counts and key hints."""
    text = (f"T{CURRENT_TRAJECTORY_ID:03d} | "
            f"pts={len(smooth_points)} | "
            f"boundaries={len(boundaries)} | "
            "[s]save  [r]reset  [q]quit")
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0),       3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ══════════════════════════════════════════════════════════════════════════════
# LIVE GRAPH WINDOW  (matplotlib -> OpenCV image)
# ══════════════════════════════════════════════════════════════════════════════

def render_live_graph(smooth_points, boundaries, start_point):
    """Render the current trajectory as a matplotlib figure, returned as a BGR image."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    from matplotlib.figure import Figure

    fig = Figure(figsize=(LIVE_GRAPH_WIN_W / LIVE_GRAPH_DPI,
                          LIVE_GRAPH_WIN_H / LIVE_GRAPH_DPI),
                 dpi=LIVE_GRAPH_DPI)
    canvas = FigureCanvas(fig)
    ax = fig.add_subplot(111)

    valid = [p for p in smooth_points if is_finite(p)]
    if len(valid) >= 2:
        xs = [p[0] for p in valid]
        ys = [p[1] for p in valid]
        ax.plot(xs, ys, color="blue", linewidth=2.5, label="Trajectory", zorder=3)

        # End point (orange)
        ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
                   edgecolors="black", linewidths=0.7, label="End point", zorder=6)

        pad = GRAPH_PADDING_PIXELS
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(max(ys) + pad, min(ys) - pad)   # Y axis inverted (image coords)

    # Start point (green)
    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.7, label="Start point", zorder=7)

    # Segment / boundary points (red)
    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red", edgecolors="black",
                   linewidths=0.7, label="Segment point", zorder=8)

    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4, color="grey")

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate(X_LABEL_TEXT,
                xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
                xytext=(6, -18), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="top")
    ax.annotate(Y_LABEL_TEXT,
                xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
                xytext=(-28, 6), textcoords="offset points",
                fontsize=9, fontweight="bold", ha="left", va="bottom")

    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="black", framealpha=1.0,
              fontsize=8, title="Legend", title_fontsize=9,
              borderpad=0.7, labelspacing=0.5)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.text(0.01, 0.01, FOOTER_TEXT, fontsize=6, color="black")

    canvas.draw()
    buf = np.asarray(canvas.buffer_rgba())
    img = cv2.cvtColor(buf[..., :3], cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, (LIVE_GRAPH_WIN_W, LIVE_GRAPH_WIN_H),
                     interpolation=cv2.INTER_AREA)
    plt.close(fig)
    return img


# ══════════════════════════════════════════════════════════════════════════════
# FINAL HIGH-QUALITY GRAPH  (saved to disk)
# ══════════════════════════════════════════════════════════════════════════════

def save_final_graph(smooth_points, boundaries, start_point):
    """Render and save the final, publication-quality trajectory plot."""
    valid = [p for p in smooth_points if is_finite(p)]
    if not valid:
        print("[Graph] No valid points — skipping graph save.", flush=True)
        return

    xs = [float(p[0]) for p in valid]
    ys = [float(p[1]) for p in valid]

    # Fixed axis limits = full camera frame, so every saved PNG has identical size.
    x_min, x_max = 0, CAMERA_WIDTH
    y_min, y_max = 0, CAMERA_HEIGHT

    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=FIGURE_DPI)

    # Trajectory line
    ax.plot(xs, ys, color="blue", linewidth=2.8, label="Trajectory", zorder=3)

    # Start point (green)
    if is_finite(start_point):
        ax.scatter([float(start_point[0])], [float(start_point[1])],
                   s=MARKER_SIZE, color="lime", edgecolors="black",
                   linewidths=0.8, label="Start point", zorder=6)

    # Segment / boundary points (red)
    if boundaries:
        bx = [b["x"] for b in boundaries]
        by = [b["y"] for b in boundaries]
        ax.scatter(bx, by, s=MARKER_SIZE, color="red",
                   edgecolors="black", linewidths=0.8,
                   label="Segment point", zorder=7)

    # End point (orange)
    ax.scatter([xs[-1]], [ys[-1]], s=MARKER_SIZE, color="orange",
               edgecolors="black", linewidths=0.8, label="End point", zorder=6)

    # Axis limits (Y inverted so 0 is at the top, matching image coordinates)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)

    # Axis labels placed at the ends of the axes
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.annotate(X_LABEL_TEXT,
                xy=(1, 0), xycoords=("axes fraction", "axes fraction"),
                xytext=(6, -20), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="top", annotation_clip=False)
    ax.annotate(Y_LABEL_TEXT,
                xy=(0, 1), xycoords=("axes fraction", "axes fraction"),
                xytext=(-30, 6), textcoords="offset points",
                fontsize=AXIS_LABEL_FONTSIZE, fontweight=AXIS_LABEL_FONTWEIGHT,
                ha="left", va="bottom", annotation_clip=False)

    # Grid
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45, color="grey")

    # Tick labels
    for tick in ax.get_xticklabels() + ax.get_yticklabels():
        tick.set_fontsize(TICK_LABEL_FONTSIZE)
        tick.set_fontweight("bold")
    ax.tick_params(axis="both", which="major", length=6, width=1.4, direction="out")

    # Spines
    for spine in ax.spines.values():
        spine.set_linewidth(1.3)
        spine.set_color("black")

    # Legend
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        fancybox=False,
        edgecolor="black",
        framealpha=1.0,
        fontsize=LEGEND_FONTSIZE,
        title="Legend",
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        borderpad=0.8,
        labelspacing=0.7,
    )
    legend.get_title().set_fontweight("bold")
    legend.get_frame().set_linewidth(1.1)

    # Footer
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.18, top=0.92)
    fig.text(0.02, 0.02, FOOTER_TEXT, fontsize=8, color="black",
             wrap=True, ha="left", va="bottom")

    fig.savefig(final_image_path,   dpi=SAVE_DPI, facecolor="white")
    fig.savefig(preview_image_path, dpi=150,      facecolor="white")
    plt.close(fig)
    print(f"[Graph] Saved → {final_image_path}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# CSV / JSON SAVE
# ══════════════════════════════════════════════════════════════════════════════

def save_framewise_csv(records, boundaries, start_frame):
    """Write one row per frame with tracking + smoothing + segment info."""
    boundary_map = {int(b["frame_number"]): b for b in boundaries}
    with open(framewise_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "frame_number", "segment_id", "segment_label",
            "tracking_source", "detected", "confidence",
            "x_raw", "y_raw", "x_smooth", "y_smooth",
            "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
            "bbox_width", "bbox_height", "bbox_area", "aspect_ratio",
            "anchor_type",
            "boundary_id", "boundary_cue", "is_boundary",
            "motion_start_frame",
        ])
        for r in records:
            fid = int(r["frame_number"])
            sid, slbl = seg_for_frame(fid, boundaries, start_frame)
            b = boundary_map.get(fid)
            w.writerow([
                fid, sid, slbl,
                r["tracking_source"], r["detected"], r["confidence"],
                r["x_raw"], r["y_raw"], r["x_smooth"], r["y_smooth"],
                r["bbox_x1"], r["bbox_y1"], r["bbox_x2"], r["bbox_y2"],
                r["bbox_width"], r["bbox_height"], r["bbox_area"], r["aspect_ratio"],
                "center",
                b["boundary_id"] if b else "",
                b["cue_type"]    if b else "none",
                bool(b),
                start_frame if start_frame is not None else "",
            ])


def save_boundary_csv(boundaries):
    """Write the detected segment boundary points."""
    with open(boundary_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["boundary_id", "frame_number", "x", "y",
                    "direction_change_deg", "boundary_score", "cue_type"])
        for b in boundaries:
            w.writerow([b["boundary_id"], b["frame_number"], b["x"], b["y"],
                        b["direction_change"], b["boundary_score"], b["cue_type"]])


def save_segment_csv(segments):
    """Write the temporal segment ranges."""
    with open(segment_csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment_id", "label", "start_frame", "end_frame", "duration_frames"])
        for s in segments:
            w.writerow([s["segment_id"], s["label"],
                        s["start_frame"], s["end_frame"], s["duration_frames"]])


def save_summary_csv(records, boundaries, segments, start_frame, start_point, total_frames):
    """Write a single-row trajectory summary and append it to the master CSV."""
    if not records:
        return
    valid = [r for r in records if not is_nan(r["x_smooth"]) and not is_nan(r["y_smooth"])]
    if not valid:
        return

    last        = valid[-1]
    end_frame   = int(last["frame_number"])
    traj_id_str = f"{RUN_NAME}_trajectory_{CURRENT_TRAJECTORY_ID:03d}"

    row = {
        "trajectory_id":      traj_id_str,
        "trajectory_number":  CURRENT_TRAJECTORY_ID,
        "target_class":       TARGET_CLASS,
        "anchor_type":        "center",
        "camera_index":       ACTIVE_CAMERA_INDEX,
        "total_frames":       total_frames,
        "start_frame":        start_frame if start_frame is not None else "",
        "start_x":            float(start_point[0]) if is_finite(start_point) else "",
        "start_y":            float(start_point[1]) if is_finite(start_point) else "",
        "end_frame":          end_frame,
        "end_x":              float(last["x_smooth"]),
        "end_y":              float(last["y_smooth"]),
        "num_segment_points": len(boundaries),
        "num_segments":       len(segments),
        "boundary_frames":    ";".join(str(b["frame_number"]) for b in boundaries),
        "segment_ranges":     ";".join(
            f"{s['label']}:{s['start_frame']}-{s['end_frame']}" for s in segments),
        "output_folder":      OUTPUT_DIR,
        "final_graph":        final_image_path,
    }

    with open(trajectory_summary_csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        w.writeheader(); w.writerow(row)

    master_exists = os.path.exists(master_summary_csv)
    with open(master_summary_csv, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not master_exists: w.writeheader()
        w.writerow(row)

    print(f"[T{CURRENT_TRAJECTORY_ID:03d}] pts={len(valid)}  "
          f"boundaries={len(boundaries)}  segments={len(segments)}", flush=True)


def save_json_output(start_frame, start_point, boundaries, segments, total_frames):
    """Write the full segmentation result (plus output file paths) as JSON."""
    with open(json_path, "w") as f:
        json.dump({
            "trajectory_id": CURRENT_TRAJECTORY_ID,
            "camera_index":  ACTIVE_CAMERA_INDEX,
            "target_class":  TARGET_CLASS,
            "anchor_type":   "center",
            "total_frames":  total_frames,
            "motion_start":  ({"frame_number": int(start_frame),
                               "x": float(start_point[0]),
                               "y": float(start_point[1])}
                              if start_frame is not None and is_finite(start_point)
                              else None),
            "boundaries":    boundaries,
            "segments":      segments,
            "output_files": {
                "framewise_csv":  framewise_csv_path,
                "boundary_csv":   boundary_csv_path,
                "segment_csv":    segment_csv_path,
                "summary_csv":    trajectory_summary_csv_path,
                "final_graph":    final_image_path,
                "preview_graph":  preview_image_path,
                "video":          output_video_path,
            },
        }, f, indent=4)


# ══════════════════════════════════════════════════════════════════════════════
# SAVE COMPLETE TRAJECTORY
# ══════════════════════════════════════════════════════════════════════════════

def save_trajectory(state, width, height):
    """Run final segmentation and write all CSV/JSON/graph/video outputs."""
    if not state["records"]:
        print("[Save] Nothing to save.", flush=True)
        return

    total = int(state["frame_number"])
    sf    = state["start_frame"]
    si    = state["start_index"]

    raw_b   = detect_boundaries(state["smooth_points"], state["frame_numbers"], si)
    final_b = clean_boundaries(raw_b, sf, total)
    final_s = make_segments(final_b, sf, total)

    sp = state["smooth_points"][si] if (si is not None and si < len(state["smooth_points"])) else None

    save_framewise_csv(state["records"], final_b, sf)
    save_boundary_csv(final_b)
    save_segment_csv(final_s)
    save_summary_csv(state["records"], final_b, final_s, sf, sp, total)
    save_json_output(sf, sp, final_b, final_s, total)
    save_final_graph(state["smooth_points"], final_b, sp)

    if state["video_writer"] is not None:
        state["video_writer"].release()
        state["video_writer"] = None

    if state["raw_video_writer"] is not None:
        state["raw_video_writer"].release()
        state["raw_video_writer"] = None

    print(f"[Save] Trajectory {CURRENT_TRAJECTORY_ID} → {OUTPUT_DIR}", flush=True)


# ══════════════════════════════════════════════════════════════════════════════
# STATE MANAGEMENT & MAIN LOOP
# ══════════════════════════════════════════════════════════════════════════════

def fresh_state():
    """Create a blank tracking-state dict for a new trajectory."""
    return {
        "frame_number":     0,
        "records":          [],
        "raw_points":       [],
        "smooth_points":    [],
        "frame_numbers":    [],
        "last_anchor":      None,
        "last_bbox":        None,
        "last_conf":        0.0,
        "last_metrics":     None,
        "prev_ema":         None,
        "hold_counter":     0,
        "start_frame":      None,
        "start_index":      None,
        "video_writer":     None,
        "raw_video_writer": None,
    }


def main():
    global CURRENT_TRAJECTORY_ID

    # ── Load the custom model ──────────────────────────────────────────────────
    model = None
    for candidate in MODEL_PATH_CANDIDATES:
        if candidate and os.path.exists(candidate):
            print(f"[YOLO] Loading weights from: {candidate}", flush=True)
            try:
                model = YOLO(candidate)
                break
            except Exception as e:
                print(f"[YOLO] Failed to load {candidate}: {e}", flush=True)

    if model is None:
        print("[CRITICAL ERROR] Could not find your model weights (best.pt).", flush=True)
        print("    Fix: place 'best.pt' next to this script, or set the", flush=True)
        print("    MODEL_PATH environment variable to its full path.", flush=True)
        return

    # ── Report the model's classes ─────────────────────────────────────────────
    print(f"[YOLO] Model classes: {model.names}", flush=True)
    class_ids_for_target = [k for k, v in model.names.items() if v == TARGET_CLASS]
    if not class_ids_for_target:
        print(f"[WARNING] Class '{TARGET_CLASS}' not found in model. "
              f"Available: {list(model.names.values())}", flush=True)
        print("    Fix: set TARGET_CLASS (in USER SETTINGS) to one of the above.", flush=True)
    else:
        print(f"[YOLO] Tracking class '{TARGET_CLASS}' (id={class_ids_for_target})", flush=True)

    # ── Open the camera ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(ACTIVE_CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[CRITICAL ERROR] Could not open camera at index {ACTIVE_CAMERA_INDEX}.", flush=True)
        print("    Fix: try a different CAMERA_INDEX (0, 1, 2, ...) in USER SETTINGS.", flush=True)
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_FPS)

    ret, initial_frame = cap.read()
    if not ret:
        print("[CRITICAL ERROR] Camera returned an empty frame.", flush=True)
        cap.release()
        return
    h, w = initial_frame.shape[:2]

    state = fresh_state()
    print("\n>>> MOP TRACKER RUNNING. Keys: [s] save  [r] reset  [q] quit", flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[Pipeline] Frame grab failure or stream ended.", flush=True)
                break

            state["frame_number"] += 1
            current_fid = state["frame_number"]

            # ── Lazily create the video writers on the first frame ─────────────
            if SAVE_ANNOTATED_VIDEO and state["video_writer"] is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                state["video_writer"] = cv2.VideoWriter(
                    output_video_path, fourcc, CAMERA_FPS, (w, h))
                state["raw_video_writer"] = cv2.VideoWriter(
                    raw_video_path, fourcc, CAMERA_FPS, (w, h))

            # ── YOLO oriented-bounding-box inference ───────────────────────────
            detections = []
            if current_fid % YOLO_EVERY_N_FRAMES == 0:
                results = model.predict(frame, conf=CONF_THRESHOLD,
                                        imgsz=YOLO_IMGSZ, verbose=False)
                obb = results[0].obb   # OBB models expose results via .obb, not .boxes
                if obb is not None:
                    for i in range(len(obb)):
                        cls_id     = int(obb.cls[i])
                        class_name = model.names[cls_id]
                        if class_name == TARGET_CLASS:
                            # xyxyxyxy may be normalized (0-1) or pixel coords;
                            # always convert to pixel space explicitly.
                            corners_raw = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2)
                            if corners_raw.max() <= 1.0:        # normalized -> pixels
                                corners = corners_raw * np.array([w, h], dtype=np.float32)
                            else:
                                corners = corners_raw.astype(np.float32)
                            x1 = int(np.clip(corners[:, 0].min(), 0, w - 1))
                            y1 = int(np.clip(corners[:, 1].min(), 0, h - 1))
                            x2 = int(np.clip(corners[:, 0].max(), 0, w - 1))
                            y2 = int(np.clip(corners[:, 1].max(), 0, h - 1))
                            bbox_clipped = clip_bbox((x1, y1, x2, y2), w, h)
                            # True OBB center = mean of the 4 corners (pixel space).
                            cx = float(corners[:, 0].mean())
                            cy = float(corners[:, 1].mean())
                            anchor = (cx, cy)
                            detections.append({
                                "bbox":       bbox_clipped,
                                "anchor":     anchor,
                                "confidence": float(obb.conf[i]),
                                "corners":    corners,
                            })

            # ── Choose which detection to follow this frame ────────────────────
            chosen = best_detection(detections, state["last_anchor"])

            if chosen is not None:
                state["hold_counter"] = 0
                state["last_anchor"]  = chosen["anchor"]
                state["last_bbox"]    = chosen["bbox"]
                state["last_conf"]    = chosen["confidence"]
                state["last_metrics"] = bbox_metrics(chosen["bbox"])
                raw_pt     = chosen["anchor"]
                source_str = "yolo_detection"
                conf_val   = chosen["confidence"]
            elif state["last_anchor"] is not None and state["hold_counter"] < MAX_HOLD_FRAMES:
                # Briefly hold the last known position when detection drops out.
                state["hold_counter"] += 1
                raw_pt     = state["last_anchor"]
                source_str = "hold_position"
                conf_val   = 0.0
            else:
                raw_pt     = None
                source_str = "lost"
                conf_val   = 0.0

            # ── Filter the point and record telemetry ──────────────────────────
            if raw_pt is not None:
                ema_pt            = ema_filter(raw_pt, state["prev_ema"], EMA_ALPHA)
                state["prev_ema"] = ema_pt
                state["raw_points"].append(raw_pt)

                smooth_pt = savgol_smooth(state["raw_points"])
                state["smooth_points"].append(smooth_pt)
                state["frame_numbers"].append(current_fid)

                if state["start_frame"] is None:
                    s_idx, s_fnum = detect_start(state["smooth_points"], state["frame_numbers"])
                    if s_idx is not None:
                        state["start_index"] = s_idx
                        state["start_frame"] = s_fnum

                metrics     = state["last_metrics"] if state["last_metrics"] else \
                              {"bbox_width": "", "bbox_height": "", "bbox_area": "", "aspect_ratio": ""}
                bbox_coords = state["last_bbox"] if state["last_bbox"] is not None else ["", "", "", ""]

                state["records"].append({
                    "frame_number":    current_fid,
                    "tracking_source": source_str,
                    "detected":        (chosen is not None),
                    "confidence":      conf_val,
                    "x_raw":           raw_pt[0],
                    "y_raw":           raw_pt[1],
                    "x_smooth":        smooth_pt[0],
                    "y_smooth":        smooth_pt[1],
                    "bbox_x1":         bbox_coords[0],
                    "bbox_y1":         bbox_coords[1],
                    "bbox_x2":         bbox_coords[2],
                    "bbox_y2":         bbox_coords[3],
                    **metrics,
                })
            else:
                # No usable point this frame; log a NaN row to keep frames aligned.
                state["records"].append({
                    "frame_number":    current_fid,
                    "tracking_source": source_str,
                    "detected":        False,
                    "confidence":      0.0,
                    "x_raw":           float("nan"), "y_raw": float("nan"),
                    "x_smooth":        float("nan"), "y_smooth": float("nan"),
                    "bbox_x1": "", "bbox_y1": "", "bbox_x2": "", "bbox_y2": "",
                    "bbox_width": "", "bbox_height": "", "bbox_area": "", "aspect_ratio": "",
                })

            # ── Live segmentation for the on-screen overlay ────────────────────
            current_boundaries = detect_boundaries(
                state["smooth_points"], state["frame_numbers"], state["start_index"])

            # ── Build the annotated frame ──────────────────────────────────────
            raw_frame       = frame.copy()
            annotated_frame = frame.copy()

            if chosen is not None:
                draw_bbox(annotated_frame, state["last_bbox"], state["last_conf"],
                          corners=chosen.get("corners"))

            start_pt_val = (state["smooth_points"][state["start_index"]]
                            if state["start_index"] is not None else None)
            draw_overlay(annotated_frame, state["smooth_points"],
                         current_boundaries, start_pt_val, state["start_frame"])
            draw_status(annotated_frame, state["smooth_points"], current_boundaries)

            if state["video_writer"] is not None:
                state["video_writer"].write(annotated_frame)
            if state["raw_video_writer"] is not None:
                state["raw_video_writer"].write(raw_frame)

            if SHOW_WINDOWS:
                cv2.imshow("Mop Center Trajectory Tracker", annotated_frame)

                if (SHOW_LIVE_GRAPH_WINDOW and
                        current_fid % LIVE_GRAPH_UPDATE_EVERY_N_FRAMES == 0):
                    graph_img = render_live_graph(
                        state["smooth_points"], current_boundaries, start_pt_val)
                    cv2.imshow("Mop Center Trajectory Graph", graph_img)

            # ── Keyboard controls ──────────────────────────────────────────────
            key = cv2.waitKey(PLAYBACK_DELAY_MS) & 0xFF
            if key == ord("q"):
                print("\n[Control] Quit. Saving trajectory …", flush=True)
                save_trajectory(state, w, h)
                break
            elif key == ord("s"):
                print(f"\n[Control] Saving trajectory {CURRENT_TRAJECTORY_ID} and rotating.", flush=True)
                save_trajectory(state, w, h)
                CURRENT_TRAJECTORY_ID += 1
                set_trajectory_output_paths(CURRENT_TRAJECTORY_ID)
                state = fresh_state()
            elif key == ord("r"):
                print(f"\n[Control] Resetting trajectory {CURRENT_TRAJECTORY_ID}.", flush=True)
                if state["video_writer"]:     state["video_writer"].release()
                if state["raw_video_writer"]: state["raw_video_writer"].release()
                state = fresh_state()

    except KeyboardInterrupt:
        print("\n[Control] Interrupted.", flush=True)
        save_trajectory(state, w, h)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nSCRIPT TERMINATED CLEANLY.", flush=True)


if __name__ == "__main__":
    main()