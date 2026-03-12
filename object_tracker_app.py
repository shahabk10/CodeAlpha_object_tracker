import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
import threading
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ── YOLOv8 ──────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    st.error("❌ Install ultralytics: pip install ultralytics")
    st.stop()

# ════════════════════════════════════════════════
#  PAGE CONFIG
# ════════════════════════════════════════════════
st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ════════════════════════════════════════════════
#  CSS
# ════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg:      #08090F;
    --surface: #0F1118;
    --panel:   #151821;
    --border:  rgba(255,255,255,0.07);
    --accent:  #6C63FF;
    --green:   #22D3A5;
    --warn:    #F59E0B;
    --red:     #F43F5E;
    --txt:     #E2E6F0;
    --muted:   #5A6280;
}

*, *::before, *::after { box-sizing: border-box; }
html, body { background: var(--bg) !important; }
.stApp { background: var(--bg) !important; font-family: 'DM Sans', sans-serif; }
[class*="st-"], p, span, div, label { color: var(--txt) !important; font-family: 'DM Sans', sans-serif !important; }
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { padding: 1.8rem 2rem 4rem !important; }

/* sidebar */
[data-testid="stSidebar"] { background: var(--surface) !important; border-right: 1px solid var(--border) !important; }
[data-testid="stSidebar"] * { color: var(--txt) !important; }
[data-testid="collapsedControl"] { background: var(--panel) !important; border: 1px solid var(--border) !important; border-radius: 0 8px 8px 0 !important; }
[data-testid="collapsedControl"] svg { fill: var(--accent) !important; }

/* hero */
.hero { text-align: center; padding: 1.5rem 0 1rem; animation: fd 0.5s ease both; }
@keyframes fd { from { opacity:0; transform:translateY(-10px); } to { opacity:1; transform:none; } }
.hero-title { font-family: 'Syne', sans-serif !important; font-size: clamp(1.7rem,3.5vw,2.4rem) !important; font-weight: 800 !important; color: var(--txt) !important; letter-spacing: -0.5px; }
.hero-title em { font-style:normal; background: linear-gradient(110deg, var(--accent), var(--green)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.hero-sub { font-size: 0.85rem; color: var(--muted) !important; margin-top: 0.4rem; }

/* divider */
.div { height:1px; background: linear-gradient(90deg,transparent,var(--border) 30%,var(--border) 70%,transparent); margin: 1rem 0; }

/* stat cards */
.stat { background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 14px 16px; text-align: center; }
.stat-val { font-family: 'Syne', sans-serif !important; font-size: 1.6rem; font-weight: 800; background: linear-gradient(110deg, var(--accent), var(--green)); -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text; }
.stat-lbl { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1.8px; color: var(--muted) !important; margin-top: 3px; }

/* badge */
.badge { display:inline-block; font-size:0.65rem; font-weight:600; letter-spacing:2px; text-transform:uppercase; color:var(--green) !important; border:1px solid rgba(34,211,165,.28); border-radius:100px; padding:4px 14px; background:rgba(34,211,165,.06); margin-bottom:0.8rem; }

/* status dot */
.dot { display:inline-block; width:7px; height:7px; border-radius:50%; margin-right:5px; vertical-align:middle; }
.dot-green { background:var(--green); box-shadow:0 0 0 2px rgba(34,211,165,.25); animation: pl 2s infinite; }
.dot-red   { background:var(--red); }
.dot-warn  { background:var(--warn); }
@keyframes pl { 0%,100% { box-shadow:0 0 0 2px rgba(34,211,165,.25); } 50% { box-shadow:0 0 0 5px rgba(34,211,165,.06); } }

/* video frame */
.vid-wrap { background: var(--panel); border: 1px solid var(--border); border-radius: 14px; overflow: hidden; padding: 8px; }

/* buttons */
.stButton > button { background: var(--panel) !important; border: 1px solid var(--border) !important; color: var(--txt) !important; font-family: 'DM Sans', sans-serif !important; border-radius: 9px !important; transition: all 0.18s !important; font-size: 0.85rem !important; }
.stButton > button:hover { background: rgba(108,99,255,.12) !important; border-color: var(--accent) !important; }

/* sliders */
.stSlider [data-baseweb="slider"] { color: var(--accent) !important; }

/* selectbox */
.stSelectbox [data-baseweb="select"] > div { background: var(--panel) !important; border-color: var(--border) !important; color: var(--txt) !important; }

/* section label */
.lbl { font-size:0.63rem; font-weight:600; letter-spacing:2.5px; text-transform:uppercase; color:var(--muted) !important; margin-bottom:0.6rem; }

hr { border-color: var(--border) !important; }
[data-testid="stMarkdownContainer"] p { color: var(--txt) !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  LOAD MODEL
# ════════════════════════════════════════════════
@st.cache_resource
def load_model(model_name):
    return YOLO(model_name)


# ════════════════════════════════════════════════
#  TRACKING HELPERS  (simple centroid tracker — no extra deps)
# ════════════════════════════════════════════════
class CentroidTracker:
    """Lightweight centroid-based tracker — no extra install needed."""
    def __init__(self, max_disappeared=30, max_distance=80):
        self.next_id   = 0
        self.objects   = {}          # id → centroid
        self.disappeared = {}        # id → frames missing
        self.max_disappeared = max_disappeared
        self.max_distance    = max_distance

    def register(self, centroid):
        self.objects[self.next_id]     = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, oid):
        del self.objects[oid]
        del self.disappeared[oid]

    def update(self, rects):
        if len(rects) == 0:
            for oid in list(self.disappeared.keys()):
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)
            return self.objects

        input_centroids = np.array([
            ((x1+x2)//2, (y1+y2)//2) for (x1,y1,x2,y2) in rects
        ])

        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
        else:
            oids      = list(self.objects.keys())
            obj_cents = np.array(list(self.objects.values()))

            # pairwise distances
            D = np.linalg.norm(obj_cents[:, None] - input_centroids[None, :], axis=2)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if r in used_rows or c in used_cols:
                    continue
                if D[r, c] > self.max_distance:
                    continue
                oid = oids[r]
                self.objects[oid]     = input_centroids[c]
                self.disappeared[oid] = 0
                used_rows.add(r); used_cols.add(c)

            unused_rows = set(range(D.shape[0])) - used_rows
            unused_cols = set(range(D.shape[1])) - used_cols

            for r in unused_rows:
                oid = oids[r]
                self.disappeared[oid] += 1
                if self.disappeared[oid] > self.max_disappeared:
                    self.deregister(oid)

            for c in unused_cols:
                self.register(input_centroids[c])

        return self.objects


# ════════════════════════════════════════════════
#  DRAWING UTILS
# ════════════════════════════════════════════════
COLORS = [
    (108, 99, 255), (34, 211, 165), (245, 158, 11),
    (244, 63, 94),  (56, 189, 248), (168, 85, 247),
    (251, 146, 60), (52, 211, 153), (248, 113, 113),
    (167, 243, 208)
]

def get_color(idx):
    return COLORS[idx % len(COLORS)]

def draw_detections(frame, boxes, track_ids, labels, confidences, show_conf, show_trail, trails):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    for i, (box, tid, label, conf) in enumerate(zip(boxes, track_ids, labels, confidences)):
        x1, y1, x2, y2 = map(int, box)
        color = get_color(tid)
        cx, cy = (x1+x2)//2, (y1+y2)//2

        # trail
        if show_trail:
            trails[tid].append((cx, cy))
            pts = list(trails[tid])
            for j in range(1, len(pts)):
                alpha = j / len(pts)
                c = tuple(int(v * alpha) for v in color)
                cv2.line(overlay, pts[j-1], pts[j], c, 2)

        # filled box background
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        cv2.addWeighted(overlay, 0.12, frame, 0.88, 0, frame)
        overlay = frame.copy()

        # border
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # label tag
        conf_txt = f" {conf:.0%}" if show_conf else ""
        tag = f"  {label} #{tid}{conf_txt}  "
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
        tag_y = max(y1 - 4, th + 6)
        cv2.rectangle(frame, (x1, tag_y - th - 6), (x1 + tw, tag_y + 2), color, -1)
        cv2.putText(frame, tag, (x1, tag_y - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)

        # centroid dot
        cv2.circle(frame, (cx, cy), 4, color, -1)

    return frame


def draw_hud(frame, fps, n_objects, frame_idx):
    h, w = frame.shape[:2]
    # top bar
    cv2.rectangle(frame, (0, 0), (w, 36), (10, 12, 22), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (108, 99, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Objects: {n_objects}", (110, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (34, 211, 165), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Frame: {frame_idx}", (w - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 180), 1, cv2.LINE_AA)
    return frame


# ════════════════════════════════════════════════
#  SESSION STATE
# ════════════════════════════════════════════════
for k, v in [("running", False), ("stop_flag", False), ("frame_count", 0),
             ("total_detected", 0), ("fps_val", 0.0), ("saved_path", None)]:
    if k not in st.session_state:
        st.session_state[k] = v


# ════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════
with st.sidebar:
    st.markdown("### 🎯 Detection Config")
    st.markdown('<span class="dot dot-green"></span><span style="font-size:0.76rem;color:#22D3A5;">YOLOv8 Ready</span>', unsafe_allow_html=True)
    st.markdown("---")

    model_choice = st.selectbox(
        "Model Size",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        help="n=nano (fastest), s=small, m=medium (most accurate)"
    )

    st.markdown("---")
    st.markdown('<p class="lbl">Detection Settings</p>', unsafe_allow_html=True)

    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.95, 0.45, 0.05)
    iou_thresh  = st.slider("IOU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
    max_det     = st.slider("Max Detections / Frame", 10, 100, 50, 5)

    st.markdown("---")
    st.markdown('<p class="lbl">Display Options</p>', unsafe_allow_html=True)

    show_conf   = st.checkbox("Show Confidence %", value=True)
    show_trail  = st.checkbox("Show Motion Trails", value=True)
    show_opencv = st.checkbox("OpenCV Window (local)", value=False)
    save_video  = st.checkbox("Save Output Video", value=False)

    st.markdown("---")
    st.markdown('<p style="font-size:0.72rem;color:#5A6280;line-height:1.7;">📦 <b>Required packages:</b><br>pip install ultralytics opencv-python streamlit</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  HERO
# ════════════════════════════════════════════════
st.markdown("""
<div class="hero">
  <div class="badge">CodeAlpha · Task 4</div>
  <div class="hero-title">Object <em>Detection & Tracking</em></div>
  <div class="hero-sub">YOLOv8 · Centroid Tracker · Real-time · Streamlit + OpenCV</div>
</div>
<div class="div"></div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  STATS ROW
# ════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(f'<div class="stat"><div class="stat-val">{st.session_state.fps_val:.1f}</div><div class="stat-lbl">FPS</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown(f'<div class="stat"><div class="stat-val">{st.session_state.frame_count}</div><div class="stat-lbl">Frames</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown(f'<div class="stat"><div class="stat-val">{st.session_state.total_detected}</div><div class="stat-lbl">Detections</div></div>', unsafe_allow_html=True)
with c4:
    status = "🟢 Running" if st.session_state.running else "⚪ Idle"
    st.markdown(f'<div class="stat"><div class="stat-val" style="font-size:1rem;">{status}</div><div class="stat-lbl">Status</div></div>', unsafe_allow_html=True)

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  SOURCE SELECTOR
# ════════════════════════════════════════════════
st.markdown('<p class="lbl">Input Source</p>', unsafe_allow_html=True)
source_tab = st.radio("", ["📁 Video File", "📷 Webcam"], horizontal=True, label_visibility="collapsed")

video_path = None

if source_tab == "📁 Video File":
    uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"], label_visibility="collapsed")
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix)
        tfile.write(uploaded.read())
        video_path = tfile.name
        st.success(f"✅ Loaded: **{uploaded.name}**")
else:
    cam_idx = st.number_input("Camera Index (0 = default)", min_value=0, max_value=5, value=0)
    video_path = int(cam_idx)
    st.info("📷 Webcam selected. Press **Start** to begin.")

st.markdown('<div class="div"></div>', unsafe_allow_html=True)


# ════════════════════════════════════════════════
#  CONTROLS
# ════════════════════════════════════════════════
btn_cols = st.columns([1, 1, 3])
with btn_cols[0]:
    start_btn = st.button("▶ Start Detection", use_container_width=True)
with btn_cols[1]:
    stop_btn  = st.button("⏹ Stop", use_container_width=True)

if stop_btn:
    st.session_state.stop_flag = True
    st.session_state.running   = False

# ════════════════════════════════════════════════
#  MAIN DETECTION LOOP
# ════════════════════════════════════════════════
if start_btn and video_path is not None:
    st.session_state.stop_flag      = False
    st.session_state.running        = True
    st.session_state.frame_count    = 0
    st.session_state.total_detected = 0
    st.session_state.fps_val        = 0.0
    st.session_state.saved_path     = None

    model = load_model(model_choice)
    tracker = CentroidTracker(max_disappeared=25, max_distance=80)
    trails  = defaultdict(lambda: [])   # tid → list of (x,y)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("❌ Could not open video source.")
        st.session_state.running = False
    else:
        fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # video writer setup
        writer = None
        out_path = None
        if save_video:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join(tempfile.gettempdir(), f"tracked_{ts}.mp4")
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(out_path, fourcc, 25, (fw, fh))

        frame_ph = st.empty()   # Streamlit video placeholder
        fps_smooth = 0.0
        prev_time  = time.time()

        while cap.isOpened() and not st.session_state.stop_flag:
            ret, frame = cap.read()
            if not ret:
                break

            # ── YOLOv8 inference ──
            results = model(frame, conf=conf_thresh, iou=iou_thresh,
                            max_det=max_det, verbose=False)[0]

            boxes_raw, labels, confs = [], [], []
            if results.boxes is not None and len(results.boxes):
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    conf   = float(box.conf[0])
                    label  = model.names[cls_id]
                    boxes_raw.append((x1, y1, x2, y2))
                    labels.append(label)
                    confs.append(conf)

            # ── tracking ──
            objects = tracker.update(boxes_raw)
            track_ids = []
            for (x1, y1, x2, y2) in boxes_raw:
                cx, cy = (x1+x2)//2, (y1+y2)//2
                best_id, best_d = 0, float('inf')
                for oid, cent in objects.items():
                    d = np.linalg.norm(np.array(cent) - np.array([cx, cy]))
                    if d < best_d:
                        best_d, best_id = d, oid
                track_ids.append(best_id)

            # ── draw ──
            frame = draw_detections(frame, boxes_raw, track_ids, labels, confs,
                                    show_conf, show_trail, trails)

            # FPS
            now = time.time()
            inst_fps = 1.0 / max(now - prev_time, 1e-6)
            fps_smooth = 0.85 * fps_smooth + 0.15 * inst_fps
            prev_time  = now

            frame = draw_hud(frame, fps_smooth, len(boxes_raw),
                             st.session_state.frame_count)

            # update stats
            st.session_state.frame_count    += 1
            st.session_state.total_detected += len(boxes_raw)
            st.session_state.fps_val         = round(fps_smooth, 1)

            # ── Streamlit display ──
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_ph.image(rgb, channels="RGB", use_container_width=True)

            # ── OpenCV window ──
            if show_opencv:
                cv2.imshow("Object Detection & Tracking — Press Q to quit", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # ── save frame ──
            if writer:
                writer.write(frame)

        cap.release()
        if show_opencv:
            cv2.destroyAllWindows()
        if writer:
            writer.release()
            st.session_state.saved_path = out_path

        st.session_state.running = False
        st.success("✅ Detection complete!")

        if st.session_state.saved_path and os.path.exists(st.session_state.saved_path):
            with open(st.session_state.saved_path, "rb") as vf:
                st.download_button(
                    "⬇️ Download Tracked Video",
                    data=vf,
                    file_name=os.path.basename(st.session_state.saved_path),
                    mime="video/mp4",
                    use_container_width=True
                )


elif start_btn and video_path is None:
    st.warning("⚠️ Please upload a video file or select webcam first.")


# ════════════════════════════════════════════════
#  EMPTY STATE
# ════════════════════════════════════════════════
if not st.session_state.running and st.session_state.frame_count == 0:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;border:1px dashed rgba(255,255,255,0.1);border-radius:14px;background:rgba(255,255,255,0.02);">
        <div style="font-size:3rem;margin-bottom:0.8rem;">🎯</div>
        <div style="font-family:'Syne',sans-serif;font-size:1.1rem;font-weight:700;color:#E2E6F0;">Ready to Detect</div>
        <div style="font-size:0.82rem;color:#5A6280;margin-top:0.4rem;">Upload a video or start webcam → Configure settings → Press Start</div>
    </div>
    """, unsafe_allow_html=True)