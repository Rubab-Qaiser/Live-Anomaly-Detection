import tkinter as tk
import tkinter.ttk as ttk
import cv2
import numpy as np
import PIL.Image, PIL.ImageTk
import time
import threading
import tensorflow as tf
import os
from collections import deque
from typing import Any, Optional, Tuple, cast
import importlib

try:
    plt = importlib.import_module("matplotlib.pyplot")
    backend_mod = importlib.import_module("matplotlib.backends.backend_tkagg")
    FigureCanvasTkAgg = backend_mod.FigureCanvasTkAgg
    try:
        plt.style.use("seaborn")
    except OSError:
        plt.style.use("default")
    HAS_MATPLOTLIB = True
except ImportError:
    plt = None
    FigureCanvasTkAgg = None
    HAS_MATPLOTLIB = False

BG_COLOR = "#f4f6fb"
CARD_COLOR = "#ffffff"
TEXT_COLOR = "#2f3b52"
ACTIVE_COLOR = "#1f76ff"
ANOMALY_COLOR = "#d64550"
NORMAL_COLOR = "#2ebf7f"

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# ================= CONFIG =================
CAMERA_URL = "http://192.168.0.110:8080/video"
FRAME_SIZE = 224
SEQUENCE_LENGTH = 8
THRESHOLD = 0.02
MODEL_PATH = "my_model.keras"

# ================= DETECTOR =================
class AnomalyDetector:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        self.buffer = []
        self.anomaly_count = 0
        self.latest_result = ("Collecting...", 0.0)
        self.error_history = deque(maxlen=50)  # For graph

    def detect(self, frame):
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, (FRAME_SIZE, FRAME_SIZE))
        frame_norm = frame_resized.astype(np.float32) / 255.0
        self.buffer.append(frame_norm)

        label = "Collecting..."
        error = 0.0

        if len(self.buffer) == SEQUENCE_LENGTH:
            sequence = np.expand_dims(np.array(self.buffer, dtype=np.float32), axis=0)
            reconstruction = self.model.predict(sequence, verbose=0)
            reconstructed_frame = reconstruction[0]

            target_frame = cv2.resize(
                frame_norm,
                (reconstructed_frame.shape[1], reconstructed_frame.shape[0]),
                interpolation=cv2.INTER_AREA,
            )

            error = float(np.mean((target_frame - reconstructed_frame) ** 2))
            self.error_history.append(error)

            if error > THRESHOLD:
                label = "ANOMALY"
                self.anomaly_count += 1
            else:
                label = "NORMAL"

            self.buffer.pop(0)

        self.latest_result = (label, error)
        return label, error

detector = AnomalyDetector()

# ================= VIDEO =================
class VideoCapture:
    def __init__(self, src: str):
        self.cap = cv2.VideoCapture(src)
        self.frame: Optional[np.ndarray] = None
        self.lock = threading.Lock()
        self.running = True
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self) -> None:
        while self.running:
            if not self.cap.isOpened():
                time.sleep(0.1)
                continue

            ret, frame = self.cap.read()
            if not ret or frame is None:
                time.sleep(0.03)
                continue

            with self.lock:
                self.frame = frame

        if self.cap.isOpened():
            self.cap.release()

    def get_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.frame is None:
                return False, None
            return True, self.frame.copy()

    def stop(self) -> None:
        self.running = False
        self.thread.join(timeout=1.0)
        if self.cap.isOpened():
            self.cap.release()

# ================= UI =================
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("Anomaly Detection Dashboard")
        self.root.configure(bg=BG_COLOR)
        self.root.geometry("1160x560")

        self.cap = VideoCapture(CAMERA_URL)
        self.prev_time = time.time()

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Layout
        self.left = tk.Frame(root, bg=BG_COLOR)
        self.left.pack(side="left", fill="both", expand=True, padx=16, pady=16)

        self.right = tk.Frame(root, bg=BG_COLOR)
        self.right.pack(side="right", fill="y", padx=16, pady=16)

        # Video Canvas
        self.canvas_border = tk.Frame(self.left, bg=CARD_COLOR, bd=0, relief="flat")
        self.canvas_border.pack(fill="both", expand=True)
        self.canvas = tk.Canvas(self.canvas_border, width=640, height=480, bg="black", bd=0, highlightthickness=0)
        self.canvas.pack(padx=12, pady=12)

        # Controls and stats
        self.title_label = tk.Label(self.right, text="Live Anomaly Detection", font=("Segoe UI", 18, "bold"), fg=TEXT_COLOR, bg=BG_COLOR)
        self.title_label.pack(anchor="w", pady=(0, 12))

        self.status_card = tk.Frame(self.right, bg=CARD_COLOR, bd=1, relief="solid")
        self.status_card.pack(fill="x", pady=(0, 12))
        self.status_label = tk.Label(self.status_card, text="Status: Collecting...", font=("Segoe UI", 14), fg=TEXT_COLOR, bg=CARD_COLOR, anchor="w")
        self.status_label.pack(fill="x", padx=12, pady=12)

        self.stats_card = tk.Frame(self.right, bg=CARD_COLOR, bd=1, relief="solid")
        self.stats_card.pack(fill="x", pady=(0, 12))
        self.anomaly_label = tk.Label(self.stats_card, text="Anomalies: 0", font=("Segoe UI", 14), fg=TEXT_COLOR, bg=CARD_COLOR, anchor="w")
        self.anomaly_label.pack(fill="x", padx=12, pady=(12, 6))
        self.error_label = tk.Label(self.stats_card, text="Error: 0.0000", font=("Segoe UI", 14), fg=TEXT_COLOR, bg=CARD_COLOR, anchor="w")
        self.error_label.pack(fill="x", padx=12, pady=(0, 6))
        self.fps_label = tk.Label(self.stats_card, text="FPS: 0", font=("Segoe UI", 14), fg=TEXT_COLOR, bg=CARD_COLOR, anchor="w")
        self.fps_label.pack(fill="x", padx=12, pady=(0, 12))

        self.separator = ttk.Separator(self.right, orient="horizontal")
        self.separator.pack(fill="x", pady=12)

        if HAS_MATPLOTLIB and plt is not None and FigureCanvasTkAgg is not None:
            self.fig, self.ax = plt.subplots(figsize=(4, 3))
            self.line, = self.ax.plot([], [], color=ACTIVE_COLOR, linewidth=2)
            self.threshold_line = self.ax.axhline(THRESHOLD, color=ANOMALY_COLOR, linestyle="--", linewidth=1)
            self.ax.set_facecolor("#f8f9fc")
            self.ax.set_title("Anomaly Score", color=TEXT_COLOR)
            self.ax.set_xlabel("Sequence index", color=TEXT_COLOR)
            self.ax.set_ylabel("Reconstruction error", color=TEXT_COLOR)
            self.ax.grid(True, color="#dde3f0", linestyle="--", linewidth=0.8)
            self.ax.set_xlim(0, 10)
            self.ax.set_ylim(0, max(THRESHOLD * 2, 0.1))
            self.ax.tick_params(axis="x", colors=TEXT_COLOR)
            self.ax.tick_params(axis="y", colors=TEXT_COLOR)
            self.ax.spines["bottom"].set_color(TEXT_COLOR)
            self.ax.spines["top"].set_color(TEXT_COLOR)
            self.ax.spines["left"].set_color(TEXT_COLOR)
            self.ax.spines["right"].set_color(TEXT_COLOR)
            self.ax.legend([self.line, self.threshold_line], ["Error", "Threshold"], frameon=False, labelcolor=TEXT_COLOR)

            self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.right)
            self.canvas_plot.get_tk_widget().pack(fill="both", expand=True)
        else:
            self.canvas_plot = None  # type: ignore[assignment]
            self.graph_label = tk.Label(
                self.right,
                text="Matplotlib not installed. Graph unavailable.",
                font=("Segoe UI", 12),
                fg=TEXT_COLOR,
                bg=BG_COLOR,
                wraplength=320,
                justify="left",
            )
            self.graph_label.pack(fill="x", pady=8)

        self.imgtk_ref = None
        self.update()

    def on_close(self) -> None:
        self.cap.stop()
        self.root.destroy()

    def update(self):
        ret, frame = self.cap.get_frame()

        if ret:
            frame = cast(np.ndarray, frame)
            label, error = detector.detect(frame)

            # Draw text
            color = (0,0,255) if label == "ANOMALY" else (0,255,0)
            cv2.putText(frame, f"{label} {error:.4f}", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            # FPS
            current = time.time()
            fps = 1/(current - self.prev_time)
            self.prev_time = current

            # Update labels
            self.status_label.config(text=f"Status: {label}")
            self.anomaly_label.config(text=f"Anomalies: {detector.anomaly_count}")
            self.error_label.config(text=f"Error: {error:.6f}")
            self.fps_label.config(text=f"FPS: {fps:.2f}")

            # Convert image
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = PIL.Image.fromarray(frame)
            imgtk = PIL.ImageTk.PhotoImage(img)

            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.imgtk_ref = imgtk

            # Update graph
            if HAS_MATPLOTLIB and self.canvas_plot is not None:
                error_values = list(detector.error_history)
                if error_values:
                    self.line.set_data(range(len(error_values)), error_values)
                    y_max = max(max(error_values) * 1.2, THRESHOLD * 2, 0.1)
                    self.ax.set_xlim(0, max(10, len(error_values)))
                    self.ax.set_ylim(0, y_max)
                else:
                    self.line.set_data([], [])
                    self.ax.set_xlim(0, 10)
                    self.ax.set_ylim(0, max(THRESHOLD * 2, 0.1))
                self.canvas_plot.draw()

        self.root.after(30, self.update)

# ================= RUN =================
root = tk.Tk()
app = App(root)
root.mainloop()