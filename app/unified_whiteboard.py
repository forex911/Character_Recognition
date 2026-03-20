import tkinter as tk
import numpy as np
import cv2
import os
import subprocess
import time
import sys

from inference.preprocess import preprocess
from inference.predict import Predictor
from utils.config import MODEL_PATH

DATA_DIR = "data/self_data"
BRUSH_SIZE = 10
PREDICT_DELAY = 0.4  # seconds


class UnifiedWhiteboard:
    def __init__(self):
        self.image = None
        self.predictor = None
        self.model_loaded = False

        self.strokes = []
        self.current_stroke = []
        self.last_predict_time = 0

        # ===== WINDOW (NOT FULLSCREEN) =====
        self.root = tk.Tk()
        self.root.title("Handwriting Recognition Tool")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)
        self.root.configure(bg="#1e1e1e")

        self.main = tk.Frame(self.root, bg="#1e1e1e")
        self.main.pack(fill="both", expand=True)

        self._create_canvas()
        self._create_panel()
        self._init_image()

        # ===== KEY BINDINGS =====
        self.root.bind("<Return>", lambda e: self.predict())
        self.root.bind("<BackSpace>", lambda e: self.undo())
        self.root.bind("<Control-z>", lambda e: self.undo())

    # ================= UI =================
    def _create_canvas(self):
        frame = tk.Frame(self.main, bg="black")
        frame.pack(side="left", fill="both", expand=True)

        self.canvas = tk.Canvas(
            frame, bg="black", cursor="cross", highlightthickness=0
        )
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.start_stroke)
        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.end_stroke)

    def _create_panel(self):
        panel = tk.Frame(self.main, width=320, bg="#252526")
        panel.pack(side="right", fill="y")
        panel.pack_propagate(False)

        self._title(panel, "CONTROL PANEL")

        self._btn(panel, "Load Model", self.load_model)
        self._btn(panel, "Predict (Enter)", self.predict)
        self._btn(panel, "Save Sample", self.save_sample)
        self._btn(panel, "Retrain Model", self.retrain_model)
        self._btn(panel, "Undo (Backspace)", self.undo)
        self._btn(panel, "Clear", self.clear)
        self._btn(panel, "Exit", self.root.destroy)

        tk.Label(panel, text="Label (A–Z)", fg="white", bg="#252526").pack(pady=(20, 5))
        self.label_entry = tk.Entry(panel, font=("Segoe UI", 14), width=5, justify="center")
        self.label_entry.pack()
        self.label_entry.insert(0, "A")

        self.pred_label = tk.Label(
            panel, text="Prediction: —",
            fg="#00ffcc", bg="#252526",
            font=("Segoe UI", 18, "bold")
        )
        self.pred_label.pack(pady=20)

        self.conf_bar = tk.Canvas(panel, height=18, bg="#1e1e1e", highlightthickness=0)
        self.conf_bar.pack(fill="x", padx=30)
        self.conf_rect = self.conf_bar.create_rectangle(0, 0, 0, 18, fill="#007acc")

        self.status = tk.Label(
            panel, text="Load model to begin",
            fg="#cccccc", bg="#252526",
            wraplength=260
        )
        self.status.pack(pady=15)

    def _btn(self, parent, text, cmd):
        tk.Button(
            parent, text=text, command=cmd,
            font=("Segoe UI", 11),
            bg="#3c3c3c", fg="white",
            relief="flat", height=2
        ).pack(fill="x", padx=30, pady=5)

    def _title(self, parent, text):
        tk.Label(
            parent, text=text,
            fg="white", bg="#252526",
            font=("Segoe UI", 15, "bold")
        ).pack(pady=15)

    def _init_image(self):
        self.image = np.zeros((600, 800), dtype=np.uint8)

    # ================= DRAW =================
    def start_stroke(self, event):
        self.current_stroke = []

    def draw(self, event):
        x, y = event.x, event.y
        self.current_stroke.append((x, y))

        cv2.circle(self.image, (x, y), BRUSH_SIZE, 255, -1)
        self.canvas.create_oval(
            x-BRUSH_SIZE, y-BRUSH_SIZE,
            x+BRUSH_SIZE, y+BRUSH_SIZE,
            fill="white", outline="white"
        )

        # Live prediction (debounced)
        if self.model_loaded and time.time() - self.last_predict_time > PREDICT_DELAY:
            self.live_predict()
            self.last_predict_time = time.time()

    def end_stroke(self, event):
        if self.current_stroke:
            self.strokes.append(self.current_stroke)

    # ================= MODEL =================
    def load_model(self):
        if not os.path.exists(MODEL_PATH):
            self._status("❌ Model not found. Train first.")
            return

        self.predictor = Predictor()
        self.model_loaded = True
        self._status("✅ Model loaded successfully")

    # ================= PREDICT =================
    def live_predict(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        processed = preprocess(img)
        pred, conf = self.predictor.predict_with_confidence(processed)
        self._update_prediction(pred, conf)

    def predict(self):
        if not self.model_loaded:
            self._status("⚠️ Load model first")
            return
        self.live_predict()
        self._status("Prediction updated")

    def _update_prediction(self, pred, conf):
        self.pred_label.config(text=f"Prediction: {pred}")
        self.conf_bar.coords(self.conf_rect, 0, 0, int(conf * 260), 18)

    # ================= SAVE =================
    def save_sample(self):
        label = self.label_entry.get().upper()
        if len(label) != 1 or not label.isalpha():
            self._status("❌ Invalid label")
            return

        save_dir = os.path.join(DATA_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        img = cv2.cvtColor(self.image, cv2.COLOR_GRAY2BGR)
        processed = preprocess(img)
        idx = len(os.listdir(save_dir))
        np.save(os.path.join(save_dir, f"{idx}.npy"), processed)

        self._status(f"💾 Saved sample for '{label}'")
        self.clear()

    # ================= RETRAIN =================
    def retrain_model(self):
        self._status("🔁 Retraining model… please wait")
        self.root.update()

        try:
            subprocess.run(
                [sys.executable, "-m", "model.train"],
                cwd=os.getcwd()
            )
            self._status("✅ Retraining complete. Reload model.")
            self.model_loaded = False
        except Exception as e:
            self._status(f"❌ Retraining failed: {e}")



    # ================= UNDO =================
    def undo(self):
        if not self.strokes:
            return

        self.strokes.pop()
        self.image.fill(0)
        self.canvas.delete("all")

        for stroke in self.strokes:
            for x, y in stroke:
                cv2.circle(self.image, (x, y), BRUSH_SIZE, 255, -1)
                self.canvas.create_oval(
                    x-BRUSH_SIZE, y-BRUSH_SIZE,
                    x+BRUSH_SIZE, y+BRUSH_SIZE,
                    fill="white", outline="white"
                )

    # ================= CLEAR =================
    def clear(self):
        self.image.fill(0)
        self.canvas.delete("all")
        self.strokes.clear()
        self.pred_label.config(text="Prediction: —")
        self.conf_bar.coords(self.conf_rect, 0, 0, 0, 18)

    def _status(self, msg):
        self.status.config(text=msg)


if __name__ == "__main__":
    UnifiedWhiteboard().root.mainloop()
