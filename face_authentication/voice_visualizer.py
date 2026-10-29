import math
import threading
import time
import tkinter as tk


class KreoVisualizer:
    def __init__(self):
        self.level = 0.0
        self.target_level = 0.0
        self.lock = threading.Lock()
        self.running = False
        self.thread = None
        self.root = None
        self.canvas = None
        self.phase = 0.0
        self.status = "KREO • STANDBY"

    def start(self):
        if self.running:
            return

        self.running = True
        self.thread = threading.Thread(
            target=self._run_window,
            daemon=True
        )
        self.thread.start()

    def set_level(self, rms):
        with self.lock:
            self.target_level = min(1.0, max(0.0, float(rms) * 8.0))

    def set_status(self, text):
        with self.lock:
            self.status = str(text)

    def close(self):
        self.running = False
        root = self.root
        if root is not None:
            try:
                root.after(0, root.destroy)
            except tk.TclError:
                pass

    def _run_window(self):
        try:
            root = tk.Tk()
            self.root = root
            root.title("KREO • Voice Interface")
            root.geometry("800x800")
            root.configure(bg="#070817")
            root.resizable(False, False)

            canvas = tk.Canvas(
                root,
                width=800,
                height=800,
                bg="#070817",
                highlightthickness=0
            )
            canvas.pack(fill="both", expand=True)
            self.canvas = canvas

            root.protocol("WM_DELETE_WINDOW", self.close)
            self._animate()
            root.mainloop()

        except Exception as exc:
            print("Kreo visualizer could not start:", exc)
        finally:
            self.running = False
            self.root = None
            self.canvas = None

    def _animate(self):
        if not self.running or self.canvas is None:
            return

        canvas = self.canvas
        canvas.delete("frame")

        with self.lock:
            self.level += (self.target_level - self.level) * 0.35
            level = self.level
            status = self.status

        cx, cy = 400, 350
        self.phase += 0.08 + level * 0.12

        for radius, color in [
            (210, "#11102D"),
            (185, "#17133C"),
            (160, "#21184D"),
        ]:
            canvas.create_oval(
                cx-radius, cy-radius, cx+radius, cy+radius,
                fill=color, outline="", tags="frame"
            )

        for i in range(96):
            angle = 2 * math.pi * i / 96
            radius = 135 + 3 * math.sin(self.phase + i * 0.16)
            x = cx + math.cos(angle) * radius
            y = cy + math.sin(angle) * radius
            canvas.create_oval(
                x-2, y-2, x+2, y+2,
                fill="#8D7AFF", outline="", tags="frame"
            )

        points = []
        for i in range(181):
            angle = 2 * math.pi * i / 180
            wave = (
                math.sin(angle * 7 + self.phase) * 0.35
                + math.sin(angle * 13 - self.phase * 1.4) * 0.25
                + math.sin(angle * 3 + self.phase * 0.7) * 0.40
            )
            radius = 148 + level * 42 * wave
            points.extend([
                cx + math.cos(angle) * radius,
                cy + math.sin(angle) * radius
            ])

        for width, color in [
            (20, "#282052"),
            (14, "#5742A2"),
            (8, "#A18AFF"),
            (4, "#FFFFFF"),
        ]:
            canvas.create_line(
                points,
                fill=color,
                width=width,
                smooth=True,
                splinesteps=24,
                tags="frame"
            )

        waveform = []
        for i in range(61):
            x = cx - 95 + i * (190 / 60)
            envelope = math.sin(math.pi * i / 60)
            y = cy + math.sin(i * 0.42 + self.phase * 2) * (
                3 + level * 50 * envelope
            )
            waveform.extend([x, y])

        canvas.create_line(
            waveform,
            fill="#E7E0FF",
            width=3.5,
            smooth=True,
            tags="frame"
        )

        canvas.create_text(
            cx, 610,
            text="K R E O",
            fill="#F0ECFF",
            font=("Segoe UI", 28, "bold"),
            tags="frame"
        )

        canvas.create_text(
            cx, 655,
            text=status,
            fill="#A69BCF",
            font=("Segoe UI", 14),
            tags="frame"
        )

        root = self.root
        if root is not None and self.running:
            root.after(25, self._animate)


visualizer = KreoVisualizer()