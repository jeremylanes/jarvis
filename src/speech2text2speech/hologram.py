"""
    Jarvis hologram visual interface.

    This module provides the PySide6-based animated hologram overlay displaying
    avengers-like circular structures, circuit board paths, and a glowing core.
    It runs in a background thread to prevent blocking main pipelines and can
    be attached directly to a voice assistant speak loop.

    - :class:`JarvisWidget` – QWidget drawing the visual hologram components.
    - :class:`JarvisHologram` – Dataclass manager for launching and controlling the widget.
    - :func:`attach_to_tts` – Function to hook a Speech-to-Text pipeline with the hologram.

    **Usage example**::

        from speech2text2speech.hologram import JarvisHologram
        import time

        hologram = JarvisHologram()
        hologram.start()
        hologram.set_speaking(True)
        hologram.push_amplitude(0.5)
        time.sleep(2)
        hologram.stop()
"""

import io
import math
import random
import threading
import time
import wave
from dataclasses import dataclass, field

import numpy as np
import sounddevice as sd
import soundfile as sf
from PySide6.QtCore import QObject, QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (QBrush, QColor, QFont, QPainter, QPainterPath, QPen,
                           QRadialGradient, QScreen)
from PySide6.QtWidgets import QApplication, QWidget

# Dimensions and visual parameters
W = H = 420
RNG = random.Random(42)
SPHERE_R = 130
FPS = 40


def lerp(a: float, b: float, t: float) -> float:
    """
        Linearly interpolate between two float values.

        :param a: Start value.
        :type a: float
        :param b: End value.
        :type b: float
        :param t: Interpolation factor.
        :type t: float
        :return: Interpolated float value.
        :rtype: float

        **Usage example**::

            res = lerp(0.0, 10.0, 0.5)
    """
    return a + (b - a) * max(0.0, min(1.0, t))


def lc(c1: QColor, c2: QColor, t: float) -> QColor:
    """
        Linearly interpolate between two QColors.

        :param c1: Start color.
        :type c1: PySide6.QtGui.QColor
        :param c2: End color.
        :type c2: PySide6.QtGui.QColor
        :param t: Interpolation factor.
        :type t: float
        :return: Interpolated QColor.
        :rtype: PySide6.QtGui.QColor

        **Usage example**::

            res = lc(QColor(0,0,0), QColor(255,255,255), 0.5)
    """
    t = max(0.0, min(1.0, t))
    return QColor(
        int(lerp(c1.red(), c2.red(), t)),
        int(lerp(c1.green(), c2.green(), t)),
        int(lerp(c1.blue(), c2.blue(), t)),
        int(lerp(c1.alpha(), c2.alpha(), t)),
    )


# ── Thread-safe Signals ────────────────────────────────────────────────────────
class _Bridge(QObject):
    """
        Signal bridge for thread-safe UI closure.

        :cvar do_close: Signal to request closing the widget.
        :type do_close: PySide6.QtCore.Signal

        **Usage example**::

            bridge = _Bridge()
            bridge.do_close.emit()
    """

    do_close = Signal()


# ── Widget ─────────────────────────────────────────────────────────────────────
class JarvisWidget(QWidget):
    """
        Floating PySide6 widget rendering the Jarvis hologram.

        A frameless, translucent, always-on-top window displaying animated
        3D rings, circuit-board arcs, and a pulsing energy core.

        :param _t: Total elapsed time in seconds.
        :type _t: float
        :param _energy: Smoothly interpolated speech energy.
        :type _energy: float
        :param _amp: Current raw sound amplitude.
        :type _amp: float
        :param _speaking: Speaks state indicator.
        :type _speaking: bool
        :param _lock: Thread lock protecting amplitude.
        :type _lock: threading.Lock
        :param _rings: Configured holographic rings data.
        :type _rings: list
        :param _ring_radii: Concentric radii for rings.
        :type _ring_radii: list
        :param _arcs: Procedural circuit board paths.
        :type _arcs: list
        :param _arc_regen_t: Arc regeneration countdown.
        :type _arc_regen_t: float
        :param _bridge: Thread-safe signal bridge.
        :type _bridge: _Bridge
        :param _timer: Animation frame timer.
        :type _timer: PySide6.QtCore.QTimer

        **Usage example**::

            widget = JarvisWidget()
            widget.show()
    """

    def __init__(self):
        """
            Initialize UI window flags, geometry, and animation elements.

            **Usage example**::

                widget = JarvisWidget()
        """
        super().__init__()

        # Window: frameless, always-on-top, not in taskbar
        flags = (
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.WindowDoesNotAcceptFocus
            | Qt.X11BypassWindowManagerHint
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)
        self.resize(W, H)

        self._position_top_right()

        self._t = 0.0
        self._energy = 0.0
        self._amp = 0.0
        self._speaking = False
        self._lock = threading.Lock()

        # Rings: [angle, speed_idle, speed_mult, reverse, tilt_y_ratio]
        self._rings = [
            [0.0, 0.20, 5.0, False, 0.38],
            [0.0, 0.35, 6.5, True, 0.18],
            [0.0, 0.15, 4.0, False, 0.52],
        ]
        self._ring_radii = [155, 168, 182]

        self._arcs = self._gen_arcs(22)
        self._arc_regen_t = 0.0

        self._bridge = _Bridge()
        self._bridge.do_close.connect(self._on_close)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // FPS)

    def _position_top_right(self):
        """
            Move widget to the top-right corner of the screen.

            **Usage example**::

                widget._position_top_right()
        """
        screen: QScreen = QApplication.primaryScreen()
        geom = screen.availableGeometry()
        margin = 12
        tx = geom.right() - W - margin
        ty = geom.top() + margin
        if self.x() != tx or self.y() != ty:
            self.move(tx, ty)

    def push_amplitude(self, rms: float):
        """
            Update audio RMS amplitude in a thread-safe way.

            :param rms: Audio RMS value.
            :type rms: float

            **Usage example**::

                widget.push_amplitude(0.15)
        """
        with self._lock:
            self._amp = max(0.0, min(1.0, rms))

    def set_speaking(self, v: bool):
        """
            Update speaking state in a thread-safe way.

            :param v: True if speaking.
            :type v: bool

            **Usage example**::

                widget.set_speaking(True)
        """
        with self._lock:
            self._speaking = v

    def request_close(self):
        """
            Emit close request signal from any thread.

            **Usage example**::

                widget.request_close()
        """
        self._bridge.do_close.emit()

    @Slot()
    def _on_close(self):
        """
            Terminate timer and close QWidget (runs on main thread).

            **Usage example**::

                widget._on_close()
        """
        self._timer.stop()
        self.close()

    def _gen_arcs(self, n: int = 22) -> list:
        """
            Generate a set of randomized circuit paths.

            :param n: Number of arcs to create.
            :type n: int
            :return: Generated arcs dataset.
            :rtype: list

            **Usage example**::

                arcs = widget._gen_arcs(15)
        """
        arcs = []
        r = random.Random(int(time.time() * 100) % 99999)
        for _ in range(n):
            start_r = r.uniform(22, SPHERE_R - 15)
            start_a = r.uniform(0, math.pi * 2)
            pts = [(start_r, start_a)]
            cr, ca = start_r, start_a
            for _ in range(r.randint(2, 6)):
                if r.random() < 0.4:
                    cr = max(12, min(SPHERE_R - 8, cr + r.uniform(-35, 35)))
                else:
                    ca += r.uniform(-0.7, 0.7)
                pts.append((cr, ca))
            arcs.append({
                'pts': pts,
                'phase': r.uniform(0, math.pi * 2),
                'speed': r.uniform(-0.4, 0.4),
                'base_alpha': r.uniform(0.25, 0.85),
                'width': r.uniform(0.8, 1.6),
            })
        return arcs

    def _tick(self):
        """
            Perform animation updates and request repaint.

            **Usage example**::

                # Automatically scheduled by QTimer
                widget._tick()
        """
        dt = 1.0 / FPS
        self._t += dt

        with self._lock:
            raw = self._amp
            self._amp *= 0.70
            speaking = self._speaking

        base = 0.10 if speaking else 0.0
        target = max(base, raw)
        rate = 0.30 if target > self._energy else 0.045
        self._energy += (target - self._energy) * rate
        e = self._energy

        for ring in self._rings:
            spd = ring[1] * (1.0 + (ring[2] - 1.0) * e)
            sign = -1 if ring[3] else 1
            ring[0] = (ring[0] + spd * sign) % 360

        self._arc_regen_t += dt
        if self._arc_regen_t > 1.5:
            self._arc_regen_t = 0.0
            new = self._gen_arcs(6)
            idxs = random.sample(range(len(self._arcs)), min(5, len(self._arcs)))
            for j, idx in enumerate(idxs):
                self._arcs[idx] = new[j % len(new)]

        self._position_top_right()
        self.raise_()
        self.update()

    def paintEvent(self, _):
        """
            Handle custom widget painting using QPainter.

            :param _: Paint event object.
            :type _: PySide6.QtGui.QPaintEvent

            **Usage example**::

                # Scheduled via update()
                widget.update()
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        e = self._energy
        t = self._t
        cx = W // 2
        cy = H // 2

        # 1. Outer Spherical Glow
        boost = 1 + e * 0.20
        r = SPHERE_R * boost

        g = QRadialGradient(cx, cy, r * 1.35)
        g.setColorAt(0.00, QColor(100, 28, 0, int(50 + e * 90)))
        g.setColorAt(0.45, QColor(60, 12, 0, int(25 + e * 55)))
        g.setColorAt(1.00, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        rr = r * 1.35
        p.drawEllipse(QRectF(cx - rr, cy - rr, rr * 2, rr * 2))

        g2 = QRadialGradient(cx, cy, r)
        g2.setColorAt(0.00, QColor(
            min(255, int(200 + e * 55)), min(255, int(95 + e * 70)), int(8 + e * 18), int(210 + e * 45)
        ))
        g2.setColorAt(0.40, QColor(min(255, int(170 + e * 60)), int(55 + e * 45), 0, int(160 + e * 60)))
        g2.setColorAt(0.72, QColor(min(255, int(130 + e * 70)), int(30 + e * 30), 0, int(100 + e * 80)))
        g2.setColorAt(1.00, QColor(50, 8, 0, 0))
        p.setBrush(QBrush(g2))
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # Surface veins
        n_veins = 6 + int(e * 8)
        for i in range(n_veins):
            ph = t * 0.55 + i * (math.pi * 2 / n_veins)
            vr = 30 + math.sin(ph * 1.3 + i) * 25
            va = math.degrees(ph) % 360
            ext = 35 + math.sin(ph * 2.2) * 18
            a = max(0, min(255, int((0.10 + e * 0.30 + 0.08 * math.sin(ph * 3.5)) * 255)))
            col = QColor(255, int(95 + e * 75), 0, a)
            pen = QPen(col, 1.3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(cx - vr, cy - vr, vr * 2, vr * 2), int(va * 16), int(ext * 16))

        # 2. PCB Arcs
        p.setBrush(Qt.NoBrush)
        for arc in self._arcs:
            ph = arc['phase'] + t * arc['speed']
            vis = 0.55 + 0.45 * math.sin(ph * 2.1)
            ity = arc['base_alpha'] * (0.25 + e * 0.75) * vis
            if ity < 0.04:
                continue

            pts = []
            for r_val, a_val in arc['pts']:
                a2 = a_val + t * 0.04
                pts.append(QPointF(cx + r_val * math.cos(a2), cy + r_val * math.sin(a2)))

            if len(pts) < 2:
                continue

            alpha = min(255, int(ity * 255))
            red = min(255, int(210 + e * 45))
            green = min(255, int(85 + e * 85))
            col = QColor(red, green, 0, alpha)

            pen = QPen(col, arc['width'] * (1 + e * 0.6))
            pen.setCapStyle(Qt.SquareCap)
            pen.setJoinStyle(Qt.MiterJoin)
            p.setPen(pen)

            path = QPainterPath()
            path.moveTo(pts[0])
            for pt in pts[1:]:
                path.lineTo(pt)
            p.drawPath(path)

            # Nodes at junctions
            node_col = QColor(min(255, red + 25), min(255, green + 35), 10, alpha)
            p.setBrush(QBrush(node_col))
            p.setPen(Qt.NoPen)
            for pt in pts[::2]:
                nr = 1.8 + e * 1.8
                p.drawEllipse(pt, nr, nr)
            p.setBrush(Qt.NoBrush)

        # 3. 3D Concentric Rings
        p.setBrush(Qt.NoBrush)
        dash_configs = [(10, 14), (8, 10), (6, 8)]

        for i, ring in enumerate(self._rings):
            rx = self._ring_radii[i]
            ry = rx * ring[4]
            angle = ring[0]
            dash_on, dash_off = dash_configs[i]
            cycle = dash_on + dash_off

            bright = 0.35 + e * 0.60
            alpha = int((0.45 + e * 0.45) * 255)
            r_val = min(255, int(185 + e * 70))
            g_val = min(255, int(75 + e * 85))
            col_hi = QColor(r_val, g_val, 0, alpha)
            col_lo = QColor(70, 18, 0, int(alpha * 0.25))

            for start in range(0, 360, cycle):
                a_start = (start + angle) % 360
                span = dash_on

                mid_deg = (a_start + span / 2) % 360
                mid_rad = math.radians(mid_deg)
                depth = 0.5 + 0.5 * math.sin(mid_rad)

                col = lc(col_lo, col_hi, depth * bright)
                w = 1.0 + depth * (1.0 + e * 2.0)
                pen = QPen(col, w)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)
                p.drawArc(
                    QRectF(cx - rx, cy - ry, rx * 2, ry * 2),
                    int(a_start * 16),
                    int(span * 16),
                )

            # Ticks
            n_ticks = 8 + i * 4
            for k in range(n_ticks):
                deg = (k * 360 / n_ticks + angle) % 360
                rad = math.radians(deg)
                ex = cx + rx * math.cos(rad)
                ey = cy + ry * math.sin(rad)
                dx, dy = cx - ex, cy - ey
                ln = math.sqrt(dx * dx + dy * dy) or 1
                tl = 5 + e * 4
                depth = 0.5 + 0.5 * math.sin(rad)
                tc = lc(col_lo, col_hi, depth)
                p.setPen(QPen(tc, 1.0))
                p.drawLine(QPointF(ex, ey), QPointF(ex + dx / ln * tl, ey + dy / ln * tl))

        # 4. Central Pulse Core
        pulse = 0.5 + 0.5 * math.sin(t * 7.5)
        ep = min(1.0, e + pulse * 0.07)

        layers = [
            (52, QColor(110, 28, 0, int(180 * (0.35 + ep * 0.65)))),
            (40, QColor(185, 65, 0, int(210 * (0.45 + ep * 0.55)))),
            (27, QColor(245, 115, 0, int(225 * (0.55 + ep * 0.45)))),
            (17, QColor(255, 185, 45, int(240 * (0.65 + ep * 0.35)))),
            (9, QColor(255, 235, 125, int(255 * (0.75 + ep * 0.25)))),
            (4, QColor(255, 255, 220, 255)),
        ]
        p.setPen(Qt.NoPen)
        for r_val, col in layers:
            ri = r_val * (1.0 + e * 0.22 + pulse * 0.04)
            g = QRadialGradient(cx, cy, ri)
            g.setColorAt(0.0, col)
            g.setColorAt(1.0, QColor(col.red(), col.green(), col.blue(), 0))
            p.setBrush(QBrush(g))
            p.drawEllipse(QRectF(cx - ri, cy - ri, ri * 2, ri * 2))

        hr = 57 * (1 + e * 0.15)
        hc = QColor(255, int(105 + e * 75), 0, int(90 + e * 130))
        p.setPen(QPen(hc, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(cx - hr, cy - hr, hr * 2, hr * 2))

        # 5. Label
        a = int((0.28 + e * 0.60) * 255)
        col = QColor(255, int(75 + e * 85), 0, a)
        p.setPen(col)
        p.setFont(QFont("Courier", 8, QFont.Bold))
        p.drawText(QRectF(0, H - 24, W, 18), Qt.AlignHCenter, "J . A . R . V . I . S .")

        p.end()


# ── Controller ─────────────────────────────────────────────────────────────────
@dataclass
class JarvisHologram:
    """
        Controller managing the JarvisWidget life cycle.

        Launches the Qt event loop in a background thread and delegates
        speaking states and sound amplitudes to the widget.

        :param _widget: Underlying Qt Widget instance.
        :type _widget: JarvisWidget
        :param _app: PySide6 application singleton.
        :type _app: PySide6.QtWidgets.QApplication
        :param _thread: Background execution thread.
        :type _thread: threading.Thread
        :param _ready: Threading sync barrier.
        :type _ready: threading.Event

        **Usage example**::

            h = JarvisHologram()
            h.start()
    """

    _widget: object = field(default=None, init=False)
    _app: object = field(default=None, init=False)
    _thread: object = field(default=None, init=False)
    _ready: object = field(default=None, init=False)

    def __post_init__(self):
        """
            Setup thread synchronization structures.

            **Usage example**::

                # Implicitly called
                h = JarvisHologram()
        """
        self._ready = threading.Event()

    def start(self):
        """
            Start background thread and wait for widget readiness.

            **Usage example**::

                h.start()
        """
        self._thread = threading.Thread(target=self._run_qt, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        """
            Safely close widget and terminate application.

            **Usage example**::

                h.stop()
        """
        if self._widget:
            self._widget.request_close()

    def set_speaking(self, v: bool):
        """
            Proxy speaking state update to the widget.

            :param v: True if speaking.
            :type v: bool

            **Usage example**::

                h.set_speaking(True)
        """
        if self._widget:
            self._widget.set_speaking(v)

    def push_amplitude(self, rms: float):
        """
            Proxy audio amplitude updates to the widget.

            :param rms: Root Mean Square value.
            :type rms: float

            **Usage example**::

                h.push_amplitude(0.25)
        """
        if self._widget:
            self._widget.push_amplitude(rms)

    def _run_qt(self):
        """
            Start Qt Application loop in the background.

            **Usage example**::

                # Executed in daemon thread
                h._run_qt()
        """
        import os
        import sys
        os.environ["QT_QPA_PLATFORM"] = "xcb"
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._widget = JarvisWidget()
        self._widget.show()
        self._ready.set()
        self._app.exec()


# ── TTS Integration ────────────────────────────────────────────────────────────
def attach_to_tts(hologram: JarvisHologram, tts_instance):
    """
        Monkey-patch TextToSpeech speak loop to sync with hologram.

        :param hologram: The target controller.
        :type hologram: JarvisHologram
        :param tts_instance: TextToSpeech pipeline instance.
        :type tts_instance: TextToSpeech

        **Usage example**::

            attach_to_tts(hologram, tts)
    """
    def _patched_speak(text: str):
        hologram.set_speaking(True)
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                tts_instance.voice.synthesize_wav(text, wf)
            buf.seek(0)
            data, sr = sf.read(buf, dtype="float32")

            block = max(1, int(sr / FPS))
            words = text.split()
            duration = len(data) / sr
            delay = duration / max(len(words), 1)

            def _words():
                for w in words:
                    print(w, end=" ", flush=True)
                    time.sleep(delay)
                print()
            threading.Thread(target=_words, daemon=True).start()

            sd.play(data, sr)
            idx = 0
            while idx < len(data):
                chunk = data[idx:idx + block]
                if len(chunk):
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    hologram.push_amplitude(min(1.0, rms / 0.12))
                idx += block
                time.sleep(1.0 / FPS)
            sd.wait()
        finally:
            hologram.set_speaking(False)

    tts_instance.speak = _patched_speak


# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h = JarvisHologram()
    h.start()
    print("Idle 3s...")
    time.sleep(3)
    print("Simulation parole...")
    h.set_speaking(True)
    for idx_val in range(280):
        h.push_amplitude(abs(math.sin(idx_val * 0.22)) * 0.9 + random.uniform(0, 0.08))
        time.sleep(0.025)
    h.set_speaking(False)
    print("Idle 3s...")
    time.sleep(3)
    h.stop()
