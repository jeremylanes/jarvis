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
        Linearly interpolate between two values.

        Computes the linear interpolation formula: a + (b - a) * t, where t is clamped
        between 0.0 and 1.0.

        :param a: The start value.
        :type a: float
        :param b: The end value.
        :type b: float
        :param t: The interpolation factor.
        :type t: float
        :return: The interpolated value.
        :rtype: float

        **Usage example**::

            val = lerp(10.0, 20.0, 0.5)
            # Returns 15.0
    """
    return a + (b - a) * max(0.0, min(1.0, t))


def lc(c1: QColor, c2: QColor, t: float) -> QColor:
    """
        Linearly interpolate between two QColor colors.

        Clamps the interpolation factor between 0.0 and 1.0, then interpolates red,
        green, blue, and alpha channels individually.

        :param c1: The start color.
        :type c1: PySide6.QtGui.QColor
        :param c2: The end color.
        :type c2: PySide6.QtGui.QColor
        :param t: The interpolation factor.
        :type t: float
        :return: The interpolated color.
        :rtype: PySide6.QtGui.QColor

        **Usage example**::

            from PySide6.QtGui import QColor
            color = lc(QColor(255, 0, 0), QColor(0, 0, 255), 0.5)
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
        Signal bridge to invoke Qt slots from non-Qt threads.

        Provides custom Qt signals that can be connected to slots inside the
        main GUI thread, enabling thread-safe UI operations from background threads.

        :cvar do_close: Signal emitted to trigger widget closure.
        :type do_close: PySide6.QtCore.Signal

        **Usage example**::

            bridge = _Bridge()
            bridge.do_close.emit()
    """

    do_close = Signal()


# ── Widget ─────────────────────────────────────────────────────────────────────
class JarvisWidget(QWidget):
    """
        Floating PySide6 widget rendering the animated holographic interface.

        A borderless, translucent, always-on-top window positioned at the
        top-right of the primary screen. Draws dynamic 3D-like rotating rings,
        holographic circuit-board arcs, and a pulsing energy core matching speech amplitude.

        :param _t: Total elapsed time in seconds for oscillation math.
        :type _t: float
        :param _energy: Smoothly interpolated speech energy.
        :type _energy: float
        :param _amp: Current raw sound amplitude.
        :type _amp: float
        :param _speaking: Whether the assistant is currently speaking.
        :type _speaking: bool
        :param _lock: Thread-safe lock protecting audio amplitude state.
        :type _lock: threading.Lock
        :param _rings: Configured holographic rings with speed and tilt attributes.
        :type _rings: list
        :param _ring_radii: Radii of the three main holographic rings.
        :type _ring_radii: list
        :param _arcs: Randomly generated circuit board arcs.
        :type _arcs: list
        :param _arc_regen_t: Time counter to control the periodic regeneration of arcs.
        :type _arc_regen_t: float
        :param _bridge: Signal bridge for thread-safe cross-thread UI controls.
        :type _bridge: _Bridge
        :param _timer: Main animation timer driving the rendering updates.
        :type _timer: PySide6.QtCore.QTimer

        **Usage example**::

            import sys
            from PySide6.QtWidgets import QApplication
            app = QApplication(sys.argv)
            widget = JarvisWidget()
            widget.show()
    """

    def __init__(self):
        """
            Initialize the widget, flags, positioning, and animation components.

            Sets frameless window flags, translucency, retrieves primary screen
            geometry to position the widget in the top-right corner, initializes
            holographic rings and arcs, sets up the cross-thread signal bridge,
            and starts the main GUI update timer at the defined FPS rate.

            **Usage example**::

                # Instantiated inside the main loop or JarvisHologram
                widget = JarvisWidget()
        """
        super().__init__()

        # ── Window: frameless, always-on-top, not in taskbar
        flags = (
            Qt.FramelessWindowHint
            | Qt.WindowStaysOnTopHint
            | Qt.Tool
            | Qt.X11BypassWindowManagerHint   # KDE/X11: force foreground
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.setFocusPolicy(Qt.NoFocus)       # does not steal focus
        self.resize(W, H)

        # ── Positioning in the top-right corner
        self._position_top_right()

        # ── Animation state
        self._t = 0.0
        self._energy = 0.0
        self._amp = 0.0
        self._speaking = False
        self._lock = threading.Lock()

        # Rings: [angle, speed_idle, speed_mult, reverse, tilt_y_ratio]
        self._rings = [
            [0.0, 0.20, 5.0, False, 0.38],   # almost circular
            [0.0, 0.35, 6.5, True, 0.18],   # very tilted
            [0.0, 0.15, 4.0, False, 0.52],   # wide, less tilted
        ]
        self._ring_radii = [155, 168, 182]

        # Printed circuit board arcs
        self._arcs = self._gen_arcs(22)
        self._arc_regen_t = 0.0

        # ── Bridge for thread-safe stop
        self._bridge = _Bridge()
        self._bridge.do_close.connect(self._on_close)

        # ── Timer in the Qt thread
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(1000 // FPS)

    def _position_top_right(self):
        """
            Position the widget at the top-right corner of the primary screen.

            Retrieves the available screen geometry and moves the widget,
            applying a margin.

            **Usage example**::

                widget._position_top_right()
        """
        screen: QScreen = QApplication.primaryScreen()
        geom = screen.availableGeometry()
        margin = 12
        self.move(geom.right() - W - margin, geom.top() + margin)

    # ── Thread-safe API ────────────────────────────────────────────────────────
    def push_amplitude(self, rms: float):
        """
            Update the audio amplitude in a thread-safe manner.

            Clamps the value between 0.0 and 1.0 and saves it to the internal
            amplitude state.

            :param rms: The Root Mean Square value of the current audio chunk.
            :type rms: float

            **Usage example**::

                widget.push_amplitude(0.05)
        """
        with self._lock:
            self._amp = max(0.0, min(1.0, rms))

    def set_speaking(self, v: bool):
        """
            Set the speaking state of the widget in a thread-safe manner.

            Controls whether the hologram animates in idle or speaking mode.

            :param v: True if speaking, False otherwise.
            :type v: bool

            **Usage example**::

                widget.set_speaking(True)
        """
        with self._lock:
            self._speaking = v

    def request_close(self):
        """
            Request the widget to close from any thread.

            Emits the `do_close` signal via the bridge to safely close the widget
            within the Qt main event loop.

            **Usage example**::

                widget.request_close()
        """
        self._bridge.do_close.emit()

    @Slot()
    def _on_close(self):
        """
            Handle widget closure within the main Qt thread.

            Stops the animation timer and closes the window.

            **Usage example**::

                widget._on_close()
        """
        self._timer.stop()
        self.close()

    # ── Arc Generation ─────────────────────────────────────────────────────────
    def _gen_arcs(self, n: int = 22) -> list:
        """
            Generate random holographic circuit board arcs.

            Creates coordinates and parameters for circuit-like geometric shapes
            with custom speed, transparency, and line width.

            :param n: Number of arcs to generate.
            :type n: int
            :return: A list of generated arc dictionaries.
            :rtype: list

            **Usage example**::

                arcs = widget._gen_arcs(10)
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

    # ── Tick ───────────────────────────────────────────────────────────────────
    def _tick(self):
        """
            Update animation variables and trigger a repaint.

            Computes elapsed time, updates rotating ring angles based on current energy,
            smoothly interpolates energy targets, and regenerates expired arcs.
            Also ensures the window stays on top of other windows.

            **Usage example**::

                # Automatically called by the QTimer; not called directly.
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

        # Always stay on top (KDE Wayland might forget it)
        self.raise_()
        self.update()

    # ── Paint ──────────────────────────────────────────────────────────────────
    def paintEvent(self, _):
        """
            Qt paint event handler to draw the hologram.

            Clears the canvas and renders the outer glow, circuit board arcs,
            3D rotating rings, central core, and bottom text label.

            :param _: The paint event object (unused).
            :type _: PySide6.QtGui.QPaintEvent

            **Usage example**::

                # Triggered by calling update(); not called directly.
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

        self._draw_glow(p, cx, cy, e)
        self._draw_arcs(p, cx, cy, e, t)
        self._draw_rings(p, cx, cy, e)
        self._draw_core(p, cx, cy, e, t)
        self._draw_label(p, e)
        p.end()

    # ── Spherical Glow ─────────────────────────────────────────────────────────
    def _draw_glow(self, p: QPainter, cx: float, cy: float, e: float):
        """
            Draw the outer radial glow and interior veins.

            Renders a diffuse outer halo, a glowing main body, and small glowing
            surface veins using radial gradients.

            :param p: The active painter.
            :type p: PySide6.QtGui.QPainter
            :param cx: Center X coordinate.
            :type cx: float
            :param cy: Center Y coordinate.
            :type cy: float
            :param e: Current energy value.
            :type e: float

            **Usage example**::

                # Called inside paintEvent.
                widget._draw_glow(painter, 210, 210, 0.5)
        """
        boost = 1 + e * 0.20
        r = SPHERE_R * boost

        # Very diffuse outer halo
        g = QRadialGradient(cx, cy, r * 1.35)
        g.setColorAt(0.00, QColor(100, 28, 0, int(50 + e * 90)))
        g.setColorAt(0.45, QColor(60, 12, 0, int(25 + e * 55)))
        g.setColorAt(1.00, QColor(0, 0, 0, 0))
        p.setBrush(QBrush(g))
        p.setPen(Qt.NoPen)
        rr = r * 1.35
        p.drawEllipse(QRectF(cx - rr, cy - rr, rr * 2, rr * 2))

        # Main body
        g2 = QRadialGradient(cx, cy, r)
        g2.setColorAt(0.00, QColor(
            min(255, int(200 + e * 55)), min(255, int(95 + e * 70)), int(8 + e * 18), int(210 + e * 45)
        ))
        g2.setColorAt(0.40, QColor(min(255, int(170 + e * 60)), int(55 + e * 45), 0, int(160 + e * 60)))
        g2.setColorAt(0.72, QColor(min(255, int(130 + e * 70)), int(30 + e * 30), 0, int(100 + e * 80)))
        g2.setColorAt(1.00, QColor(50, 8, 0, 0))
        p.setBrush(QBrush(g2))
        p.drawEllipse(QRectF(cx - r, cy - r, r * 2, r * 2))

        # Surface veins (animated internal arcs)
        n = 6 + int(e * 8)
        for i in range(n):
            ph = self._t * 0.55 + i * (math.pi * 2 / n)
            vr = 30 + math.sin(ph * 1.3 + i) * 25
            va = math.degrees(ph) % 360
            ext = 35 + math.sin(ph * 2.2) * 18
            a = max(0, min(255, int((0.10 + e * 0.30 + 0.08 * math.sin(ph * 3.5)) * 255)))
            col = QColor(255, int(95 + e * 75), 0, a)
            pen = QPen(col, 1.3)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(QRectF(cx - vr, cy - vr, vr * 2, vr * 2),
                      int(va * 16), int(ext * 16))

    # ── PCB Arcs ───────────────────────────────────────────────────────────────
    def _draw_arcs(self, p: QPainter, cx: float, cy: float, e: float, t: float):
        """
            Draw the circuit board arcs and connection nodes.

            Renders segmented paths with node indicators at vertex positions,
            applying alpha oscillation based on phase and current speech energy.

            :param p: The active painter.
            :type p: PySide6.QtGui.QPainter
            :param cx: Center X coordinate.
            :type cx: float
            :param cy: Center Y coordinate.
            :type cy: float
            :param e: Current energy value.
            :type e: float
            :param t: Elapsed time in seconds.
            :type t: float

            **Usage example**::

                # Called inside paintEvent.
                widget._draw_arcs(painter, 210, 210, 0.5, 1.2)
        """
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

    # ── 3D Rings ───────────────────────────────────────────────────────────────
    def _draw_rings(self, p: QPainter, cx: float, cy: float, e: float):
        """
            Draw the 3D tilted concentric rings.

            Simulates perspective tilt by drawing ellipses with depth-based shading,
            brighter on the closer side, and adding ticks along the rings.

            :param p: The active painter.
            :type p: PySide6.QtGui.QPainter
            :param cx: Center X coordinate.
            :type cx: float
            :param cy: Center Y coordinate.
            :type cy: float
            :param e: Current energy value.
            :type e: float

            **Usage example**::

                # Called inside paintEvent.
                widget._draw_rings(painter, 210, 210, 0.5)
        """
        p.setBrush(Qt.NoBrush)

        dash_configs = [(10, 14), (8, 10), (6, 8)]

        for i, ring in enumerate(self._rings):
            rx = self._ring_radii[i]
            ry = rx * ring[4]          # tilt_y_ratio → flattening
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

                # Depth = Y position on the ellipse
                mid_deg = (a_start + span / 2) % 360
                mid_rad = math.radians(mid_deg)
                # Positive sine = bottom = "front" (perspective convention)
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
                p.drawLine(QPointF(ex, ey),
                           QPointF(ex + dx / ln * tl, ey + dy / ln * tl))

    # ── Core ───────────────────────────────────────────────────────────────────
    def _draw_core(self, p: QPainter, cx: float, cy: float, e: float, t: float):
        """
            Draw the pulsing central energy core.

            Layers multiple soft glowing circles with high opacity at the center,
            using a sine wave to create a continuous pulsing effect.

            :param p: The active painter.
            :type p: PySide6.QtGui.QPainter
            :param cx: Center X coordinate.
            :type cx: float
            :param cy: Center Y coordinate.
            :type cy: float
            :param e: Current energy value.
            :type e: float
            :param t: Elapsed time in seconds.
            :type t: float

            **Usage example**::

                # Called inside paintEvent.
                widget._draw_core(painter, 210, 210, 0.5, 1.2)
        """
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

    # ── Label ──────────────────────────────────────────────────────────────────
    def _draw_label(self, p: QPainter, e: float):
        """
            Draw the bottom "J . A . R . V . I . S ." text.

            Positions the text at the bottom of the widget container, adjusting
            opacity based on speech energy.

            :param p: The active painter.
            :type p: PySide6.QtGui.QPainter
            :param e: Current energy value.
            :type e: float

            **Usage example**::

                # Called inside paintEvent.
                widget._draw_label(painter, 0.5)
        """
        a = int((0.28 + e * 0.60) * 255)
        col = QColor(255, int(75 + e * 85), 0, a)
        p.setPen(col)
        p.setFont(QFont("Courier", 8, QFont.Bold))
        p.drawText(QRectF(0, H - 24, W, 18), Qt.AlignHCenter,
                   "J . A . R . V . I . S .")


# ── Controller ─────────────────────────────────────────────────────────────────
@dataclass
class JarvisHologram:
    """
        Thread-safe controller for managing the Jarvis widget life cycle.

        Launches the PySide6 application loop in a dedicated background daemon
        thread and exposes public methods to modify the widget's speech state
        and audio amplitude safely from other threads.

        :param _widget: The active widget instance.
        :type _widget: JarvisWidget
        :param _app: The active PySide6 application.
        :type _app: PySide6.QtWidgets.QApplication
        :param _thread: The background thread running the Qt event loop.
        :type _thread: threading.Thread
        :param _ready: Threading event indicating the widget has finished initialization.
        :type _ready: threading.Event

        **Usage example**::

            hologram = JarvisHologram()
            hologram.start()
            hologram.set_speaking(True)
            hologram.stop()
    """

    _widget: object = field(default=None, init=False)
    _app: object = field(default=None, init=False)
    _thread: object = field(default=None, init=False)
    _ready: object = field(default=None, init=False)

    def __post_init__(self):
        """
            Initialize threading events.

            Sets up the ready synchronization event before starting any background threads.

            **Usage example**::

                # Automatically called on instantiation.
                h = JarvisHologram()
        """
        self._ready = threading.Event()

    def start(self):
        """
            Start the Qt application thread and wait for initialization.

            Creates a daemon thread running the Qt event loop and blocks the current
            thread for up to 5 seconds until the widget is fully showing.

            **Usage example**::

                hologram = JarvisHologram()
                hologram.start()
        """
        self._thread = threading.Thread(target=self._run_qt, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        """
            Safely terminate the Qt application and close the widget.

            Requests closure through the widget's thread-safe signal bridge.

            **Usage example**::

                hologram.stop()
        """
        if self._widget:
            self._widget.request_close()   # thread-safe via Signal

    def set_speaking(self, v: bool):
        """
            Set the widget's speaking state.

            Proxies the call to the widget if it has been initialized.

            :param v: True if speaking, False otherwise.
            :type v: bool

            **Usage example**::

                hologram.set_speaking(True)
        """
        if self._widget:
            self._widget.set_speaking(v)

    def push_amplitude(self, rms: float):
        """
            Send the audio amplitude value to the widget.

            Proxies the call to the widget if it has been initialized.

            :param rms: Sound amplitude (Root Mean Square).
            :type rms: float

            **Usage example**::

                hologram.push_amplitude(0.12)
        """
        if self._widget:
            self._widget.push_amplitude(rms)

    def _run_qt(self):
        """
            Instantiate the Qt application, show the widget, and enter the main event loop.

            Runs in the background thread, sets up QApplication, and marks the ready event.

            **Usage example**::

                # Called internally by start() within a thread.
                hologram._run_qt()
        """
        import sys
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._widget = JarvisWidget()
        self._widget.show()
        self._ready.set()
        self._app.exec()


# ── TTS Integration ────────────────────────────────────────────────────────────
def attach_to_tts(hologram: JarvisHologram, tts_instance):
    """
        Monkey-patch a TextToSpeech instance to synchronize it with the hologram.

        Replaces the instance's speak method with a custom implementation that
        sends the audio's RMS amplitude values to the hologram while playing sound.

        :param hologram: The hologram controller.
        :type hologram: JarvisHologram
        :param tts_instance: The TextToSpeech instance to patch.
        :type tts_instance: TextToSpeech

        **Usage example**::

            hologram = JarvisHologram()
            hologram.start()
            tts = TextToSpeech()
            attach_to_tts(hologram, tts)
            tts.speak("Hello, sir.")
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
