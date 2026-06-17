"""
Jarvis AOU hologram visualisation module — PySide6.

This module implements a frameless, fully transparent Qt window that renders
an animated Jarvis-style holographic sphere inspired by *Avengers: Age of
Ultron*.  The visualisation is audio-reactive: an RMS amplitude value can be
pushed from the TTS pipeline at runtime to drive ring rotation speed, sphere
brightness, and circuit-arc intensity.

Key components:

- :func:`lerp` / :func:`lerp_color` / :func:`rot2d` — low-level math helpers.
- :class:`JarvisWidget` — the :class:`~PySide6.QtWidgets.QWidget` that owns the
  painting logic (plasma sphere, PCB arcs, outer rings, glowing core, label).
- :class:`JarvisHologram` — dataclass controller that runs the Qt event loop in
  a daemon thread and exposes a thread-safe public API.
- :func:`attach_to_tts` — monkey-patches the ``speak()`` method of any
  :class:`~speech2text2speech.main.TextToSpeech` instance to feed real-time RMS
  amplitude to a :class:`JarvisHologram` during audio playback.

**Usage example**::

    from speech2text2speech.hologram import JarvisHologram, attach_to_tts
    from speech2text2speech.main import TextToSpeech

    tts = TextToSpeech()
    hologram = JarvisHologram()
    hologram.start()
    attach_to_tts(hologram, tts)   # monkey-patch speak()

    tts.speak("Hello, I am JARVIS.")
"""
import math
import random
import threading
import time
from dataclasses import dataclass, field

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer
from PySide6.QtGui import (QBrush, QColor, QPainter, QPainterPath, QPen,
                           QRadialGradient)
from PySide6.QtWidgets import QApplication, QWidget

# ── Dimensions ───────────────────────────────────────────────────────────────
W = H = 480
CX = CY = W // 2
SPHERE_R = 145       # rayon sphère plasma
RING_RADII = [158, 170, 184]   # anneaux extérieurs

# ── Palette Jarvis AOU ────────────────────────────────────────────────────────
# orange glowing aura, formes angulaires circuit imprimé
C_BG = QColor(0, 0, 0, 0)  # fond totalement transparent
C_CORE = QColor(255, 255, 200, 255)
C_HOT = QColor(255, 200, 60,  255)
C_BRIGHT = QColor(255, 130, 0,   255)
C_MID = QColor(200, 70,  0,   255)
C_DIM = QColor(120, 30,  0,   200)
C_DARK = QColor(60,  10,  0,   150)
C_GLOW = QColor(40,  5,   0,   80)

RNG = random.Random(42)


def lerp(a, b, t):
    """
    Linearly interpolate between *a* and *b*, clamping *t* to ``[0.0, 1.0]``.

    :param a: Start value.
    :type a: float
    :param b: End value.
    :type b: float
    :param t: Interpolation factor, clamped to ``[0.0, 1.0]``.
    :type t: float
    :returns: Interpolated value between *a* and *b*.
    :rtype: float

    **Usage example**::

        lerp(0.0, 10.0, 0.5)   # → 5.0
        lerp(0.0, 10.0, 2.0)   # → 10.0  (clamped)
    """
    return a + (b - a) * max(0.0, min(1.0, t))


def lerp_color(c1: QColor, c2: QColor, t: float) -> QColor:
    """
    Linearly interpolate between two :class:`~PySide6.QtGui.QColor` values.

    Each RGBA channel is interpolated independently using :func:`lerp`,
    with *t* clamped to ``[0.0, 1.0]``.

    :param c1: Start colour.
    :type c1: PySide6.QtGui.QColor
    :param c2: End colour.
    :type c2: PySide6.QtGui.QColor
    :param t: Interpolation factor, clamped to ``[0.0, 1.0]``.
    :type t: float
    :returns: Blended colour.
    :rtype: PySide6.QtGui.QColor

    **Usage example**::

        mid = lerp_color(QColor(0, 0, 0, 255), QColor(255, 255, 255, 255), 0.5)
        # mid ≈ QColor(127, 127, 127, 255)
    """
    t = max(0.0, min(1.0, t))
    return QColor(
        int(lerp(c1.red(),   c2.red(),   t)),
        int(lerp(c1.green(), c2.green(), t)),
        int(lerp(c1.blue(),  c2.blue(),  t)),
        int(lerp(c1.alpha(), c2.alpha(), t)),
    )


def rot2d(x, y, angle_rad):
    """
    Rotate the 2-D point *(x, y)* by *angle_rad* radians around the origin.

    :param x: X coordinate of the point to rotate.
    :type x: float
    :param y: Y coordinate of the point to rotate.
    :type y: float
    :param angle_rad: Rotation angle in radians (counter-clockwise).
    :type angle_rad: float
    :returns: Rotated coordinates ``(x', y')``.
    :rtype: tuple[float, float]

    **Usage example**::

        rx, ry = rot2d(1.0, 0.0, math.pi / 2)
        # rx ≈ 0.0, ry ≈ 1.0
    """
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return x * c - y * s, x * s + y * c


# ── Widget Qt ─────────────────────────────────────────────────────────────────
class JarvisWidget(QWidget):
    def __init__(self):
        """
        Initialise the frameless, translucent Qt window and start the render timer.

        Sets the window flags to frameless / always-on-top / tool (hidden from the
        taskbar) and enables per-pixel alpha transparency.  Initialises all
        animation state variables and starts a :class:`~PySide6.QtCore.QTimer`
        that fires every 25 ms (~40 fps) to drive the render loop via
        :meth:`_tick`.

        **Usage example**::

            app = QApplication([])
            widget = JarvisWidget()
            widget.show()
            app.exec()
        """
        super().__init__()
        # Fenêtre sans décoration, toujours au-dessus, fond transparent par pixel
        self.setWindowFlags(
            Qt.FramelessWindowHint |
            Qt.WindowStaysOnTopHint |
            Qt.Tool                      # pas dans la barre des tâches
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_NoSystemBackground)
        self.resize(W, H)

        # État animation
        self._t = 0.0
        self._energy = 0.0
        self._amp = 0.0
        self._amp_lock = threading.Lock()

        # Anneaux : (angle courant, vitesse idle, mult parole, reverse)
        self._rings = [
            [0.0, 0.22, 4.5, False],
            [0.0, 0.35, 5.5, True],
            [0.0, 0.18, 3.5, False],
        ]
        # Arcs "circuit imprimé" : chaque arc = liste de segments angulaires
        self._circuit_arcs = self._gen_circuits()
        self._circuit_timer = 0.0

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(25)   # ~40 fps

    # ── API publique ──────────────────────────────────────────────────────────
    def push_amplitude(self, rms: float):
        """
        Thread-safe injection of a normalised RMS amplitude value.

        Called from the TTS thread (via :func:`attach_to_tts`) after each audio
        block is submitted to sounddevice.  The value is clamped to
        ``[0.0, 1.0]`` and stored under a lock so the Qt render thread can read
        it safely during :meth:`_tick`.

        :param rms: Normalised RMS amplitude of the current audio block.
        :type rms: float

        **Usage example**::

            widget.push_amplitude(0.75)   # signal moderate speech energy
        """
        with self._amp_lock:
            self._amp = max(0.0, min(1.0, rms))

    def set_speaking(self, v: bool):
        """
        Toggle the *speaking* state that sustains a baseline energy level.

        When ``True`` the animation maintains a minimum idle energy of ``0.08``
        so the sphere does not go fully dark between audio blocks.

        :param v: ``True`` while the TTS engine is synthesising / playing audio,
            ``False`` otherwise.
        :type v: bool

        **Usage example**::

            widget.set_speaking(True)    # activate before audio playback
            # ... play audio, push amplitudes ...
            widget.set_speaking(False)   # deactivate when done
        """
        self._speaking = v if hasattr(self, '_speaking') else v
        self._speaking = v

    # ── Génération arcs circuit imprimé ───────────────────────────────────────
    def _gen_circuits(self):
        """
        Generate a list of PCB-style angular arc descriptors.

        Each arc is a dictionary with polar-coordinate waypoints and animation
        parameters (phase, rotation speed, base opacity).  Arcs alternate between
        radial and angular segments to mimic printed-circuit-board traces, a
        signature visual of the Jarvis AOU interface.

        :returns: List of arc descriptor dicts, each containing:

            * ``'pts'`` — ``list[tuple[float, float]]`` of ``(radius, angle)``
              waypoints in polar coordinates.
            * ``'phase'`` — initial animation phase offset (radians).
            * ``'speed'`` — rotation speed (radians/s, may be negative).
            * ``'base_alpha'`` — baseline opacity in ``[0.2, 0.7]``.

        :rtype: list[dict]

        **Usage example**::

            # Called automatically in __init__; partially re-called in _tick.
            arcs = self._gen_circuits()
        """
        arcs = []
        for _ in range(18):
            start_r = RNG.uniform(30, SPHERE_R - 20)
            start_a = RNG.uniform(0, math.pi * 2)
            pts = [(start_r, start_a)]
            n_segs = RNG.randint(2, 5)
            r, a = start_r, start_a
            for _ in range(n_segs):
                step_type = RNG.choice(['radial', 'angular', 'angular'])
                if step_type == 'radial':
                    r = max(15, min(SPHERE_R - 5, r + RNG.uniform(-30, 30)))
                else:
                    a += RNG.uniform(-0.6, 0.6)
                pts.append((r, a))
            arcs.append({
                'pts': pts,
                'phase': RNG.uniform(0, math.pi * 2),
                'speed': RNG.uniform(-0.3, 0.3),
                'base_alpha': RNG.uniform(0.2, 0.7),
            })
        return arcs

    # ── Tick ──────────────────────────────────────────────────────────────────
    def _tick(self):
        """
        Animation tick called every ~25 ms by the Qt timer.

        Performs the following steps each frame:

        1. Advances the global time counter ``_t`` by ``dt = 0.025`` s.
        2. Reads and decays the raw amplitude from the lock-protected buffer.
        3. Computes ``_energy`` — a smoothed amplitude with asymmetric attack
           (fast) / release (slow) rates.
        4. Updates each ring's rotation angle proportional to the energy.
        5. Periodically replaces a random subset of circuit arcs to simulate
           live data streams.
        6. Calls :meth:`~PySide6.QtWidgets.QWidget.update` to schedule a repaint.

        **Usage example**::

            # Invoked automatically by the QTimer; do not call directly.
        """
        dt = 0.025
        self._t += dt

        with self._amp_lock:
            raw = self._amp
            self._amp *= 0.72

        speaking = getattr(self, '_speaking', False)
        base = 0.08 if speaking else 0.0
        target = max(base, raw)
        rate = 0.28 if target > self._energy else 0.05
        self._energy += (target - self._energy) * rate

        e = self._energy
        for ring in self._rings:
            spd = ring[1] * (1.0 + (ring[2] - 1.0) * e)
            sign = -1 if ring[3] else 1
            ring[0] = (ring[0] + spd * sign) % 360

        # Régénère les circuits périodiquement
        self._circuit_timer += dt
        if self._circuit_timer > 1.2:
            self._circuit_timer = 0.0
            # Remplace quelques arcs aléatoirement
            for i in RNG.sample(range(len(self._circuit_arcs)), 4):
                new = self._gen_circuits()
                self._circuit_arcs[i] = new[0]

        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────
    def paintEvent(self, _event):
        """
        Qt paint event — compose the full hologram frame.

        Called by the Qt framework whenever the widget needs repainting
        (triggered by :meth:`_tick` via :meth:`~PySide6.QtWidgets.QWidget.update`).
        Clears the canvas to full transparency, then delegates to the individual
        layer-drawing helpers in back-to-front order:

        1. :meth:`_draw_plasma_sphere` — ambient glow and data-stream veins.
        2. :meth:`_draw_circuit_arcs` — PCB-trace arcs.
        3. :meth:`_draw_outer_rings` — audio-reactive spinning rings.
        4. :meth:`_draw_core` — layered bright core with pulsing halo.
        5. :meth:`_draw_label` — ``J . A . R . V . I . S .`` text.

        :param _event: Qt paint event (unused; widget is fully redrawn each frame).
        :type _event: PySide6.QtGui.QPaintEvent

        **Usage example**::

            # Invoked automatically by Qt; do not call directly.
        """
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setRenderHint(QPainter.SmoothPixmapTransform)

        # Fond : totalement transparent
        p.setCompositionMode(QPainter.CompositionMode_Source)
        p.fillRect(self.rect(), Qt.transparent)
        p.setCompositionMode(QPainter.CompositionMode_SourceOver)

        e = self._energy
        t = self._t

        self._draw_plasma_sphere(p, e, t)
        self._draw_circuit_arcs(p, e, t)
        self._draw_outer_rings(p, e)
        self._draw_core(p, e, t)
        self._draw_label(p, e)
        p.end()

    # ── Sphère plasma ─────────────────────────────────────────────────────────
    def _draw_plasma_sphere(self, p: QPainter, e: float, t: float):
        """
        Draw the ambient plasma sphere using stacked radial gradients.

        Renders three overlapping layers:

        1. An outer diffuse halo that expands with *energy*.
        2. A hot amber radial gradient forming the main sphere body.
        3. Animated arc-shaped *data-stream veins* whose count, radius, and
           opacity all increase with *energy*.

        :param p: Active :class:`~PySide6.QtGui.QPainter` for this frame.
        :type p: PySide6.QtGui.QPainter
        :param e: Current smoothed energy in ``[0.0, 1.0]``.
        :type e: float
        :param t: Global time counter in seconds.
        :type t: float

        **Usage example**::

            # Called automatically from paintEvent(); do not call directly.
        """
        boost = 1.0 + e * 0.18
        r = SPHERE_R * boost

        # Couche extérieure : halo diffus
        grad = QRadialGradient(CX, CY, r * 1.25)
        grad.setColorAt(0.0, QColor(80, 20, 0, int(60 + e * 80)))
        grad.setColorAt(0.5, QColor(40, 8,  0, int(30 + e * 50)))
        grad.setColorAt(1.0, QColor(0,  0,  0, 0))
        p.setBrush(QBrush(grad))
        p.setPen(Qt.NoPen)
        rr = r * 1.25
        p.drawEllipse(QRectF(CX - rr, CY - rr, rr * 2, rr * 2))

        # Corps principal : gradient radial chaud
        grad2 = QRadialGradient(CX, CY, r)
        # Centre chaud
        c_center = QColor(
            min(255, int(180 + e * 75)),
            min(255, int(80 + e * 60)),
            int(10 + e * 20),
            int(200 + e * 55)
        )
        # Bord : orange vif
        c_edge = QColor(
            min(255, int(140 + e * 80)),
            int(40 + e * 40),
            0,
            int(120 + e * 80)
        )
        grad2.setColorAt(0.0,  c_center)
        grad2.setColorAt(0.45, lerp_color(c_center, c_edge, 0.5))
        grad2.setColorAt(0.75, c_edge)
        grad2.setColorAt(1.0,  QColor(60, 10, 0, 0))
        p.setBrush(QBrush(grad2))
        p.drawEllipse(QRectF(CX - r, CY - r, r * 2, r * 2))

        # Veines / lignes internes qui bougent (aspect "data streams")
        p.setPen(Qt.NoPen)
        n_veins = 5 + int(e * 7)
        for i in range(n_veins):
            phase = t * 0.6 + i * (math.pi * 2 / n_veins)
            vr = 35 + math.sin(phase * 1.4 + i) * 22
            va = math.degrees(phase)
            ve = 30 + math.sin(phase * 2.1) * 15
            alpha = int((0.12 + e * 0.35 + 0.1 * math.sin(phase * 3)) * 255)
            col = QColor(255, int(100 + e * 80), 0, max(0, min(255, alpha)))
            pen = QPen(col, 1.2)
            pen.setCapStyle(Qt.RoundCap)
            p.setPen(pen)
            p.setBrush(Qt.NoBrush)
            p.drawArc(
                QRectF(CX - vr, CY - vr, vr * 2, vr * 2),
                int(va * 16),
                int(ve * 16)
            )

    # ── Arcs circuit imprimé ──────────────────────────────────────────────────
    def _draw_circuit_arcs(self, p: QPainter, e: float, t: float):
        """
        Draw PCB-trace-style angular arcs inside the plasma sphere.

        Iterates over :attr:`_circuit_arcs`, converting each arc's polar waypoints
        to Cartesian coordinates (with a slow global rotation), then draws
        connected line segments via :class:`~PySide6.QtGui.QPainterPath`.  Circular
        node pads are placed at every other waypoint to strengthen the PCB
        aesthetic.  Arc brightness and line width scale with *energy*.

        :param p: Active :class:`~PySide6.QtGui.QPainter` for this frame.
        :type p: PySide6.QtGui.QPainter
        :param e: Current smoothed energy in ``[0.0, 1.0]``.
        :type e: float
        :param t: Global time counter in seconds.
        :type t: float

        **Usage example**::

            # Called automatically from paintEvent(); do not call directly.
        """
        p.setBrush(Qt.NoBrush)

        for arc in self._circuit_arcs:
            phase = arc['phase'] + t * arc['speed']
            visibility = 0.5 + 0.5 * math.sin(phase * 2.3)
            intensity = arc['base_alpha'] * (0.3 + e * 0.7) * visibility
            if intensity < 0.05:
                continue

            pts_world = []
            for r, a in arc['pts']:
                # rotation globale lente
                a_rot = a + t * 0.05
                x = CX + r * math.cos(a_rot)
                y = CY + r * math.sin(a_rot)
                pts_world.append(QPointF(x, y))

            if len(pts_world) < 2:
                continue

            alpha = int(min(255, intensity * 255))
            # Couleur : orange → ambre selon profondeur/énergie
            red = min(255, int(200 + e * 55))
            green = min(255, int(80 + e * 80))
            col = QColor(red, green, 0, alpha)

            pen = QPen(col, 1.0 + e * 0.8)
            pen.setCapStyle(Qt.SquareCap)  # aspect circuit imprimé : angles droits
            pen.setJoinStyle(Qt.MiterJoin)
            p.setPen(pen)

            path = QPainterPath()
            path.moveTo(pts_world[0])
            for pt in pts_world[1:]:
                path.lineTo(pt)
            p.drawPath(path)

            # Nœud (pastille) à chaque angle
            node_col = QColor(min(255, red + 30), min(255, green + 40), 20, alpha)
            p.setBrush(QBrush(node_col))
            p.setPen(Qt.NoPen)
            for pt in pts_world[::2]:  # un nœud sur deux
                nr = 1.5 + e * 1.5
                p.drawEllipse(pt, nr, nr)
            p.setBrush(Qt.NoBrush)

    # ── Anneaux extérieurs audio-réactifs ────────────────────────────────────
    def _draw_outer_rings(self, p: QPainter, e: float):
        """
        Draw the three audio-reactive spinning elliptical rings.

        Each ring is rendered as a series of dashed arcs on a flattened ellipse
        (simulating a 3-D tilted ring viewed from the side).  A depth factor
        derived from each arc's angular position modulates colour and stroke width
        to fake per-segment shading.  Radial tick marks are drawn at evenly spaced
        angles to reinforce the mechanical look.

        :param p: Active :class:`~PySide6.QtGui.QPainter` for this frame.
        :type p: PySide6.QtGui.QPainter
        :param e: Current smoothed energy in ``[0.0, 1.0]``.
        :type e: float

        **Usage example**::

            # Called automatically from paintEvent(); do not call directly.
        """
        p.setBrush(Qt.NoBrush)

        ring_configs = [
            # (rayon, inclinaison tilt simulée, dash_on, dash_off)
            (158, 1.0,  10, 14),
            (170, 0.85, 8,  10),
            (184, 0.95, 6,  8),
        ]

        for idx, (ring, config) in enumerate(zip(self._rings, ring_configs)):
            r_base, tilt, dash_on, dash_off = config
            angle_offset = ring[0]

            # Simule la 3D : ellipse aplatie selon tilt
            rx = r_base
            ry = r_base * (0.25 + 0.1 * idx)   # très aplati = vue de côté

            brightness = 0.3 + e * 0.65
            alpha = int((0.4 + e * 0.5) * 255)

            red = min(255, int(180 + e * 75))
            green = min(255, int(70 + e * 80))
            col_bright = QColor(red, green, 0, alpha)
            col_dim = QColor(80, 20, 0, int(alpha * 0.3))

            cycle = dash_on + dash_off
            n_steps = 360

            for start_deg in range(0, n_steps, cycle):
                end_deg = start_deg + dash_on
                # angle réel avec rotation
                a_start = (start_deg + angle_offset) % 360
                a_end = (end_deg + angle_offset) % 360

                # Depth: simule quelle partie de l'anneau est "devant"
                mid_a = math.radians((a_start + a_end) / 2)
                depth = 0.5 + 0.5 * math.sin(mid_a)  # 0=derrière, 1=devant

                col = lerp_color(col_dim, col_bright, depth * brightness)
                w = 1.0 + depth * e * 2.0
                pen = QPen(col, w)
                pen.setCapStyle(Qt.RoundCap)
                p.setPen(pen)

                span = dash_on
                p.drawArc(
                    QRectF(CX - rx, CY - ry, rx * 2, ry * 2),
                    int(a_start * 16),
                    int(span * 16)
                )

            # Marques de tick sur l'anneau
            n_ticks = 8 + idx * 4
            for i in range(n_ticks):
                a = math.radians((i * 360 / n_ticks + angle_offset) % 360)
                # point sur l'ellipse
                ex = CX + rx * math.cos(a)
                ey = CY + ry * math.sin(a)
                # direction vers centre
                dx, dy = CX - ex, CY - ey
                ln = math.sqrt(dx*dx + dy*dy) or 1
                tl = 5 + e * 4
                depth = 0.5 + 0.5 * math.sin(a)
                tcol = lerp_color(col_dim, col_bright, depth)
                p.setPen(QPen(tcol, 1.0))
                p.drawLine(
                    QPointF(ex, ey),
                    QPointF(ex + dx/ln * tl, ey + dy/ln * tl)
                )

    # ── Core ──────────────────────────────────────────────────────────────────
    def _draw_core(self, p: QPainter, e: float, t: float):
        """
        Draw the layered glowing core at the centre of the hologram.

        Stacks six concentric radial-gradient ellipses from dark amber to
        near-white to simulate a hot plasma core.  Each layer's radius is
        modulated by a fast sine pulse and by *energy* so the core breathes
        visibly.  A thin halo ring is drawn around the outermost layer.

        :param p: Active :class:`~PySide6.QtGui.QPainter` for this frame.
        :type p: PySide6.QtGui.QPainter
        :param e: Current smoothed energy in ``[0.0, 1.0]``.
        :type e: float
        :param t: Global time counter in seconds.
        :type t: float

        **Usage example**::

            # Called automatically from paintEvent(); do not call directly.
        """
        pulse = 0.5 + 0.5 * math.sin(t * 7.0)
        ep = min(1.0, e + pulse * 0.07)

        layers = [
            (50, QColor(100, 25, 0, int(180 * (0.4 + ep * 0.6)))),
            (38, QColor(180, 60, 0, int(210 * (0.5 + ep * 0.5)))),
            (26, QColor(240, 110, 0, int(230 * (0.6 + ep * 0.4)))),
            (16, QColor(255, 180, 40, int(245 * (0.7 + ep * 0.3)))),
            (9,  QColor(255, 230, 120, int(255 * (0.8 + ep * 0.2)))),
            (4,  QColor(255, 255, 220, 255)),
        ]
        p.setPen(Qt.NoPen)
        for r, col in layers:
            ri = r * (1.0 + e * 0.2 + pulse * 0.04)
            grad = QRadialGradient(CX, CY, ri)
            grad.setColorAt(0.0, col)
            fade = QColor(col.red(), col.green(), col.blue(), 0)
            grad.setColorAt(1.0, fade)
            p.setBrush(QBrush(grad))
            p.drawEllipse(QRectF(CX - ri, CY - ri, ri * 2, ri * 2))

        # Halo annulaire autour du core
        halo_r = 55 * (1 + e * 0.15)
        halo_col = QColor(255, int(100 + e * 80), 0, int(80 + e * 120))
        p.setPen(QPen(halo_col, 1.5))
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(QRectF(CX - halo_r, CY - halo_r, halo_r * 2, halo_r * 2))

    # ── Label ──────────────────────────────────────────────────────────────────
    def _draw_label(self, p: QPainter, e: float):
        """
        Draw the ``J . A . R . V . I . S .`` label at the bottom of the widget.

        Uses a monospaced Courier Bold font.  Label opacity and colour intensity
        scale with *energy* so the text brightens when the hologram is active.

        :param p: Active :class:`~PySide6.QtGui.QPainter` for this frame.
        :type p: PySide6.QtGui.QPainter
        :param e: Current smoothed energy in ``[0.0, 1.0]``.
        :type e: float

        **Usage example**::

            # Called automatically from paintEvent(); do not call directly.
        """
        alpha = int((0.3 + e * 0.6) * 255)
        col = QColor(255, int(80 + e * 80), 0, alpha)
        p.setPen(col)
        p.setFont(p.font())
        from PySide6.QtGui import QFont
        font = QFont("Courier", 9, QFont.Bold)
        p.setFont(font)
        p.drawText(QRectF(0, H - 28, W, 20), Qt.AlignHCenter, "J . A . R . V . I . S .")


# ── Hologram controller ───────────────────────────────────────────────────────
@dataclass
class JarvisHologram:
    fps: int = 40
    _widget: object = field(default=None, init=False)
    _app: object = field(default=None, init=False)
    _thread: object = field(default=None, init=False)
    _running: bool = field(default=False, init=False)
    _ready: object = field(default=None, init=False)

    def __post_init__(self):
        self._ready = threading.Event()

    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._run_qt, daemon=True)
        self._thread.start()
        self._ready.wait(timeout=5.0)

    def stop(self):
        self._running = False
        if self._widget:
            try:
                from PySide6.QtCore import QMetaObject
                from PySide6.QtCore import Qt as Qtt
                QMetaObject.invokeMethod(self._widget, "close", Qtt.QueuedConnection)
            except Exception:
                pass

    def set_speaking(self, v: bool):
        if self._widget:
            self._widget.set_speaking(v)

    def push_amplitude(self, rms: float):
        if self._widget:
            self._widget.push_amplitude(rms)

    def _run_qt(self):
        import sys
        self._app = QApplication.instance() or QApplication(sys.argv)
        self._widget = JarvisWidget()
        self._widget.show()
        self._ready.set()
        self._app.exec()


# ── Intégration TTS ───────────────────────────────────────────────────────────
def attach_to_tts(hologram: JarvisHologram, tts_instance):
    import io
    import wave

    import numpy as np
    import sounddevice as sd
    import soundfile as sf

    def _patched_speak(text: str):
        hologram.set_speaking(True)
        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wf:
                tts_instance.voice.synthesize_wav(text, wf)
            buf.seek(0)
            data, sr = sf.read(buf, dtype="float32")

            block = max(1, int(sr / hologram.fps))
            words = text.split()
            duration = len(data) / sr
            delay = duration / max(len(words), 1)

            import threading as _th

            def stream_words():
                for w in words:
                    print(w, end=" ", flush=True)
                    time.sleep(delay)
                print()
            _th.Thread(target=stream_words, daemon=True).start()

            sd.play(data, sr)
            idx = 0
            while idx < len(data):
                chunk = data[idx:idx + block]
                if len(chunk):
                    rms = float(np.sqrt(np.mean(chunk ** 2)))
                    hologram.push_amplitude(min(1.0, rms / 0.12))
                idx += block
                time.sleep(1.0 / hologram.fps)
            sd.wait()
        finally:
            hologram.set_speaking(False)

    tts_instance.speak = _patched_speak


# ── Demo ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    h = JarvisHologram()
    h.start()
    print("Idle 3s...")
    time.sleep(3)

    print("Simulation parole...")
    h.set_speaking(True)
    for i in range(280):
        amp = abs(math.sin(i * 0.22)) * 0.9 + random.uniform(0, 0.08)
        h.push_amplitude(amp)
        time.sleep(0.025)

    h.set_speaking(False)
    print("Idle 3s...")
    time.sleep(3)
    h.stop()
