"""
Frame annotation for the Spatial Scene Monitor.
"""

from __future__ import annotations

import cv2
import numpy as np

from modules.fusion.data_types import TrackState

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_FONT_SCALE = 0.5
_FONT_THICKNESS = 1
_BOX_THICKNESS = 2


class Visualiser:
    """
    Renders one annotated frame from a raw frame + TrackStateStore.

    Pure rendering — never displays a window or writes to disk itself;
    main.py decides what to do with the returned array (cv2.imshow,
    VideoWriter, both, or neither), per CLAUDE.md's module responsibility.
    """

    def __init__(self, draw_trajectory: bool = True) -> None:
        self.draw_trajectory = draw_trajectory

    def render(self, frame: np.ndarray, track_states: dict[int, TrackState]) -> np.ndarray:
        """Returns a new annotated frame; never mutates the input."""
        annotated = frame.copy()
        # Badges placed so far this frame, for collision avoidance in _draw_label.
        placed_badges: list[tuple[int, int, int, int]] = []

        for state in track_states.values():
            if state.box is None:
                continue  # not expected in practice — every birthed track gets a box immediately
            if self.draw_trajectory:
                self._draw_trajectory(annotated, state)
            self._draw_box(annotated, state)
            self._draw_label(annotated, state, placed_badges)

        return annotated

    def _draw_box(self, frame: np.ndarray, state: TrackState) -> None:
        x1, y1, x2, y2 = (int(round(v)) for v in state.box)
        cv2.rectangle(frame, (x1, y1), (x2, y2), state.risk_level.colour_bgr, _BOX_THICKNESS)

    def _draw_label(
        self,
        frame: np.ndarray,
        state: TrackState,
        placed_badges: list[tuple[int, int, int, int]],
    ) -> None:
        """
        Draws this track's label badge, clamped fully on-screen and nudged
        below any badge already placed this frame that it would overlap.

        Found by testing against real KITTI footage, not synthetic boxes:
        a row of parked cars produced badges clipped off the frame's right
        edge, and dense clusters of nearby cars produced unreadable
        overlapping badges. Neither showed up with the widely-spaced
        synthetic test boxes used when this module was first built.
        """
        x1, y1, _, _ = (int(round(v)) for v in state.box)
        frame_h, frame_w = frame.shape[:2]
        colour = state.risk_level.colour_bgr

        text = f"#{state.track_id} {state.class_name} d={state.depth_smoothed:.2f} {state.risk_level.value}"
        (text_w, text_h), baseline = cv2.getTextSize(text, _FONT, _FONT_SCALE, _FONT_THICKNESS)
        badge_w = text_w + 6
        badge_h = text_h + baseline + 4

        # Preferred position: just above the box's top-left corner.
        badge_y2 = max(y1, badge_h)
        badge_y1 = badge_y2 - badge_h
        badge_x1 = x1

        # Horizontal clamp (mirrors the existing vertical clamp below): keep
        # the whole badge on-screen rather than letting text run off the edge.
        badge_x1 = max(0, min(badge_x1, frame_w - badge_w))
        badge_x2 = badge_x1 + badge_w

        # Vertical collision avoidance: push straight down past any
        # already-placed badge this one would overlap. Repeats so a chain of
        # 3+ overlapping badges cascades cleanly instead of just dodging the
        # first one. Deliberately doesn't clamp to the bottom edge — in a
        # pathologically dense cluster a badge can run off the bottom, which
        # is an honest "too cluttered to label" signal rather than hiding it.
        moved = True
        while moved:
            moved = False
            for px1, py1, px2, py2 in placed_badges:
                overlap_x = badge_x1 < px2 and badge_x2 > px1
                overlap_y = badge_y1 < py2 and badge_y2 > py1
                if overlap_x and overlap_y:
                    badge_y1 = py2 + 2
                    badge_y2 = badge_y1 + badge_h
                    moved = True

        placed_badges.append((badge_x1, badge_y1, badge_x2, badge_y2))

        cv2.rectangle(frame, (badge_x1, badge_y1), (badge_x2, badge_y2), colour, cv2.FILLED)
        cv2.putText(
            frame, text, (badge_x1 + 3, badge_y2 - baseline - 2),
            _FONT, _FONT_SCALE, (0, 0, 0), _FONT_THICKNESS, cv2.LINE_AA,
        )

    def _draw_trajectory(self, frame: np.ndarray, state: TrackState) -> None:
        """
        Pseudo-3D trail: thickness encodes depth (closer = thicker), and older
        segments are dimmed, so recent, close motion visually reads as "in
        front of" older, farther motion — a depth cue with no real 3D render.
        """
        points = list(state.trajectory_2d)
        depths = list(state.trajectory_depth)
        if len(points) < 2:
            return

        base_colour = state.risk_level.colour_bgr
        n = len(points)
        for i in range(1, n):
            pt1 = (int(round(points[i - 1][0])), int(round(points[i - 1][1])))
            pt2 = (int(round(points[i][0])), int(round(points[i][1])))

            thickness = int(np.clip(round(depths[i] * 1.5), 1, 5))

            age_fraction = i / (n - 1)  # 0 = oldest segment, 1 = newest
            fade = 0.3 + 0.7 * age_fraction
            colour = tuple(int(c * fade) for c in base_colour)

            cv2.line(frame, pt1, pt2, colour, thickness)
