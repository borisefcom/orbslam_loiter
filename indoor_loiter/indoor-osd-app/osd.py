#!/usr/bin/env python3
import os, math, re
from typing import Any, Callable, Optional, Tuple, List
import imgui
import glfw

from utils import (
    OSD_ASSETS_DIR,
    pil_to_gl_texture,
    dl_add_image, dl_add_text, dl_add_rect_filled, dl_add_line
)
from mavlink import Telemetry


# ----------------- helpers -----------------
def _wrap360(deg: float) -> float:
    d = deg % 360.0
    return d if d >= 0 else d + 360.0

def _wrap180(delta_deg: float) -> float:
    """Wrap an angle delta to [-180, +180]."""
    d = (delta_deg + 180.0) % 360.0 - 180.0
    return d

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))

def _rad(deg): return deg * math.pi / 180.0


_FMT_VALUE_RE = re.compile(r"\{value[^}]*\}")


def _format_telem_value(fmt: str, value: Any, *, missing: str = "—") -> str:
    """
    Best-effort formatting for OSD text elements.
    - If value is missing/None, preserve units by replacing "{value...}" with "—".
    - Otherwise apply fmt.format(value=...).
    """
    if value is None or value == "":
        try:
            return _FMT_VALUE_RE.sub(missing, str(fmt))
        except Exception:
            return missing
    try:
        return str(fmt).format(value=value)
    except Exception:
        try:
            return str(value)
        except Exception:
            return missing


class DetectionsOverlay:
    def __init__(self):
        self.selected_track_id: Optional[int] = None
        self._roi_active: bool = False
        self._roi_center_fb: Tuple[float, float] = (0.0, 0.0)
        self._roi_rect_fb: Optional[Tuple[float, float, float, float]] = None

    def _mouse_fb(self, window, fb_w: int, fb_h: int) -> Optional[Tuple[float, float, float, float]]:
        """Return (mx_win, my_win, mx_fb, my_fb) or None if window is invalid."""
        mx_win, my_win = glfw.get_cursor_pos(window)
        ww_win, wh_win = glfw.get_window_size(window)
        if ww_win <= 0 or wh_win <= 0:
            return None
        mx_fb = mx_win * (fb_w / float(ww_win))
        my_fb = my_win * (fb_h / float(wh_win))
        return float(mx_win), float(my_win), float(mx_fb), float(my_fb)

    def _fb_rect_to_img_rect(
        self,
        telem: Telemetry,
        fb_w: int,
        fb_h: int,
        rect_fb: Tuple[float, float, float, float],
    ) -> Optional[Tuple[int, int, int, int]]:
        if fb_w <= 0 or fb_h <= 0:
            return None

        x0, y0, x1, y1 = rect_fb
        x0, y0, x1, y1 = float(x0), float(y0), float(x1), float(y1)
        if x1 < x0:
            x0, x1 = x1, x0
        if y1 < y0:
            y0, y1 = y1, y0

        img_size = getattr(telem, "detect_img_size", None)
        if img_size and img_size[0] not in (None, 0) and img_size[1] not in (None, 0):
            img_w = float(img_size[0])
            img_h = float(img_size[1])
        else:
            img_w = float(fb_w)
            img_h = float(fb_h)

        if img_w <= 0 or img_h <= 0:
            return None

        sx = img_w / float(fb_w)
        sy = img_h / float(fb_h)

        x = int(max(0, min(img_w - 1, x0 * sx)))
        y = int(max(0, min(img_h - 1, y0 * sy)))
        w = int(max(1, (x1 - x0) * sx))
        h = int(max(1, (y1 - y0) * sy))

        # Clamp to image bounds.
        if x + w > int(img_w):
            w = max(1, int(img_w) - x)
        if y + h > int(img_h):
            h = max(1, int(img_h) - y)

        return int(x), int(y), int(w), int(h)

    def handle_roi_select(
        self,
        window,
        fb_w: int,
        fb_h: int,
        telem: Telemetry,
        *,
        send_command: Callable[[dict], bool],
        ignore_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """
        Hold RMB to draw an ROI box from a fixed center point.
        On release, send:
          {"roi_select": {"x": int, "y": int, "w": int, "h": int}}
        Coordinates are in the detection image pixel space when available (telem.detect_img_size).
        """
        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        m = self._mouse_fb(window, fb_w, fb_h)
        if m is None:
            return
        mx_win, my_win, mx_fb, my_fb = m

        # Start (right-click down): set ROI center in framebuffer coords.
        if imgui.is_mouse_clicked(1):
            if ignore_rect is not None:
                wx, wy, ww, wh = ignore_rect
                if (wx <= mx_win <= wx + ww) and (wy <= my_win <= wy + wh):
                    return

            self._roi_active = True
            self._roi_center_fb = (mx_fb, my_fb)
            self._roi_rect_fb = (mx_fb, my_fb, mx_fb, my_fb)

        if not self._roi_active:
            return

        # Update while held.
        if imgui.is_mouse_down(1):
            cx, cy = self._roi_center_fb
            hw = abs(mx_fb - cx)
            hh = abs(my_fb - cy)
            x0 = _clamp(cx - hw, 0.0, float(fb_w))
            x1 = _clamp(cx + hw, 0.0, float(fb_w))
            y0 = _clamp(cy - hh, 0.0, float(fb_h))
            y1 = _clamp(cy + hh, 0.0, float(fb_h))
            self._roi_rect_fb = (x0, y0, x1, y1)

        # Finish on release: send roi_select.
        if imgui.is_mouse_released(1):
            rect_fb = self._roi_rect_fb
            self._roi_active = False
            self._roi_rect_fb = None

            if rect_fb is None:
                return

            rect_img = self._fb_rect_to_img_rect(telem, fb_w, fb_h, rect_fb)
            if rect_img is None:
                return
            x, y, w, h = rect_img

            try:
                send_command({"roi_select": {"x": int(x), "y": int(y), "w": int(w), "h": int(h)}})
                self.selected_track_id = 0
                print("[CTRL] roi_select:", {"x": int(x), "y": int(y), "w": int(w), "h": int(h)})
            except Exception as e:
                print("[CTRL] roi_select send error:", e)

    def draw(self, fb_w: int, fb_h: int, telem: Telemetry) -> None:
        # Overlay/input gating based on active camera selection.
        try:
            cam_active = getattr(telem, "camera_active", None)
            cam_active = str(cam_active).lower().strip() if cam_active is not None else ""
        except Exception:
            cam_active = ""
        is_v4l = (cam_active == "v4l")

        # --- 3D acquisition overlay (preview/selection/confirmed) ---
        try:
            stage = getattr(telem, "acq_stage", None)
            img_size = getattr(telem, "acq_img_size", None)
            verts_uv = getattr(telem, "acq_verts_uv", None)
        except Exception:
            stage = None
            img_size = None
            verts_uv = None

        if (not is_v4l) and stage is not None and img_size and verts_uv:
            try:
                img_w = float(img_size[0])
                img_h = float(img_size[1])
            except Exception:
                img_w = 0.0
                img_h = 0.0
            if img_w > 0 and img_h > 0 and fb_w > 0 and fb_h > 0:
                sx = fb_w / img_w
                sy = fb_h / img_h

                # Colors per stage (confirmed by user).
                col_yellow = imgui.get_color_u32_rgba(1.0, 0.95, 0.2, 1.0)
                col_purple = imgui.get_color_u32_rgba(0.7, 0.4, 1.0, 1.0)
                col_red_x = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)
                col_green_x = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0)
                col_fill_purple = imgui.get_color_u32_rgba(0.7, 0.4, 1.0, 0.33)

                if int(stage) == 0:
                    col_line = col_yellow
                    col_x = col_red_x
                    do_fill = False
                elif int(stage) == 1:
                    col_line = col_purple
                    col_x = col_yellow
                    do_fill = False
                else:
                    col_line = col_purple
                    col_x = col_green_x
                    do_fill = True

                pts_fb: List[Tuple[float, float]] = [(float(u) * sx, float(v) * sy) for (u, v) in (verts_uv or [])]
                if len(pts_fb) >= 3:
                    dl = imgui.get_foreground_draw_list()
                    if do_fill:
                        try:
                            fn = getattr(dl, "add_convex_poly_filled", None)
                            if callable(fn):
                                fn(pts_fb, col_fill_purple)
                        except Exception:
                            pass

                    # Outline
                    for i in range(len(pts_fb)):
                        x0, y0 = pts_fb[i]
                        x1, y1 = pts_fb[(i + 1) % len(pts_fb)]
                        dl_add_line(dl, x0, y0, x1, y1, col_line, 3.0)

                    # Center X
                    c = getattr(telem, "acq_center_uv", None)
                    if c is not None and isinstance(c, (list, tuple)) and len(c) >= 2:
                        cx_uv, cy_uv = float(c[0]), float(c[1])
                    elif c is not None and isinstance(c, tuple) and len(c) >= 2:
                        cx_uv, cy_uv = float(c[0]), float(c[1])
                    else:
                        try:
                            cx_uv = sum(p[0] for p in verts_uv) / float(len(verts_uv))
                            cy_uv = sum(p[1] for p in verts_uv) / float(len(verts_uv))
                        except Exception:
                            cx_uv, cy_uv = (img_w * 0.5, img_h * 0.5)

                    cx = cx_uv * sx
                    cy = cy_uv * sy
                    cross = 14.0
                    dl_add_line(dl, cx - cross, cy - cross, cx + cross, cy + cross, col_x, 3.0)
                    dl_add_line(dl, cx - cross, cy + cross, cx + cross, cy - cross, col_x, 3.0)
        # --- end acquisition overlay ---

        # --- 3D VO features overlay (track3d) ---
        try:
            feat_pts = getattr(telem, "vo_feat_pts", None)
            feat_img_size = getattr(telem, "vo_feat_img_size", None)
        except Exception:
            feat_pts = None
            feat_img_size = None

        if (not is_v4l) and feat_pts and feat_img_size and fb_w > 0 and fb_h > 0:
            try:
                img_w = float(feat_img_size[0])
                img_h = float(feat_img_size[1])
            except Exception:
                img_w = 0.0
                img_h = 0.0
            if img_w > 0 and img_h > 0:
                sx = fb_w / img_w
                sy = fb_h / img_h
                dl = imgui.get_foreground_draw_list()

                col_g1 = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0)   # green
                col_g2 = imgui.get_color_u32_rgba(1.0, 1.0, 0.2, 1.0)   # yellow
                col_g3 = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)   # red
                col_g0 = imgui.get_color_u32_rgba(0.8, 0.8, 0.8, 1.0)   # gray

                r = 2.0
                for (u, v, g) in (feat_pts or []):
                    x = float(u) * sx
                    y = float(v) * sy
                    if x < 0 or y < 0 or x >= float(fb_w) or y >= float(fb_h):
                        continue
                    col = col_g1 if int(g) == 1 else (col_g2 if int(g) == 2 else (col_g3 if int(g) == 3 else col_g0))
                    try:
                        dl.add_circle_filled(float(x), float(y), float(r), col, 12)
                    except Exception:
                        pass

        # --- 2D ROI tracker bbox overlay (track2d) ---
        if is_v4l and fb_w > 0 and fb_h > 0:
            bb = getattr(telem, "track2d_bbox_norm", None)
            if bb and isinstance(bb, (list, tuple)) and len(bb) >= 4:
                try:
                    x0n, y0n, x1n, y1n = float(bb[0]), float(bb[1]), float(bb[2]), float(bb[3])
                    x0n, x1n = (min(x0n, x1n), max(x0n, x1n))
                    y0n, y1n = (min(y0n, y1n), max(y0n, y1n))
                    x0 = _clamp(x0n * float(fb_w), 0.0, float(fb_w))
                    y0 = _clamp(y0n * float(fb_h), 0.0, float(fb_h))
                    x1 = _clamp(x1n * float(fb_w), 0.0, float(fb_w))
                    y1 = _clamp(y1n * float(fb_h), 0.0, float(fb_h))
                    dl = imgui.get_foreground_draw_list()
                    col = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 1.0)
                    dl_add_line(dl, x0, y0, x1, y0, col, 3.0)
                    dl_add_line(dl, x1, y0, x1, y1, col, 3.0)
                    dl_add_line(dl, x1, y1, x0, y1, col, 3.0)
                    dl_add_line(dl, x0, y1, x0, y0, col, 3.0)

                    # Optional DynMedianFlow internal grid overlay.
                    grid = getattr(telem, "track2d_grid", None)
                    if grid and isinstance(grid, (list, tuple)) and len(grid) >= 2:
                        try:
                            cols = int(grid[0])
                            rows = int(grid[1])
                        except Exception:
                            cols = rows = 0
                        if cols > 1 or rows > 1:
                            col_grid = imgui.get_color_u32_rgba(0.2, 1.0, 0.2, 0.35)
                            if cols > 1:
                                for i in range(1, cols):
                                    xx = x0 + (x1 - x0) * (float(i) / float(cols))
                                    dl_add_line(dl, xx, y0, xx, y1, col_grid, 1.0)
                            if rows > 1:
                                for j in range(1, rows):
                                    yy = y0 + (y1 - y0) * (float(j) / float(rows))
                                    dl_add_line(dl, x0, yy, x1, yy, col_grid, 1.0)
                except Exception:
                    pass

        if self._roi_active and self._roi_rect_fb is not None:
            dl = imgui.get_foreground_draw_list()
            col_purple = imgui.get_color_u32_rgba(0.7, 0.4, 1.0, 1.0)

            cx, cy = self._roi_center_fb
            cross = 18.0
            dl_add_line(dl, cx - cross, cy, cx + cross, cy, col_purple, 3.0)
            dl_add_line(dl, cx, cy - cross, cx, cy + cross, col_purple, 3.0)

            x0, y0, x1, y1 = self._roi_rect_fb
            dl_add_line(dl, x0, y0, x1, y0, col_purple, 3.0)
            dl_add_line(dl, x1, y0, x1, y1, col_purple, 3.0)
            dl_add_line(dl, x1, y1, x0, y1, col_purple, 3.0)
            dl_add_line(dl, x0, y1, x0, y0, col_purple, 3.0)
            return

        dets = getattr(telem, "detections", []) or []
        img_size = getattr(telem, "detect_img_size", None)
        if not dets or not img_size or img_size[0] in (None, 0) or img_size[1] in (None, 0):
            return

        img_w, img_h = float(img_size[0]), float(img_size[1])
        sx = fb_w / img_w
        sy = fb_h / img_h

        dl = imgui.get_foreground_draw_list()
        col_yellow = imgui.get_color_u32_rgba(1.0, 0.95, 0.2, 1.0)
        col_purple = imgui.get_color_u32_rgba(0.7, 0.4, 1.0, 1.0)

        for det in dets:
            bbox = det.get("bbox_px") or {}
            x = float(bbox.get("x", 0)) * sx
            y = float(bbox.get("y", 0)) * sy
            w = float(bbox.get("w", 0)) * sx
            h = float(bbox.get("h", 0)) * sy

            x0, y0, x1, y1 = x, y, x + w, y + h
            tid = det.get("id")
            is_sel = (tid is not None) and (tid == self.selected_track_id)

            col = col_purple if is_sel else col_yellow
            # box
            dl_add_line(dl, x0, y0, x1, y0, col, 3.0)
            dl_add_line(dl, x1, y0, x1, y1, col, 3.0)
            dl_add_line(dl, x1, y1, x0, y1, col, 3.0)
            dl_add_line(dl, x0, y1, x0, y0, col, 3.0)

            # label
            label = det.get("label", "") or ""
            conf = det.get("conf", None)
            if conf is not None:
                try:
                    label = f"{label} {float(conf):.2f}"
                except Exception:
                    label = f"{label} {conf}"
            tw = max(60.0, 10.0 * len(label))
            th = 24.0
            dl_add_rect_filled(
                dl,
                x0,
                max(0.0, y0 - th - 2),
                x0 + tw,
                max(0.0, y0 - 2),
                imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.7),
                6.0,
            )
            dl_add_text(dl, x0 + 6, max(0.0, y0 - th), (1, 1, 1, 1), label, font_scale=1.0)

    def handle_clicks(
        self,
        window,
        fb_w: int,
        fb_h: int,
        telem: Telemetry,
        *,
        send_command: Callable[[dict], bool],
        allow_cancel: bool,
        ignore_rect: Optional[Tuple[float, float, float, float]] = None,
    ) -> None:
        """Handle LMB clicks for starting/stopping tracking.
        - Click inside a bbox => send_command({"track_select":{"track_id":ID}})
        - Click empty space (if allow_cancel=True) => send_command({"track_cancel": True})
        - Clicks inside ignore_rect are ignored (tracking persists)
        """
        if self._roi_active or imgui.is_mouse_down(1):
            return
        if not imgui.is_mouse_clicked(0):
            return

        io = imgui.get_io()
        if io.want_capture_mouse:
            return

        mx_win, my_win = glfw.get_cursor_pos(window)

        if ignore_rect is not None:
            wx, wy, ww, wh = ignore_rect
            if (wx <= mx_win <= wx + ww) and (wy <= my_win <= wy + wh):
                return

        # Convert mouse to framebuffer coords (overlay is in fb space)
        ww_win, wh_win = glfw.get_window_size(window)
        if ww_win <= 0 or wh_win <= 0:
            return
        mx_fb = mx_win * (fb_w / float(ww_win))
        my_fb = my_win * (fb_h / float(wh_win))

        dets = getattr(telem, "detections", []) or []
        img_size = getattr(telem, "detect_img_size", None)
        if not img_size or img_size[0] in (None, 0) or img_size[1] in (None, 0):
            if allow_cancel:
                try:
                    send_command({"track_cancel": True})
                    self.selected_track_id = None
                    print("[CTRL] track_cancel (no image)")
                except Exception as e:
                    print("[CTRL] cancel send error:", e)
            return

        img_w, img_h = float(img_size[0]), float(img_size[1])
        sx = fb_w / img_w
        sy = fb_h / img_h

        # Hit-test
        clicked_id = None
        for det in dets:
            bbox = det.get("bbox_px") or {}
            x = float(bbox.get("x", 0)) * sx
            y = float(bbox.get("y", 0)) * sy
            w = float(bbox.get("w", 0)) * sx
            h = float(bbox.get("h", 0)) * sy
            if (mx_fb >= x) and (mx_fb <= x + w) and (my_fb >= y) and (my_fb <= y + h):
                try:
                    clicked_id = int(det.get("id"))
                except Exception:
                    clicked_id = None
                break

        if clicked_id is not None:
            self.selected_track_id = clicked_id
            try:
                send_command({"track_select": {"track_id": clicked_id}})
                print("[CTRL] track_select:", clicked_id)
            except Exception as e:
                print("[CTRL] track_select send error:", e)
        else:
            if allow_cancel:
                try:
                    send_command({"track_cancel": True})
                    self.selected_track_id = None
                    print("[CTRL] track_cancel (empty click)")
                except Exception as e:
                    print("[CTRL] track_cancel send error:", e)


# ----------------- COMPASS TAPE (true spinning wheel) -----------------
def draw_compass_tape(draw_fg, win_w: int, y: float, heading_deg: float):
    """
    Spinning compass tape:
      - Majors every 30°, half ticks every 15°
      - Ticks drawn from top downward, labels under ticks
      - All ticks computed as absolute headings (0..359) and positioned
        relative to current heading by a wrapped delta -> stable “wheel”
    """
    cx = win_w * 0.5
    width = 700.0
    height = 56.0
    top = y
    bottom = y + height
    left = cx - width * 0.5
    right = cx + width * 0.5

    # Background a tad taller to safely contain labels
    box_col = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.6)
    dl_add_rect_filled(draw_fg, left, top, right, bottom + 16.0, box_col, 8.0)

    col_tick = imgui.get_color_u32_rgba(1, 1, 1, 0.95)
    col_text = (1, 1, 1, 1)

    mid = float(_wrap360(heading_deg if heading_deg is not None else 0.0))
    px_per_deg = width / 360.0

    # Tick geometry
    major_h = 16.0
    half_h  = 10.0
    tick_top_y = top + 6.0
    label_y    = bottom - 18.0

    # Iterate absolute headings (0..359 step 15) and place them by wrapped delta to mid
    for deg in range(0, 360, 15):
        # delta in [-180, +180] so tape scrolls smoothly
        delta = _wrap180(deg - mid)
        x = cx + delta * px_per_deg
        if x < left + 6 or x > right - 6:
            continue

        is_major = (deg % 30) == 0
        h = major_h if is_major else half_h

        # Tick
        dl_add_line(draw_fg, x, tick_top_y, x, tick_top_y + h, col_tick, 2.0)

        # Label for majors, under the tick
        if is_major:
            label = f"{deg:03d}"
            # center text roughly by subtracting half label width (~24px for "000")
            dl_add_text(draw_fg, x - 12, label_y, col_text, label, font_scale=1.0)

    # Center indicator line
    center_col = imgui.get_color_u32_rgba(1, 1, 1, 0.9)
    dl_add_line(draw_fg, cx, top + 1, cx, bottom - 1, center_col, 2.0)


# ----------------- ALTITUDE LADDER (stable around any center, incl. negatives) -----------------
def draw_altitude_ladder(draw_fg, win_w: int, win_h: int, alt_m: float):
    """Right-side vertical altitude ladder. Ticks every 10 m, labels every 50 m."""
    width = 120.0
    height = 300.0
    x = win_w - width - 20
    y = (win_h * 0.5) - (height * 0.5)

    # Background
    box_col = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.5)
    dl_add_rect_filled(draw_fg, x, y, x + width, y + height, box_col, 8.0)

    col_tick = imgui.get_color_u32_rgba(1,1,1,0.9)
    col_text = (1,1,1,1)

    center = float(alt_m if alt_m is not None else 0.0)
    px_per_meter = height / 300.0

    # Visible tick range aligned to 10 m
    start_val = math.floor((center - 150.0) / 10.0) * 10.0
    end_val   = math.ceil((center + 150.0) / 10.0) * 10.0

    v = start_val
    while v <= end_val + 0.1:
        off = v - center
        yy = (y + height * 0.5) - (off * px_per_meter)
        if yy >= y + 10 and yy <= y + height - 10:
            major = (int(v) % 50) == 0
            w = 14 if major else 8
            dl_add_line(draw_fg, x + width - 2 - w, yy, x + width - 2, yy, col_tick, 2.0)
            if major:
                # label placed slightly *below* the tick line
                dl_add_text(draw_fg, x + 10, yy + 8, col_text, f"{int(v)}", font_scale=1.0)
        v += 10.0


# ----------------- ARMED / FPS (two-line pill) -----------------
def draw_arming_widget(draw_fg, x: float, y: float, armed: bool, fps: float):
    """Two-line status pill: line1 ARMED/DISARMED, line2 FPS."""
    w, h = 210.0, 48.0
    bg = imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 0.7)
    dl_add_rect_filled(draw_fg, x, y, x + w, y + h, bg, 10.0)

    col_state = (0.3, 1.0, 0.4, 1.0) if armed else (1.0, 0.4, 0.3, 1.0)
    state = "ARMED" if armed else "DISARMED"
    dl_add_text(draw_fg, x + 10, y + 6,  col_state,       state,             font_scale=1.0)
    dl_add_text(draw_fg, x + 10, y + 26, (1,1,1,1),       f"FPS: {fps:.1f}", font_scale=1.0)


class IconCache:
    def __init__(self):
        self._tex = {}

    def get(self, path: str):
        """Return (tex_id, w, h) for the image, caching the GL texture. If missing, return None."""
        if path in self._tex:
            return self._tex[path]
        if not os.path.exists(path):
            return None
        try:
            from PIL import Image
            img = Image.open(path).convert('RGBA')
            w, h = img.size
            tex = pil_to_gl_texture(img)
            self._tex[path] = (tex, w, h)
            return self._tex[path]
        except Exception:
            return None

    def shutdown(self) -> None:
        try:
            import OpenGL.GL as gl
        except Exception:
            self._tex.clear()
            return
        for val in list(self._tex.values()):
            try:
                tex = int(val[0])
                gl.glDeleteTextures(tex)
            except Exception:
                pass
        self._tex.clear()


class OSDView:
    """
    Draws:
      - Icon+text elements from JSON (osd_config.json -> "elements": [{type:"icon_text", ...}])
      - Bottom-right logo (assets/osd/logo.png)
      - Artificial horizon + pitch ladder
      - Compass tape (top)
      - Altitude ladder (right)
      - Arming widget (two-line)
      - RC stick boxes (adjacent; left from JSON, right = left + gap)
    """

    def __init__(self, osd_cfg_getter, fps_getter):
        self.get_cfg = osd_cfg_getter
        self.get_fps = fps_getter
        self.icons = IconCache()

    # --- Compatibility shim: accept both call styles ---
    #   1) draw(win_w, win_h, telem)
    #   2) draw(telem, (win_w, win_h))
    def draw(self, *args):
        if len(args) == 3 and isinstance(args[0], (int, float)) and isinstance(args[1], (int, float)):
            win_w, win_h, telem = int(args[0]), int(args[1]), args[2]
        elif len(args) == 2:
            a0, a1 = args
            # (telem, (w,h))
            if hasattr(a0, "__dict__") and isinstance(a1, tuple) and len(a1) == 2:
                telem, (win_w, win_h) = a0, (int(a1[0]), int(a1[1]))
            # ((w,h), telem)
            elif isinstance(a0, tuple) and len(a0) == 2 and hasattr(a1, "__dict__"):
                (win_w, win_h), telem = (int(a0[0]), int(a0[1])), a1
            else:
                raise TypeError("OSDView.draw expected (win_w, win_h, telem) or (telem, (win_w, win_h))")
        else:
            raise TypeError("OSDView.draw expected (win_w, win_h, telem) or (telem, (win_w, win_h))")

        draw_fg = imgui.get_foreground_draw_list()
        cfg = self.get_cfg()

        # 1) Icon+text cards
        for it in cfg.get("elements", []):
            if not it.get("enabled", True):
                continue
            if (it.get("type") or "").lower() != "icon_text":
                continue

            x = float(it.get("x", 0))
            y = float(it.get("y", 0))
            iw, ih = it.get("size_px", [64, 64])
            box = bool(it.get("box", True))
            font_scale = float(it.get("font_scale", 1.2))
            color = tuple(it.get("color_rgba", [1, 1, 1, 1]))
            icon_path = os.path.join(OSD_ASSETS_DIR, it.get("file", ""))
            tex = self.icons.get(icon_path)

            # Native icon size if available; do not scale with box size
            if isinstance(tex, tuple):
                tex_id, tex_w, tex_h = tex
                tex_w, tex_h = float(tex_w), float(tex_h)
            else:
                tex_id, tex_w, tex_h = (tex, float(iw), float(ih)) if tex else (None, float(iw), float(ih))

            # Card grows to fit icon if needed
            card_w = max(float(iw), tex_w)
            card_h = max(float(ih), tex_h, 30.0)

            # Background card
            if box:
                box_col = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 0.65)
                dl_add_rect_filled(draw_fg, x - 6, y - 6, x + card_w + 240, y + card_h + 10, box_col, 8.0)

            # Icon at native size
            if tex_id:
                dl_add_image(draw_fg, tex_id, x, y, tex_w, tex_h, flip_v=True)

            # Text (single or multi-line)
            text_x = x + tex_w + 10
            text_y = y + 4
            if "lines" in it:
                for idx, ln in enumerate(it["lines"]):
                    src = ln.get("source", "")
                    fmt = ln.get("fmt", "{value}")
                    val = getattr(telem, src, "")
                    s = _format_telem_value(str(fmt), val)
                    dl_add_text(draw_fg, text_x, text_y + idx * 22, color, s, font_scale=font_scale)
            else:
                src = it.get("source", "")
                fmt = it.get("fmt", "{value}")
                val = getattr(telem, src, "")
                s = _format_telem_value(str(fmt), val)
                dl_add_text(draw_fg, text_x, text_y, color, s, font_scale=font_scale)

        # 2) Bottom-right logo
        logo_tex = self.icons.get(os.path.join(OSD_ASSETS_DIR, "logo.png"))
        if logo_tex:
            if isinstance(logo_tex, tuple):
                logo_tex_id = logo_tex[0]
            else:
                logo_tex_id = logo_tex
            lw, lh = 160, 60
            x0 = win_w - lw - 20
            y0 = win_h - lh - 20
            box_col = imgui.get_color_u32_rgba(0.15, 0.15, 0.15, 0.6)
            dl_add_rect_filled(draw_fg, x0 - 6, y0 - 6, x0 + lw + 6, y0 + lh + 6, box_col, 8.0)
            dl_add_image(draw_fg, logo_tex_id, x0, y0, lw, lh, flip_v=True)

        # 3) Tapes & arming widgets
        draw_compass_tape(draw_fg, win_w, y=20, heading_deg=getattr(telem, "heading_deg", 0.0))

        # --- FIX: altitude source preference (rel_alt_m -> alt_amsl_m -> alt_m) ---
        alt_val = getattr(telem, "rel_alt_m", None)
        if alt_val is None:
            alt_val = getattr(telem, "alt_amsl_m", None)
        if alt_val is None:
            alt_val = getattr(telem, "alt_m", 0.0)
        draw_altitude_ladder(draw_fg, win_w, win_h, alt_m=alt_val)

        draw_arming_widget(draw_fg, x=1750, y=20, armed=bool(getattr(telem, "armed", False)), fps=self.get_fps())

        # 4) Central artificial horizon + pitch ladder
        self._draw_horizon_and_ladder(draw_fg, win_w, win_h, telem.roll, telem.pitch)

        # 5) Sticks
        self._draw_sticks(draw_fg, win_w, win_h, telem)

    # -------- internals --------
    def _draw_horizon_and_ladder(self, draw_fg, win_w, win_h, roll_rad: float, pitch_rad: float):
        """
        Artificial horizon drawn through the screen center, rotating around the center.
        Pitch ladder slides along the horizon's own normal and rotates with it.
        """
        cx, cy = win_w * 0.5, win_h * 0.5
        L = min(win_w, win_h) * 0.45

        # Pixels per radian of pitch
        pp_rad = 180.0

        # Unit vectors: u = along horizon, n = its normal ("up" from the line)
        rr = roll_rad or 0.0
        u = (math.cos(rr), math.sin(rr))
        n = (-math.sin(rr), math.cos(rr))

        # Horizon line through the center
        hx0 = cx - L * u[0]
        hy0 = cy - L * u[1]
        hx1 = cx + L * u[0]
        hy1 = cy + L * u[1]
        col = imgui.get_color_u32_rgba(1, 1, 1, 0.9)
        dl_add_line(draw_fg, hx0, hy0, hx1, hy1, col, 3.0)

        # Pitch ladder lines and labels
        wx = 60.0  # half-length of each ladder segment
        for off_deg in range(-30, 31, 10):
            if off_deg == 0:
                continue
            off_rad = _rad(off_deg)
            d = (pitch_rad or 0.0) + off_rad      # radians
            offset = -d * pp_rad                  # "+pitch" moves ladder "down" (ADI convention)

            # Center point of this ladder line
            px = cx + offset * n[0]
            py = cy + offset * n[1]

            # Endpoints along horizon direction
            lx0 = px - wx * u[0]
            ly0 = py - wx * u[1]
            lx1 = px + wx * u[0]
            ly1 = py + wx * u[1]

            dl_add_line(draw_fg, lx0, ly0, lx1, ly1, col, 2.0)

            # Label near right end, *above* the line along -n
            label_off_n = -10.0
            label_off_px = 0.0
            tx = lx1 + label_off_px * u[0] + label_off_n * n[0]
            ty = ly1 + label_off_px * u[1] + label_off_n * n[1] - 6.0
            dl_add_text(draw_fg, tx, ty, (1,1,1,1), f"{off_deg:+d}", font_scale=1.0)

    def _draw_compass_ring(self, draw_fg, win_w, win_h, yaw_rad: float, roll_rad: float):
        """(optional) Heading ray from center."""
        cx, cy = win_w * 0.5, win_h * 0.5
        R = 160.0
        col = imgui.get_color_u32_rgba(1, 1, 1, 0.9)
        ang = yaw_rad or 0.0
        x0 = cx + R * math.cos(ang)
        y0 = cy + R * math.sin(ang)
        dl_add_line(draw_fg, cx, cy, x0, y0, col, 2.0)

    # ----------------- sticks (adjacent; only LEFT configured, RIGHT = LEFT + gap) -----------------
    def _read_left_pos(self, cfg: dict, win_w: int, win_h: int, box_w: float, box_h: float):
        """
        Read left stick position from existing JSON keys (no new keys required).
        Supported patterns (examples):
          - cfg["sticks"]["left"]     -> {"x": 20, "y": 860}  or  {"pos":[20,860]}
          - cfg["sticks"]["left_pos"] -> [20, 860]
          - cfg["sticks_left"]        -> {"x": 20, "y": 860}  or  {"pos":[20,860]}
          - cfg["sticks_left_pos"]    -> [20, 860]
        Falls back to bottom-left if none found.
        """
        def clamp_xy(x, y):
            x = max(0, min(int(x), int(win_w - box_w)))
            y = max(0, min(int(y), int(win_h - box_h)))
            return float(x), float(y)

        sticks = cfg.get("sticks", {})
        if isinstance(sticks, dict):
            node = sticks.get("left")
            if isinstance(node, dict):
                if "pos" in node and isinstance(node["pos"], (list, tuple)) and len(node["pos"]) == 2:
                    return clamp_xy(node["pos"][0], node["pos"][1])
                if "x" in node and "y" in node:
                    return clamp_xy(node["x"], node["y"])
            node = sticks.get("left_pos")
            if isinstance(node, (list, tuple)) and len(node) == 2:
                return clamp_xy(node[0], node[1])

        node = cfg.get(f"sticks_left")
        if isinstance(node, dict):
            if "pos" in node and isinstance(node["pos"], (list, tuple)) and len(node["pos"]) == 2:
                return clamp_xy(node["pos"][0], node["pos"][1])
            if "x" in node and "y" in node:
                return clamp_xy(node["x"], node["y"])

        node = cfg.get(f"sticks_left_pos")
        if isinstance(node, (list, tuple)) and len(node) == 2:
            return clamp_xy(node[0], node[1])

        # fallback
        return clamp_xy(20, win_h - box_h - 20)

    def _draw_sticks(self, draw_fg, win_w, win_h, telem: Telemetry):
        """Two small boxes for RC sticks; adjacent placement. Configure only LEFT; RIGHT = LEFT + gap."""
        cfg = self.get_cfg()
        sticks_cfg = cfg.get("sticks", {}) if isinstance(cfg.get("sticks", {}), dict) else {}

        # DEFAULTS: half size (90x90) and 16px gap (backward-compatible)
        default_size = [90, 90]
        size_px = sticks_cfg.get("size_px", default_size)
        try:
            box_w, box_h = float(size_px[0]), float(size_px[1])
        except Exception:
            box_w, box_h = float(default_size[0]), float(default_size[1])

        gap = float(sticks_cfg.get("gap", 16.0))

        # Left from JSON (or fallback), Right derived from Left + gap
        lx0, ly0 = self._read_left_pos(cfg, win_w, win_h, box_w, box_h)
        rx0 = min(max(lx0 + box_w + gap, 0.0), max(0.0, win_w - box_w))
        ry0 = ly0

        lx1, ly1 = lx0 + box_w, ly0 + box_h
        rx1, ry1 = rx0 + box_w, ry0 + box_h

        col_bg = imgui.get_color_u32_rgba(0.08, 0.08, 0.08, 0.5)
        col_fg = imgui.get_color_u32_rgba(1,1,1,0.9)
        col_dot = imgui.get_color_u32_rgba(0.3,1.0,0.4,1.0)

        # Boxes
        dl_add_rect_filled(draw_fg, lx0, ly0, lx1, ly1, col_bg, 8.0)
        dl_add_rect_filled(draw_fg, rx0, ry0, rx1, ry1, col_bg, 8.0)

        # Crosshairs
        dl_add_line(draw_fg, (lx0+lx1)*0.5, ly0+6, (lx0+lx1)*0.5, ly1-6, col_fg, 1.5)
        dl_add_line(draw_fg, lx0+6, (ly0+ly1)*0.5, lx1-6, (ly0+ly1)*0.5, col_fg, 1.5)
        dl_add_line(draw_fg, (rx0+rx1)*0.5, ry0+6, (rx0+rx1)*0.5, ry1-6, col_fg, 1.5)
        dl_add_line(draw_fg, rx0+6, (ry0+ry1)*0.5, rx1-6, (ry0+ry1)*0.5, col_fg, 1.5)

        # Centers
        lcx, lcy = (lx0+lx1)*0.5, (ly0+ly1)*0.5
        rcx, rcy = (rx0+rx1)*0.5, (ry0+ry1)*0.5

        # Stick values (robust fallbacks)
        roll  = _clamp(getattr(telem, "roll", 0.0) / 0.6, -1.0, 1.0)
        pitch = _clamp(getattr(telem, "pitch", 0.0) / 0.6, -1.0, 1.0)
        yaw   = _clamp(getattr(telem, "yaw", 0.0) / 0.6, -1.0, 1.0)

        thr = getattr(telem, "throttle_norm", None)
        if thr is None:
            thr = getattr(telem, "throttle", None)
        if thr is None:
            pct = getattr(telem, "throttle_pct", None)
            thr = (float(pct) / 100.0) if pct is not None else 0.5
        thr = _clamp(float(thr), 0.0, 1.0)

        # Left dot (roll/pitch)
        half_w = (lx1 - lx0) * 0.5
        half_h = (ly1 - ly0) * 0.5
        draw_fg.add_circle_filled(
            float(lcx + roll * (half_w - 10.0)),
            float(lcy - pitch * (half_h - 10.0)),
            5.5, col_dot, 20
        )

        # Right dot (yaw/throttle)
        r_half_w = (rx1 - rx0) * 0.5
        r_half_h = (ry1 - ry0) * 0.5
        dot_rx = rcx + yaw * (r_half_w - 10.0)
        dot_ry = ry1 - thr * (2.0 * (r_half_h - 10.0))
        dot_rx = _clamp(dot_rx, rcx - (r_half_w - 10.0), rcx + (r_half_w - 10.0))
        dot_ry = _clamp(dot_ry, rcy - (r_half_h - 10.0), rcy + (r_half_h - 10.0))
        draw_fg.add_circle_filled(float(dot_rx), float(dot_ry), 5.5, col_dot, 20)
