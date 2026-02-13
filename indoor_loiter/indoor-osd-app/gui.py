#!/usr/bin/env python3
import json
import math
import time
import imgui, OpenGL.GL as gl
import glfw
from typing import Any, Callable, Dict, List, Optional, Tuple
from utils import dl_add_image, dl_add_line, dl_add_rect_filled, dl_add_text, load_json, pil_to_gl_texture, plot_line, save_json
from osd import DetectionsOverlay

def build_placeholder_map(w:int, h:int)->int:
    from PIL import Image
    img = Image.new("RGBA", (w, h), (30, 30, 30, 255))
    return pil_to_gl_texture(img)


# ----------------- app modes -----------------
MODE_OSD = 0
MODE_LOCAL_MAP = 1
MODE_MAP = 2
MODE_VIDEO = 3
MODE_CONTROL = 4
MODE_CONFIG = 5
MODE_TRACK = 6

MODE_LABELS = {
    MODE_OSD: "OSD",
    MODE_LOCAL_MAP: "Local Map",
    MODE_MAP: "GPS Map",
    MODE_VIDEO: "Video Settings",
    MODE_CONTROL: "Control",
    MODE_CONFIG: "Config",
    MODE_TRACK: "PID Tune",
}

TRACK_MODE_OFF = 0
TRACK_MODE_HOLE_3D = 1
TRACK_MODE_PLANE_3D = 2
TRACK_MODE_ROI_2D = 3

TRACK_MODE_LABELS = {
    TRACK_MODE_OFF: "Tracking off",
    TRACK_MODE_HOLE_3D: "3D hole tracker",
    TRACK_MODE_PLANE_3D: "3D plane tracker",
    TRACK_MODE_ROI_2D: "2D ROI tracker",
}


def draw_fullscreen(tex: int, fb_w: int, fb_h: int) -> None:
    dl = imgui.get_background_draw_list()
    dl_add_image(dl, tex, 0, 0, float(fb_w), float(fb_h))


def draw_video_widget(
    *,
    tex: Optional[int],
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
) -> None:
    dl = imgui.get_background_draw_list()
    if tex:
        dl_add_image(dl, int(tex), x, y, w, h)
    else:
        dl_add_rect_filled(dl, x, y, x + w, y + h, (0, 0, 0, 1), 0.0)
        dl_add_text(dl, x + 8, y + 8, (1, 0.3, 0.3, 1), "No video", font_scale=1.0)

    dl.add_rect(
        float(x),
        float(y),
        float(x + w),
        float(y + h),
        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.85),
        0.0,
        0,
        2.0,
    )
    dl_add_text(dl, x + 8, y + h - 26, (1, 1, 1, 0.9), label, font_scale=1.0)


class WizardFooter:
    def __init__(self):
        self.rect: Tuple[float, float, float, float] = (0, 0, 0, 0)

    def draw(self, win_w: int, win_h: int, mode: int) -> int:
        imgui.set_next_window_bg_alpha(0.35)
        imgui.set_next_window_position(20, win_h - 70)
        imgui.begin(
            "wizard",
            False,
            imgui.WINDOW_NO_TITLE_BAR | imgui.WINDOW_ALWAYS_AUTO_RESIZE | imgui.WINDOW_NO_MOVE,
        )

        # record rect so clicks here don't cancel tracking
        wx, wy = imgui.get_window_position()
        ww, wh = imgui.get_window_size()
        self.rect = (float(wx), float(wy), float(ww), float(wh))

        mode_count = len(MODE_LABELS)
        if imgui.button("< Prev", 160, 0):
            mode = (int(mode) - 1) % mode_count

        imgui.same_line()

        # Fixed-width label so the "Next" button doesn't shift.
        cur_text = f"Mode: {MODE_LABELS.get(int(mode), str(mode))}"
        cur_w, _ = imgui.calc_text_size(cur_text)
        max_w = 0.0
        for lbl in MODE_LABELS.values():
            w, _ = imgui.calc_text_size(f"Mode: {lbl}")
            max_w = max(max_w, float(w))

        imgui.text(cur_text)
        imgui.same_line()
        pad = max(0.0, float(max_w) - float(cur_w))
        if pad > 0:
            imgui.dummy(pad, 0)
            imgui.same_line()

        if imgui.button("Next >", 160, 0):
            mode = (int(mode) + 1) % mode_count

        imgui.end()
        return int(mode)


class OsdLayoutEditor:
    def __init__(self, cfg_getter: Callable[[], dict], cfg_setter: Callable[[dict], None], path: str):
        self.get_cfg = cfg_getter
        self.set_cfg = cfg_setter
        self.path = path
        self.config_text = json.dumps(self.get_cfg(), indent=2)
        self.config_text_orig = self.config_text
        self._error: Optional[str] = None

    def draw_window(self, win_w: int, win_h: int) -> None:
        imgui.set_next_window_position(40, 40)
        imgui.set_next_window_size(win_w - 80, win_h - 160)
        imgui.begin("OSD Layout JSON Editor", True, imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_COLLAPSE)

        changed, self.config_text = imgui.input_text_multiline(
            "##json",
            self.config_text,
            1 << 20,
            width=win_w - 120,
            height=win_h - 260,
            flags=imgui.INPUT_TEXT_ALLOW_TAB_INPUT,
        )
        if changed and self._error:
            self._error = None

        if imgui.button("Save", 180, 0):
            try:
                new_cfg = json.loads(self.config_text)
            except Exception as e:
                self._error = f"Invalid JSON: {e}"
            else:
                try:
                    save_json(self.path, new_cfg)
                    self.set_cfg(new_cfg)
                    self.config_text_orig = self.config_text
                    self._error = None
                except Exception as e:
                    self._error = f"Save failed: {e}"

        imgui.same_line()
        if imgui.button("Cancel", 160, 0):
            try:
                cfg = load_json(self.path)
                self.set_cfg(cfg)
                self.config_text = json.dumps(cfg, indent=2)
                self.config_text_orig = self.config_text
                self._error = None
            except Exception as e:
                self._error = f"Reload failed: {e}"

        if self._error:
            imgui.same_line()
            imgui.text_colored(self._error, 1, 0.3, 0.3, 1)

        imgui.end()


class PidTuneWindow:
    def draw(self, win_w: int, win_h: int, telem, video_tex: Optional[int], fb_w: int, fb_h: int) -> None:
        """
        PID Debug window:
        - almost fullscreen (leaves space for the wizard footer)
        - stacked plots spanning width (CX, CY, Yaw, Pitch, Thrust)
        """
        # Leave bottom space for the wizard buttons (~70px footer + margin)
        bottom_reserved = 90
        left_margin, top_margin, right_margin = 10, 10, 10

        if video_tex:
            draw_fullscreen(int(video_tex), fb_w, fb_h)

        # Window placement & sizing
        win_x = left_margin
        win_y = top_margin
        win_wi = max(320, win_w - (left_margin + right_margin))
        win_hi = max(200, win_h - bottom_reserved - top_margin)

        imgui.set_next_window_position(win_x, win_y)
        imgui.set_next_window_size(win_wi, win_hi)
        imgui.begin("PID Tune", False, imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE)

        snap = telem

        # Header (only the fields you wanted displayed)
        pid_state = getattr(snap, "pid_state", "idle")
        gate_rc = getattr(snap, "pid_gate_rc", None)
        gate_mode = getattr(snap, "pid_gate_mode", None)

        cx_px = getattr(snap, "pid_cx_px", None)
        cy_px = getattr(snap, "pid_cy_px", None)
        ex_px = getattr(snap, "pid_ex_px", None)
        ey_px = getattr(snap, "pid_ey_px", None)
        nx = getattr(snap, "pid_nx", None)
        ny = getattr(snap, "pid_ny", None)

        yaw_cmd = getattr(snap, "pid_yaw", None)
        pitch_cmd = getattr(snap, "pid_pitch", None)
        thrust_cmd = getattr(snap, "pid_thrust", None)

        hist = list(getattr(snap, "pid_hist", []))[-300:]

        xs = [t for (t, _, _, _, _, _) in hist]
        cxs = [v for (_, v, _, _, _, _) in hist]
        cys = [v for (_, _, v, _, _, _) in hist]
        yws = [v for (_, _, _, v, _, _) in hist]
        pts = [v for (_, _, _, _, v, _) in hist]
        ths = [v for (_, _, _, _, _, v) in hist]

        # Filter out None and trim to last 200
        def fv(x, y):
            xx, yy = [], []
            for a, b in zip(x, y):
                if b is not None:
                    xx.append(a)
                    yy.append(b)
            return xx[-200:], yy[-200:]

        xs_cx, cxs = fv(xs, cxs)
        xs_cy, cys = fv(xs, cys)
        xs_yw, yws = fv(xs, yws)
        xs_pt, pts = fv(xs, pts)
        xs_th, ths = fv(xs, ths)

        # Fixed plot ranges (disable autoscale): use best-known image size for pixel plots.
        img_w = img_h = None
        try:
            img_size = getattr(snap, "acq_img_size", None) or getattr(snap, "detect_img_size", None)
            if isinstance(img_size, (tuple, list)) and len(img_size) >= 2:
                img_w = int(img_size[0])
                img_h = int(img_size[1])
        except Exception:
            img_w = img_h = None
        if not img_w:
            img_w = int(fb_w) if fb_w > 0 else int(win_wi)
        if not img_h:
            img_h = int(fb_h) if fb_h > 0 else int(win_hi)
        img_w = max(1, int(img_w))
        img_h = max(1, int(img_h))

        # Header lines
        rc_txt = "—" if gate_rc is None else ("ON" if gate_rc else "OFF")
        ok_txt = "—" if gate_mode is None else ("YES" if gate_mode else "NO")
        imgui.text(f"Tracking state: {pid_state}")
        imgui.same_line()
        imgui.text(f"  RC Gate: {rc_txt}   |   Gate OK: {ok_txt} ")
        imgui.same_line()
        imgui.text(
            (f"yaw: {yaw_cmd:+.3f}" if yaw_cmd is not None else "yaw: —")
            + "   "
            + (f"pitch: {pitch_cmd:+.3f}" if pitch_cmd is not None else "pitch: —")
            + "   "
            + (f"thrust: {thrust_cmd:+.3f}" if thrust_cmd is not None else "thrust: —")
        )

        imgui.text(
            (f"cx_px: {cx_px:.1f}" if cx_px is not None else "cx_px: —")
            + "   "
            + (f"cy_px: {cy_px:.1f}" if cy_px is not None else "cy_px: —")
            + "   |   "
            + (f"ex_px: {ex_px:.1f}" if ex_px is not None else "ex_px: —")
            + "   "
            + (f"ey_px: {ey_px:.1f}" if ey_px is not None else "ey_px: —")
            + "   |   "
            + (f"nx: {nx:.3f}" if nx is not None else "nx: —")
            + "   "
            + (f"ny: {ny:.3f}" if ny is not None else "ny: —")
        )

        # Draw stacked plots (span the window width)
        dl = imgui.get_foreground_draw_list()
        origin_x, origin_y = imgui.get_cursor_screen_pos()

        # Account for some inner padding
        inner_pad = 10
        plot_x = origin_x + inner_pad
        plot_w = win_wi - (inner_pad * 2) - 20
        row_h = 120
        row_gap = 49

        # Row 1: CX
        plot_y = origin_y + 22
        plot_line(
            dl,
            plot_x,
            plot_y,
            plot_w,
            row_h,
            xs_cx,
            cxs,
            title="CX (px)",
            fmt_val="{:+.1f}",
            show_minmax=True,
            y_min=0.0,
            y_max=float(img_w),
        )

        # Row 2: CY
        plot_y += row_h + row_gap
        plot_line(
            dl,
            plot_x,
            plot_y,
            plot_w,
            row_h,
            xs_cy,
            cys,
            title="CY (px)",
            fmt_val="{:+.1f}",
            show_minmax=True,
            y_min=0.0,
            y_max=float(img_h),
        )

        # Row 3: Yaw
        plot_y += row_h + row_gap
        plot_line(
            dl,
            plot_x,
            plot_y,
            plot_w,
            row_h,
            xs_yw,
            yws,
            title="Yaw cmd",
            fmt_val="{:+.3f}",
            show_minmax=True,
            y_min=-1.0,
            y_max=1.0,
        )

        # Row 4: Pitch
        plot_y += row_h + row_gap
        plot_line(
            dl,
            plot_x,
            plot_y,
            plot_w,
            row_h,
            xs_pt,
            pts,
            title="Pitch cmd",
            fmt_val="{:+.3f}",
            show_minmax=True,
            y_min=-1.0,
            y_max=1.0,
        )

        # Row 5: Thrust
        plot_y += row_h + row_gap
        plot_line(
            dl,
            plot_x,
            plot_y,
            plot_w,
            row_h,
            xs_th,
            ths,
            title="Thrust cmd",
            fmt_val="{:+.3f}",
            show_minmax=True,
            y_min=-1.0,
            y_max=1.0,
        )

        # Advance ImGui cursor so later widgets (if any) don’t overlap drawings
        total_h = (plot_y + row_h) - origin_y
        imgui.dummy(plot_w, total_h + 6)

        imgui.end()


class LocalMapView:
    def __init__(self):
        # Display scale: pixels per 1 meter square.
        self.px_per_m: float = 122.0

        # World coordinate at the center of the view (meters, in the local frame).
        self.center_m: Tuple[float, float] = (0.0, 0.0)

        # First received pose becomes local origin (so the map is "local" even if the VO frame is global).
        self._origin_xy: Optional[Tuple[float, float]] = None

        # Path points in local meters [(x,y), ...]
        self._path: List[Tuple[float, float]] = []
        self._last_pose_rx_count: int = 0

    def _ingest_pose(self, telem) -> None:
        try:
            rx = int(getattr(telem, "pose_rx_count", 0) or 0)
        except Exception:
            rx = 0
        if rx == self._last_pose_rx_count:
            return
        self._last_pose_rx_count = int(rx)

        ok = bool(getattr(telem, "vis_ok", False))
        x = getattr(telem, "vis_x", None)
        y = getattr(telem, "vis_y", None)
        if not ok or x is None or y is None:
            return
        try:
            x = float(x)
            y = float(y)
        except Exception:
            return

        if self._origin_xy is None:
            self._origin_xy = (x, y)
            self._path = [(0.0, 0.0)]
            self.center_m = (0.0, 0.0)
            return

        ox, oy = self._origin_xy
        lx = float(x) - float(ox)
        ly = float(y) - float(oy)

        # De-noise: ignore tiny movements so we don't draw dense jitter at high telemetry rates.
        if self._path:
            px, py = self._path[-1]
            dx = lx - float(px)
            dy = ly - float(py)
            if (dx * dx + dy * dy) < (0.02 * 0.02):  # <2 cm
                return

        self._path.append((float(lx), float(ly)))
        if len(self._path) > 20000:
            self._path = self._path[-20000:]

    def _world_to_screen(
        self,
        *,
        x_m: float,
        y_m: float,
        origin_x: float,
        origin_y: float,
        w_px: float,
        h_px: float,
    ) -> Tuple[float, float]:
        cx_m, cy_m = self.center_m
        pxm = float(self.px_per_m)
        sx = float(origin_x) + (float(w_px) * 0.5) + (float(x_m) - float(cx_m)) * pxm
        sy = float(origin_y) + (float(h_px) * 0.5) - (float(y_m) - float(cy_m)) * pxm
        return float(sx), float(sy)

    def draw(self, *, win_w: int, win_h: int, fb_w: int, fb_h: int, telem) -> None:
        self._ingest_pose(telem)

        # ---- Window (same margins as other modes) ----
        imgui.set_next_window_position(40, 40)
        imgui.set_next_window_size(max(320, int(win_w) - 80), max(200, int(win_h) - 160))
        imgui.begin("Local Map", True, imgui.WINDOW_NO_COLLAPSE)

        # Clamp zoom scale
        self.px_per_m = max(6.0, min(240.0, float(self.px_per_m)))

        # Controls row: zoom (left) + pan (right)
        start_x, start_y = imgui.get_cursor_screen_pos()
        win_pos = imgui.get_window_position()
        win_sz = imgui.get_window_size()
        pad_x, _pad_y = imgui.get_style().window_padding
        spacing_x, spacing_y = imgui.get_style().item_spacing

        content_left = float(start_x)
        content_right = float(win_pos.x) + float(win_sz.x) - float(pad_x)

        if imgui.button("Clear map", 140, 0):
            self._origin_xy = None
            self._path = []
            self.center_m = (0.0, 0.0)
            self._last_pose_rx_count = 0
        imgui.same_line()
        if imgui.button("Zoom In", 140, 0):
            self.px_per_m = min(240.0, float(self.px_per_m) * 1.25)
        imgui.same_line()
        if imgui.button("Zoom Out", 140, 0):
            self.px_per_m = max(6.0, float(self.px_per_m) / 1.25)
        imgui.same_line()
        imgui.text(f"Scale: 1m = {float(self.px_per_m):.0f}px")

        btn_w = 100.0
        btn_h = float(imgui.get_frame_height())
        pan_w = (btn_w * 3.0) + (float(spacing_x) * 2.0)
        pan_h = (btn_h * 3.0) + (float(spacing_y) * 2.0)
        pan_x = max(content_left, content_right - pan_w)

        step_m = 1.0
        row_step = float(btn_h) + float(spacing_y)
        # Row 1 (Up centered)
        imgui.set_cursor_screen_pos((pan_x, float(start_y)))
        imgui.dummy(btn_w, btn_h)
        imgui.same_line()
        if imgui.button("Up", btn_w, btn_h):
            cx_m, cy_m = self.center_m
            self.center_m = (float(cx_m), float(cy_m) - step_m)
        imgui.same_line()
        imgui.dummy(btn_w, btn_h)

        # Row 2 (Left / Right)
        imgui.set_cursor_screen_pos((pan_x, float(start_y) + row_step))
        if imgui.button("Left", btn_w, btn_h):
            cx_m, cy_m = self.center_m
            self.center_m = (float(cx_m) + step_m, float(cy_m))
        imgui.same_line()
        imgui.dummy(btn_w, btn_h)
        imgui.same_line()
        if imgui.button("Right", btn_w, btn_h):
            cx_m, cy_m = self.center_m
            self.center_m = (float(cx_m) - step_m, float(cy_m))

        # Row 3 (Down centered)
        imgui.set_cursor_screen_pos((pan_x, float(start_y) + row_step * 2.0))
        imgui.dummy(btn_w, btn_h)
        imgui.same_line()
        if imgui.button("Down", btn_w, btn_h):
            cx_m, cy_m = self.center_m
            self.center_m = (float(cx_m), float(cy_m) + step_m)
        imgui.same_line()
        imgui.dummy(btn_w, btn_h)

        # Move cursor below the pan controls and resume at left.
        imgui.set_cursor_screen_pos((content_left, float(start_y) + pan_h + float(spacing_y)))
        imgui.separator()

        # ---- Canvas (black) + 1m grid ----
        try:
            avail = imgui.get_content_region_available()
            canvas_w = float(avail.x)
            canvas_h = float(avail.y)
        except Exception:
            canvas_w = float(max(2, int(win_w) - 120))
            canvas_h = float(max(2, int(win_h) - 260))

        canvas_w = max(2.0, canvas_w)
        canvas_h = max(2.0, canvas_h)
        canvas_x, canvas_y = imgui.get_cursor_screen_pos()

        # Auto-zoom out when the VO path approaches the edge of the view.
        # Keeps at least ~1-2 "grid squares" of margin so the path doesn't run into the
        # outermost visible rectangle.
        if self._path:
            try:
                lx_m, ly_m = self._path[-1]
                lx_m, ly_m = float(lx_m), float(ly_m)
            except Exception:
                lx_m = ly_m = None

            if lx_m is not None and ly_m is not None:
                try:
                    cx_m, cy_m = self.center_m
                except Exception:
                    cx_m, cy_m = (0.0, 0.0)

                pxm = float(self.px_per_m)
                if pxm > 1e-6:
                    half_w_m = (canvas_w * 0.5) / pxm
                    half_h_m = (canvas_h * 0.5) / pxm
                    dx_m = abs(float(lx_m) - float(cx_m))
                    dy_m = abs(float(ly_m) - float(cy_m))

                    trigger_margin_m = 1.0
                    target_margin_m = 2.0

                    req_x = pxm
                    if (half_w_m - float(dx_m)) < float(trigger_margin_m):
                        denom = float(dx_m) + float(target_margin_m)
                        if denom > 1e-6:
                            req_x = (canvas_w * 0.5) / denom

                    req_y = pxm
                    if (half_h_m - float(dy_m)) < float(trigger_margin_m):
                        denom = float(dy_m) + float(target_margin_m)
                        if denom > 1e-6:
                            req_y = (canvas_h * 0.5) / denom

                    new_pxm = min(float(pxm), float(req_x), float(req_y))
                    self.px_per_m = max(6.0, min(240.0, float(new_pxm)))

        draw = imgui.get_window_draw_list()
        dl_add_rect_filled(draw, canvas_x, canvas_y, canvas_x + canvas_w, canvas_y + canvas_h, (0, 0, 0, 1), 0.0)

        col_grid = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.30)
        col_axis = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.60)

        pxm = float(self.px_per_m)
        cx_m, cy_m = self.center_m
        half_w_m = (canvas_w * 0.5) / pxm if pxm > 1e-6 else 0.0
        half_h_m = (canvas_h * 0.5) / pxm if pxm > 1e-6 else 0.0

        x0 = int(math.floor(float(cx_m) - half_w_m))
        x1 = int(math.ceil(float(cx_m) + half_w_m))
        y0 = int(math.floor(float(cy_m) - half_h_m))
        y1 = int(math.ceil(float(cy_m) + half_h_m))

        for xi in range(x0, x1 + 1):
            sx, _ = self._world_to_screen(
                x_m=float(xi),
                y_m=float(cy_m),
                origin_x=float(canvas_x),
                origin_y=float(canvas_y),
                w_px=canvas_w,
                h_px=canvas_h,
            )
            col = col_axis if xi == 0 else col_grid
            dl_add_line(draw, sx, float(canvas_y), sx, float(canvas_y) + canvas_h, col, 1.0)

        for yi in range(y0, y1 + 1):
            _, sy = self._world_to_screen(
                x_m=float(cx_m),
                y_m=float(yi),
                origin_x=float(canvas_x),
                origin_y=float(canvas_y),
                w_px=canvas_w,
                h_px=canvas_h,
            )
            col = col_axis if yi == 0 else col_grid
            dl_add_line(draw, float(canvas_x), sy, float(canvas_x) + canvas_w, sy, col, 1.0)

        # ---- Path drawing (white) + drone (red dot) ----
        pts: List[Tuple[float, float]] = []
        if self._path:
            for (x_m, y_m) in self._path:
                pts.append(
                    self._world_to_screen(
                        x_m=float(x_m),
                        y_m=float(y_m),
                        origin_x=float(canvas_x),
                        origin_y=float(canvas_y),
                        w_px=canvas_w,
                        h_px=canvas_h,
                    )
                )

        if len(pts) >= 2:
            col_path = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.90)
            try:
                draw.add_polyline(pts, col_path, False, 2.0)
            except Exception:
                for i in range(1, len(pts)):
                    x0_, y0_ = pts[i - 1]
                    x1_, y1_ = pts[i]
                    dl_add_line(draw, x0_, y0_, x1_, y1_, col_path, 2.0)

        if pts:
            lx, ly = pts[-1]
            col_drone = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)
            try:
                draw.add_circle_filled(float(lx), float(ly), 4.0, col_drone, 16)
            except TypeError:
                try:
                    draw.add_circle_filled((float(lx), float(ly)), 4.0, col_drone, 16)
                except Exception:
                    pass
            except Exception:
                pass

        # Reserve the canvas region in the layout.
        imgui.dummy(canvas_w, canvas_h)
        imgui.end()


class AppUI:
    def __init__(
        self,
        osd_view,
        map_widget,
        video_settings: "VideoSettings",
        control_view: "ControlView",
        layout_editor: OsdLayoutEditor,
    ):
        self.mode = MODE_OSD
        self.osd_view = osd_view
        self.local_map = LocalMapView()
        self.map_widget = map_widget
        self.video_settings = video_settings
        self.control_view = control_view
        self.layout_editor = layout_editor
        self.pid_tune = PidTuneWindow()
        self.wizard = WizardFooter()
        self.dets_overlay = DetectionsOverlay()

        # OSD "start button" (bottom-right logo) + tracking mode menu.
        # Default on startup: tracking OFF (RGB view).
        self.track_mode: int = TRACK_MODE_OFF
        self._tracking_enable_last_desired: Optional[Tuple[int, int, int]] = None  # (hole, plane, track2d)
        self._tracking_enable_burst_remaining: int = 0
        self._tracking_enable_last_send_t: float = 0.0

        self._start_menu_target_open: bool = False
        self._start_menu_anim: float = 0.0  # 0..1
        self._start_menu_anim_from: float = 0.0
        self._start_menu_anim_to: float = 0.0
        self._start_menu_anim_start_t: float = 0.0
        self._start_menu_anim_dur_s: float = 0.14
        self._start_menu_btn_rect: Optional[Tuple[float, float, float, float]] = None  # x,y,w,h (fb coords)
        self._start_menu_menu_rect: Optional[Tuple[float, float, float, float]] = None  # x,y,w,h (fb coords)
        self._osd_ui_capture_mouse: bool = False
        self._osd_ui_click_consumed: bool = False

        # Mouse reporting (OSD main video only)
        self._mouse_last_win: Optional[Tuple[float, float]] = None
        self._mouse_last_move_send_t: float = 0.0
        self._track2d_drag_active: bool = False
        self._track2d_drag_start_fb: Optional[Tuple[float, float]] = None
        self._track2d_drag_cur_fb: Optional[Tuple[float, float]] = None

    def _maybe_send_tracking_enables(self, *, send_command: Callable[[dict], bool]) -> None:
        """
        Aligns with server-side stream switching:
          - Off:      hole=0, plane=0, track2d=0 (RGB stream)
          - 3D hole:  hole=1, plane=0, track2d=0 (IR stream)
          - 3D plane: hole=0, plane=1, track2d=0 (IR stream)
          - 2D ROI:   hole=0, plane=0, track2d=1 (v4l stream)

        Best-effort UDP: the drone server does not ACK these. To make mode switching robust on Wi-Fi,
        send a short burst after mode changes. (No continuous keepalive to avoid control-loop spam.)
        """
        desired_hole = 1 if int(self.track_mode) == TRACK_MODE_HOLE_3D else 0
        desired_plane = 1 if int(self.track_mode) == TRACK_MODE_PLANE_3D else 0
        desired_t2d = 1 if int(self.track_mode) == TRACK_MODE_ROI_2D else 0
        # Enforce mutual exclusion (defensive; the UI is single-select).
        if desired_t2d:
            desired_hole = 0
            desired_plane = 0
        if desired_hole:
            desired_plane = 0
        if desired_plane:
            desired_hole = 0

        desired = (int(desired_hole), int(desired_plane), int(desired_t2d))
        if self._tracking_enable_last_desired is None or tuple(self._tracking_enable_last_desired) != tuple(desired):
            self._tracking_enable_last_desired = tuple(desired)
            # Single-shot mode change update (no keepalive).
            self._tracking_enable_burst_remaining = 1
            self._tracking_enable_last_send_t = 0.0

        now = time.monotonic()
        # Burst after changes only (no keepalive).
        if int(self._tracking_enable_burst_remaining) <= 0:
            return
        min_period_s = 0.20

        if (now - float(self._tracking_enable_last_send_t)) < float(min_period_s):
            return
        self._tracking_enable_last_send_t = float(now)

        def _send_en(t: str, en: int) -> bool:
            try:
                return bool(send_command({"type": str(t), "enable": int(en)}))
            except Exception:
                return False

        ok_hole = ok_plane = ok_t2d = False
        # Send the "active" enable first to avoid transient OFF state during switching.
        if desired_t2d:
            ok_t2d = _send_en("track2d_enable", int(desired_t2d))
            ok_hole = _send_en("hole_enable", int(desired_hole))
            ok_plane = _send_en("plane_enable", int(desired_plane))
        elif desired_plane:
            ok_plane = _send_en("plane_enable", int(desired_plane))
            ok_hole = _send_en("hole_enable", int(desired_hole))
            ok_t2d = _send_en("track2d_enable", int(desired_t2d))
        elif desired_hole:
            ok_hole = _send_en("hole_enable", int(desired_hole))
            ok_plane = _send_en("plane_enable", int(desired_plane))
            ok_t2d = _send_en("track2d_enable", int(desired_t2d))
        else:
            ok_hole = _send_en("hole_enable", int(desired_hole))
            ok_plane = _send_en("plane_enable", int(desired_plane))
            ok_t2d = _send_en("track2d_enable", int(desired_t2d))

        # Best-effort UDP: the server does not ACK these, so the only thing we can safely guarantee
        # client-side is to avoid spamming. Decrement after an attempt even if the send_command() return
        # value is falsy (some call sites may not return a bool).
        if int(self._tracking_enable_burst_remaining) > 0:
            self._tracking_enable_burst_remaining = int(max(0, int(self._tracking_enable_burst_remaining) - 1))

    def _osd_start_button_and_menu(
        self,
        *,
        window,
        fb_w: int,
        fb_h: int,
        send_command: Callable[[dict], bool],
    ) -> None:
        """
        Treat the bottom-right logo as a "start" button:
        - Hover highlight
        - Click toggles an animated menu above it
        - Menu is a single-selection list (off/hole/plane/2D ROI)
        """
        # Defaults each frame
        self._osd_ui_capture_mouse = False
        self._osd_ui_click_consumed = False

        # Button geometry: match osd.py's bottom-right logo card.
        lw, lh = 160.0, 60.0
        margin = 20.0
        pad = 6.0
        def _px(v: float) -> float:
            return float(int(round(float(v))))

        btn_x_cur = _px(float(fb_w) - lw - margin - pad)
        btn_y_cur = _px(float(fb_h) - lh - margin - pad)
        btn_w_cur = _px(lw + (pad * 2.0))
        btn_h_cur = _px(lh + (pad * 2.0))

        # Mouse position in framebuffer coords
        mx_fb = my_fb = None
        try:
            mx_win, my_win = glfw.get_cursor_pos(window)
            ww_win, wh_win = glfw.get_window_size(window)
            if ww_win > 0 and wh_win > 0:
                mx_fb = float(mx_win) * (float(fb_w) / float(ww_win))
                my_fb = float(my_win) * (float(fb_h) / float(wh_win))
        except Exception:
            mx_fb = my_fb = None

        def _in_rect(x: float, y: float, rx: float, ry: float, rw: float, rh: float) -> bool:
            return (rx <= x <= rx + rw) and (ry <= y <= ry + rh)

        hovered_btn = False
        if mx_fb is not None and my_fb is not None:
            # Use latched rect while the menu is open/animating to avoid hover flicker if the
            # underlying framebuffer size jitters by a pixel (some driver/OS combos do).
            btn_rect_for_hit = self._start_menu_btn_rect
            if (not btn_rect_for_hit) or (not bool(self._start_menu_target_open) and float(self._start_menu_anim) <= 1e-3):
                btn_rect_for_hit = (btn_x_cur, btn_y_cur, btn_w_cur, btn_h_cur)
            bx, by, bw, bh = btn_rect_for_hit
            hovered_btn = _in_rect(float(mx_fb), float(my_fb), float(bx), float(by), float(bw), float(bh))

        # Click handling
        left_click = False
        try:
            left_click = bool(imgui.is_mouse_clicked(0))
        except Exception:
            left_click = False

        # Menu layout
        labels = [
            (TRACK_MODE_OFF, TRACK_MODE_LABELS[TRACK_MODE_OFF], True),
            (TRACK_MODE_HOLE_3D, TRACK_MODE_LABELS[TRACK_MODE_HOLE_3D], True),
            (TRACK_MODE_PLANE_3D, TRACK_MODE_LABELS[TRACK_MODE_PLANE_3D], True),
            (TRACK_MODE_ROI_2D, TRACK_MODE_LABELS[TRACK_MODE_ROI_2D], True),
        ]

        # Menu size based on text
        pad_x = 14.0
        pad_y = 10.0
        # Keep row height fixed to avoid subtle font-metric jitter causing menu position wobble.
        row_h = 46.0

        try:
            max_tw = 0.0
            for _mid, lbl, _en in labels:
                tw, _th = imgui.calc_text_size(f"[ ] {lbl}")
                max_tw = max(max_tw, float(tw))
            menu_w = max(btn_w_cur, max_tw + pad_x * 2.0 + 10.0)
        except Exception:
            menu_w = max(btn_w_cur, 320.0)

        full_h = (pad_y * 2.0) + (row_h * float(len(labels)))
        # Current (unlatched) menu geometry anchored above the button.
        gap = 10.0
        menu_h_cur = float(full_h)
        menu_x_cur = _px(float(btn_x_cur + btn_w_cur - menu_w))
        menu_bottom_cur = float(btn_y_cur) - gap
        menu_y_final_cur = _px(menu_bottom_cur - menu_h_cur)
        menu_rect_cur = (float(menu_x_cur), float(menu_y_final_cur), float(menu_w), float(menu_h_cur))

        now = time.monotonic()

        def _begin_menu_anim(open_menu: bool) -> None:
            """Start open/close animation once and latch the layout rects."""
            open_menu = bool(open_menu)
            if bool(open_menu) == bool(self._start_menu_target_open):
                return

            self._start_menu_target_open = bool(open_menu)
            self._start_menu_anim_from = float(self._start_menu_anim)
            self._start_menu_anim_to = 1.0 if bool(open_menu) else 0.0
            self._start_menu_anim_start_t = float(now)

            # Latch geometry so the menu doesn't "jump" if fb dims fluctuate.
            if self._start_menu_btn_rect is None:
                self._start_menu_btn_rect = (float(btn_x_cur), float(btn_y_cur), float(btn_w_cur), float(btn_h_cur))
            if self._start_menu_menu_rect is None:
                self._start_menu_menu_rect = menu_rect_cur
            if bool(open_menu):
                self._start_menu_btn_rect = (float(btn_x_cur), float(btn_y_cur), float(btn_w_cur), float(btn_h_cur))
                self._start_menu_menu_rect = menu_rect_cur

        def _update_menu_anim() -> None:
            dur = max(1e-3, float(self._start_menu_anim_dur_s))
            t = (float(now) - float(self._start_menu_anim_start_t)) / dur
            if t >= 1.0:
                self._start_menu_anim = float(self._start_menu_anim_to)
                # Clear latched rects once fully closed.
                if float(self._start_menu_anim) <= 0.0 and (not bool(self._start_menu_target_open)):
                    self._start_menu_btn_rect = None
                    self._start_menu_menu_rect = None
                return
            if t <= 0.0:
                self._start_menu_anim = float(self._start_menu_anim_from)
                return
            # Smoothstep easing.
            tt = float(t)
            tt = tt * tt * (3.0 - 2.0 * tt)
            self._start_menu_anim = float(self._start_menu_anim_from) + (float(self._start_menu_anim_to) - float(self._start_menu_anim_from)) * tt

        # Toggle menu when clicking the button (edge-triggered by ImGui).
        if left_click and hovered_btn:
            _begin_menu_anim(not bool(self._start_menu_target_open))
            self._osd_ui_click_consumed = True

        # If we are already open/closing and we somehow don't have latched rects, seed them.
        if (bool(self._start_menu_target_open) or float(self._start_menu_anim) > 1e-3) and (self._start_menu_menu_rect is None):
            self._start_menu_btn_rect = (float(btn_x_cur), float(btn_y_cur), float(btn_w_cur), float(btn_h_cur))
            self._start_menu_menu_rect = menu_rect_cur

        _update_menu_anim()

        anim = float(self._start_menu_anim)
        if anim <= 1e-3 and (not bool(self._start_menu_target_open)):
            # Still allow hover highlight and capture on the button.
            self._osd_ui_capture_mouse = bool(hovered_btn)
            if bool(hovered_btn):
                dl = imgui.get_foreground_draw_list()
                col = (1.0, 1.0, 1.0, 0.18)
                dl_add_rect_filled(dl, btn_x_cur, btn_y_cur, btn_x_cur + btn_w_cur, btn_y_cur + btn_h_cur, col, 10.0)
                try:
                    dl.add_rect(
                        float(btn_x_cur),
                        float(btn_y_cur),
                        float(btn_x_cur + btn_w_cur),
                        float(btn_y_cur + btn_h_cur),
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.65),
                        10.0,
                        0,
                        2.0,
                    )
                except Exception:
                    pass
            # Still send enable toggles even when the menu is closed.
            self._maybe_send_tracking_enables(send_command=send_command)
            return

        # Use latched geometry while open/animating.
        btn_x, btn_y, btn_w, btn_h = (btn_x_cur, btn_y_cur, btn_w_cur, btn_h_cur)
        if self._start_menu_btn_rect is not None:
            btn_x, btn_y, btn_w, btn_h = self._start_menu_btn_rect
        menu_x, menu_y_final, menu_w2, menu_h = menu_rect_cur
        if self._start_menu_menu_rect is not None:
            menu_x, menu_y_final, menu_w2, menu_h = self._start_menu_menu_rect

        # Slider animation: reveal from the bottom edge.
        menu_bottom = float(menu_y_final) + float(menu_h)
        vis_h = float(menu_h) * float(anim)
        menu_y = float(menu_bottom) - float(vis_h)

        hovered_menu = False
        if mx_fb is not None and my_fb is not None:
            hovered_menu = _in_rect(float(mx_fb), float(my_fb), float(menu_x), float(menu_y), float(menu_w2), float(vis_h))

        if left_click and hovered_menu:
            self._osd_ui_click_consumed = True

        # Close on outside click (and swallow the click so it doesn't become a mouse_click to the server).
        if left_click and bool(self._start_menu_target_open) and (not hovered_btn) and (not hovered_menu):
            _begin_menu_anim(False)
            self._osd_ui_click_consumed = True

        # Capture mouse while hovering button/menu, while menu is open, or while it is animating.
        self._osd_ui_capture_mouse = bool(hovered_btn) or bool(hovered_menu) or bool(self._start_menu_target_open) or (anim > 1e-3)

        # Draw menu (foreground so it sits above overlays)
        dl = imgui.get_foreground_draw_list()
        bg = (0.10, 0.10, 0.10, 0.92 * anim)
        border_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.35 * anim)
        dl_add_rect_filled(dl, menu_x, menu_y, menu_x + float(menu_w2), menu_y + vis_h, bg, 10.0)
        try:
            dl.add_rect(float(menu_x), float(menu_y), float(menu_x + float(menu_w2)), float(menu_y + vis_h), border_col, 10.0, 0, 2.0)
        except Exception:
            pass

        # Rows
        for idx, (mode_id, lbl, enabled) in enumerate(labels):
            row_y = menu_y + pad_y + (float(idx) * row_h)
            if (row_y + row_h) > (menu_y + vis_h - pad_y + 1.0):
                break

            selected = (int(self.track_mode) == int(mode_id))
            hovered_row = False
            if mx_fb is not None and my_fb is not None:
                hovered_row = _in_rect(float(mx_fb), float(my_fb), menu_x, row_y, float(menu_w2), row_h)

            if selected:
                dl_add_rect_filled(dl, menu_x + 2, row_y, menu_x + float(menu_w2) - 2, row_y + row_h, (0.20, 0.55, 1.00, 0.28 * anim), 8.0)
            elif hovered_row and enabled:
                dl_add_rect_filled(dl, menu_x + 2, row_y, menu_x + float(menu_w2) - 2, row_y + row_h, (1.0, 1.0, 1.0, 0.08 * anim), 8.0)

            prefix = "[x]" if selected else "[ ]"
            text = f"{prefix} {lbl}"
            col = (1.0, 1.0, 1.0, 0.92 * anim) if enabled else (1.0, 1.0, 1.0, 0.35 * anim)
            try:
                tw, th = imgui.calc_text_size(text)
                ty = row_y + (row_h - float(th)) * 0.5
            except Exception:
                ty = row_y + 10.0
            dl_add_text(dl, menu_x + pad_x, ty, col, text, font_scale=1.0)

            # Selection click
            if left_click and hovered_row and enabled:
                if int(self.track_mode) != int(mode_id):
                    self.track_mode = int(mode_id)
                    # Force immediate re-send on user action.
                    self._tracking_enable_last_desired = None
                    self._tracking_enable_burst_remaining = 0
                    self._tracking_enable_last_send_t = 0.0
                    # Cancel any in-progress ROI drag when switching modes.
                    self._track2d_drag_active = False
                    self._track2d_drag_start_fb = None
                    self._track2d_drag_cur_fb = None
                _begin_menu_anim(False)
                self._osd_ui_click_consumed = True

        # Button hover highlight (draw last)
        if bool(hovered_btn) or bool(self._start_menu_target_open):
            hl_a = 0.22 if hovered_btn else 0.14
            dl_add_rect_filled(dl, btn_x, btn_y, btn_x + btn_w, btn_y + btn_h, (1.0, 1.0, 1.0, hl_a), 10.0)
            try:
                dl.add_rect(
                    float(btn_x),
                    float(btn_y),
                    float(btn_x + btn_w),
                    float(btn_y + btn_h),
                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.75),
                    10.0,
                    0,
                    2.0,
                )
            except Exception:
                pass

        # Apply selection -> send enable toggles (best-effort, throttled).
        self._maybe_send_tracking_enables(send_command=send_command)

    def draw(
        self,
        *,
        window,
        win_w: int,
        win_h: int,
        fb_w: int,
        fb_h: int,
        video_tex: Optional[int],
        video_widgets: Optional[List[Dict[str, Any]]] = None,
        tracking_enabled: bool = True,
        request_video_widget_swap: Optional[Callable[[int], None]] = None,
        telem,
        send_command: Callable[[dict], bool],
    ) -> None:
        video_widgets = video_widgets or []
        widget_clicked = False
        if self.mode != MODE_OSD:
            self._start_menu_target_open = False
            self._start_menu_anim = 0.0
            self._start_menu_anim_from = 0.0
            self._start_menu_anim_to = 0.0
            self._start_menu_anim_start_t = 0.0
            self._start_menu_btn_rect = None
            self._start_menu_menu_rect = None
            self._osd_ui_capture_mouse = False
            self._osd_ui_click_consumed = False
            self._track2d_drag_active = False
            self._track2d_drag_start_fb = None
            self._track2d_drag_cur_fb = None

        # Modes
        if self.mode == MODE_OSD:
            # ROI select is intentionally disabled for the 3D acquisition flow
            # (RMB is reserved for clear/cancel).

            if video_tex:
                draw_fullscreen(int(video_tex), fb_w, fb_h)

            # PiP video widgets (background layer, above fullscreen video).
            for w in video_widgets:
                if not w.get("enabled", True):
                    continue
                draw_video_widget(
                    tex=w.get("tex"),
                    x=float(w.get("x", 0)),
                    y=float(w.get("y", 0)),
                    w=float(w.get("w", 0)),
                    h=float(w.get("h", 0)),
                    label=str(w.get("label", "")),
                )

            if tracking_enabled:
                self.dets_overlay.draw(fb_w, fb_h, telem)
            self.osd_view.draw(telem, (fb_w, fb_h))
            self._osd_start_button_and_menu(window=window, fb_w=fb_w, fb_h=fb_h, send_command=send_command)

        elif self.mode == MODE_LOCAL_MAP:
            self.local_map.draw(win_w=win_w, win_h=win_h, fb_w=fb_w, fb_h=fb_h, telem=telem)

        elif self.mode == MODE_MAP:
            if video_tex:
                draw_fullscreen(int(video_tex), fb_w, fb_h)
            self.map_widget.draw_window(win_w, win_h, telem)

        elif self.mode == MODE_VIDEO:
            if video_tex:
                draw_fullscreen(int(video_tex), fb_w, fb_h)
            self.video_settings.draw_window(win_w, win_h)

        elif self.mode == MODE_CONTROL:
            if video_tex:
                draw_fullscreen(int(video_tex), fb_w, fb_h)
            self.control_view.draw_window(win_w, win_h)

        elif self.mode == MODE_CONFIG:
            if video_tex:
                draw_fullscreen(int(video_tex), fb_w, fb_h)
            self.layout_editor.draw_window(win_w, win_h)

        elif self.mode == MODE_TRACK:
            self.pid_tune.draw(win_w, win_h, telem, video_tex, fb_w, fb_h)

        # wizard footer LAST so it's clickable
        self.mode = self.wizard.draw(win_w, win_h, self.mode)

        # Widget click-to-swap takes priority over click-to-track.
        if (self.mode == MODE_OSD) and (not self._osd_ui_click_consumed) and video_widgets and imgui.is_mouse_clicked(0) and request_video_widget_swap:
            io = imgui.get_io()
            if not io.want_capture_mouse:
                mx_win, my_win = glfw.get_cursor_pos(window)

                # Ignore clicks in the wizard footer.
                wx, wy, ww, wh = self.wizard.rect
                if not ((wx <= mx_win <= wx + ww) and (wy <= my_win <= wy + wh)):
                    ww_win, wh_win = glfw.get_window_size(window)
                    if ww_win > 0 and wh_win > 0:
                        mx_fb = mx_win * (fb_w / float(ww_win))
                        my_fb = my_win * (fb_h / float(wh_win))
                        for w in video_widgets:
                            if not w.get("enabled", True):
                                continue
                            x = float(w.get("x", 0))
                            y = float(w.get("y", 0))
                            ww_ = float(w.get("w", 0))
                            wh_ = float(w.get("h", 0))
                            if (x <= mx_fb <= x + ww_) and (y <= my_fb <= y + wh_):
                                try:
                                    request_video_widget_swap(int(w.get("idx", 0)))
                                finally:
                                    widget_clicked = True
                                break

        # Click-to-track (legacy detect/ROI selection) is intentionally disabled for the 3D acquisition flow.

        # OSD interactions -> server (OSD main video only; not over PiP widgets or UI).
        # - 3D hole/plane mode: mouse_move / mouse_click (only when enabled)
        # - 2D ROI mode: track2d_rect (on LMB drag release) + track2d_cancel (RMB)
        if self.mode != MODE_OSD:
            self._mouse_last_win = None
            self._mouse_last_move_send_t = 0.0
            self._track2d_drag_active = False
            self._track2d_drag_start_fb = None
            self._track2d_drag_cur_fb = None
        else:
            # Use camera.active (from JSON "camera") to gate overlay/input behavior.
            cam_active = getattr(telem, "camera_active", None)
            try:
                cam_active = str(cam_active).lower().strip() if cam_active is not None else None
            except Exception:
                cam_active = None
            if not cam_active:
                # Fallback before we receive a "camera" message.
                if int(self.track_mode) == TRACK_MODE_ROI_2D:
                    cam_active = "v4l"
                elif int(self.track_mode) in (TRACK_MODE_HOLE_3D, TRACK_MODE_PLANE_3D):
                    cam_active = "ir"
                else:
                    cam_active = "rgb"

            hole_enabled = int(self.track_mode) == TRACK_MODE_HOLE_3D
            plane_enabled = int(self.track_mode) == TRACK_MODE_PLANE_3D
            t2d_enabled = int(self.track_mode) == TRACK_MODE_ROI_2D
            v4l_active = (str(cam_active).lower() == "v4l")

            allow_track3d_input = (bool(hole_enabled) or bool(plane_enabled)) and (not bool(v4l_active))
            allow_track2d_input = bool(t2d_enabled) and bool(v4l_active)

            if not allow_track2d_input:
                self._track2d_drag_active = False
                self._track2d_drag_start_fb = None
                self._track2d_drag_cur_fb = None

            try:
                mx_win, my_win = glfw.get_cursor_pos(window)
            except Exception:
                mx_win, my_win = (None, None)

            if (mx_win is not None) and (my_win is not None):
                now = time.monotonic()
                cur_win = (float(mx_win), float(my_win))
                last_win = self._mouse_last_win
                moved = False
                if last_win is None:
                    self._mouse_last_win = cur_win
                else:
                    dx = cur_win[0] - float(last_win[0])
                    dy = cur_win[1] - float(last_win[1])
                    if (dx * dx + dy * dy) >= (2.0 * 2.0):  # >=2px movement
                        self._mouse_last_win = cur_win
                        moved = True

                # Mouse in framebuffer coords (used for overlay hit-tests + normalization).
                mx_fb = my_fb = None
                if win_w > 0 and win_h > 0 and fb_w > 0 and fb_h > 0:
                    mx_fb = cur_win[0] * (fb_w / float(win_w))
                    my_fb = cur_win[1] * (fb_h / float(win_h))

                io = imgui.get_io()
                can_send_base = True
                if win_w <= 0 or win_h <= 0:
                    can_send_base = False
                elif not (0.0 <= cur_win[0] <= float(win_w) and 0.0 <= cur_win[1] <= float(win_h)):
                    can_send_base = False
                try:
                    if io.want_capture_mouse:
                        can_send_base = False
                except Exception:
                    pass

                # Ignore wizard footer region (window coords).
                if can_send_base:
                    try:
                        wx, wy, ww, wh = self.wizard.rect
                        if (wx <= cur_win[0] <= wx + ww) and (wy <= cur_win[1] <= wy + wh):
                            can_send_base = False
                    except Exception:
                        pass

                # Ignore PiP widgets (framebuffer coords).
                if can_send_base and mx_fb is not None and my_fb is not None:
                    for w in (video_widgets or []):
                        if not w.get("enabled", True):
                            continue
                        x = float(w.get("x", 0))
                        y = float(w.get("y", 0))
                        ww_ = float(w.get("w", 0))
                        wh_ = float(w.get("h", 0))
                        if (x <= mx_fb <= x + ww_) and (y <= my_fb <= y + wh_):
                            can_send_base = False
                            break

                # Ignore UI overlays (start button/menu).
                if can_send_base and bool(getattr(self, "_osd_ui_capture_mouse", False)):
                    can_send_base = False

                def _clamp01(v: float) -> float:
                    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else float(v))

                # ----- 2D ROI tracker interactions -----
                if allow_track2d_input:
                    # Cancel (RMB).
                    if can_send_base:
                        try:
                            if imgui.is_mouse_clicked(1):
                                send_command({"type": "track2d_cancel"})
                        except Exception:
                            pass

                    # Start ROI drag (LMB).
                    if can_send_base and (not self._track2d_drag_active) and mx_fb is not None and my_fb is not None:
                        try:
                            if imgui.is_mouse_clicked(0):
                                self._track2d_drag_active = True
                                self._track2d_drag_start_fb = (float(mx_fb), float(my_fb))
                                self._track2d_drag_cur_fb = (float(mx_fb), float(my_fb))
                        except Exception:
                            pass

                    # Update drag while held.
                    if self._track2d_drag_active and mx_fb is not None and my_fb is not None:
                        try:
                            if imgui.is_mouse_down(0):
                                self._track2d_drag_cur_fb = (float(mx_fb), float(my_fb))
                        except Exception:
                            pass

                    # Draw drag rect (feedback).
                    if self._track2d_drag_active and self._track2d_drag_start_fb and self._track2d_drag_cur_fb:
                        x0, y0 = self._track2d_drag_start_fb
                        x1, y1 = self._track2d_drag_cur_fb
                        rx0, rx1 = (min(float(x0), float(x1)), max(float(x0), float(x1)))
                        ry0, ry1 = (min(float(y0), float(y1)), max(float(y0), float(y1)))
                        dl = imgui.get_foreground_draw_list()
                        col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.85)
                        dl_add_line(dl, rx0, ry0, rx1, ry0, col, 2.0)
                        dl_add_line(dl, rx1, ry0, rx1, ry1, col, 2.0)
                        dl_add_line(dl, rx1, ry1, rx0, ry1, col, 2.0)
                        dl_add_line(dl, rx0, ry1, rx0, ry0, col, 2.0)

                    # Finish ROI on release.
                    if self._track2d_drag_active:
                        try:
                            released = bool(imgui.is_mouse_released(0))
                        except Exception:
                            released = False
                        if released:
                            start = self._track2d_drag_start_fb
                            end = self._track2d_drag_cur_fb or ((float(mx_fb) if mx_fb is not None else 0.0), (float(my_fb) if my_fb is not None else 0.0))
                            self._track2d_drag_active = False
                            self._track2d_drag_start_fb = None
                            self._track2d_drag_cur_fb = None

                            if can_send_base and start and end and fb_w > 0 and fb_h > 0:
                                x0, y0 = start
                                x1, y1 = end
                                if (abs(float(x1) - float(x0)) >= 6.0) and (abs(float(y1) - float(y0)) >= 6.0):
                                    x0n = _clamp01(min(float(x0), float(x1)) / float(fb_w))
                                    y0n = _clamp01(min(float(y0), float(y1)) / float(fb_h))
                                    x1n = _clamp01(max(float(x0), float(x1)) / float(fb_w))
                                    y1n = _clamp01(max(float(y0), float(y1)) / float(fb_h))
                                    try:
                                        send_command(
                                            {
                                                "type": "track2d_rect",
                                                "surface": "osd_main",
                                                "x0_norm": float(x0n),
                                                "y0_norm": float(y0n),
                                                "x1_norm": float(x1n),
                                                "y1_norm": float(y1n),
                                            }
                                        )
                                    except Exception:
                                        pass

                # ----- 3D hole acquisition interactions -----
                if allow_track3d_input:
                    can_send_move = can_send_base
                    try:
                        if any(io.mouse_down):
                            can_send_move = False
                    except Exception:
                        pass

                    # Mouse-move: send while moving (rate-limited).
                    if can_send_move and moved and mx_fb is not None and my_fb is not None:
                        try:
                            x_norm = _clamp01(0.0 if fb_w <= 0 else (float(mx_fb) / float(fb_w)))
                            y_norm = _clamp01(0.0 if fb_h <= 0 else (float(my_fb) / float(fb_h)))
                            if (now - float(self._mouse_last_move_send_t)) >= 0.03:
                                send_command(
                                    {
                                        "type": "mouse_move",
                                        "ts": int(time.time() * 1000),
                                        "surface": "osd_main",
                                        "x_norm": float(x_norm),
                                        "y_norm": float(y_norm),
                                    }
                                )
                                self._mouse_last_move_send_t = now
                        except Exception:
                            pass

                    # Mouse-click: left/right click events (commit/cancel).
                    if can_send_base and mx_fb is not None and my_fb is not None:
                        btn = None
                        try:
                            if imgui.is_mouse_clicked(0):
                                btn = "left"
                            elif imgui.is_mouse_clicked(1):
                                btn = "right"
                        except Exception:
                            btn = None
                        if btn:
                            try:
                                x_norm = _clamp01(0.0 if fb_w <= 0 else (float(mx_fb) / float(fb_w)))
                                y_norm = _clamp01(0.0 if fb_h <= 0 else (float(my_fb) / float(fb_h)))
                                send_command(
                                    {
                                        "type": "mouse_click",
                                        "ts": int(time.time() * 1000),
                                        "surface": "osd_main",
                                        "button": str(btn),
                                        "x_norm": float(x_norm),
                                        "y_norm": float(y_norm),
                                    }
                                )
                            except Exception:
                                pass

class VideoSettings:
    def __init__(self, app_cfg_getter, app_cfg_setter, apply_callback):
        self.get_app = app_cfg_getter
        self.set_app = app_cfg_setter
        self.apply_cb = apply_callback
        self.edit = None  # live editing copy

    def _ensure_stream_defaults(self, v: Dict[str, Any]) -> None:
        v.setdefault("source", "videotest")   # videotest | udp | rtsp
        v.setdefault("framerate", "60/1")
        v.setdefault("udp_port", 5600)
        v.setdefault("codec", "h265")
        v.setdefault("rtsp_url", "")
        v.setdefault("rtsp_autodetect", True)
        v.setdefault("rtsp_codec", "auto")  # auto | h264 | h265 (non-HTTP RTSP only)
        v.setdefault("width", 1280)
        v.setdefault("height", 720)
        v.setdefault("format", "NV12")
        v.setdefault("flip_h", False)
        v.setdefault("flip_v", False)

    def _ensure_widgets_defaults(self, video_cfg: Dict[str, Any]) -> None:
        widgets = video_cfg.get("widgets", [])
        if not isinstance(widgets, list):
            widgets = []

        while len(widgets) < 3:
            widgets.append({})
        widgets = widgets[:3]

        for idx in range(3):
            w = widgets[idx]
            if not isinstance(w, dict):
                w = {}
                widgets[idx] = w

            w.setdefault("enabled", False)
            w.setdefault("x", 1540)
            w.setdefault("y", 40 + idx * 232)
            sp = w.get("size_px")
            if not (isinstance(sp, (list, tuple)) and len(sp) == 2):
                w["size_px"] = [360, 202]
            else:
                try:
                    w["size_px"] = [int(sp[0]), int(sp[1])]
                except Exception:
                    w["size_px"] = [360, 202]

            st = w.get("stream", {})
            if not isinstance(st, dict):
                st = {}
                w["stream"] = st
            self._ensure_stream_defaults(st)

        video_cfg["widgets"] = widgets

    def _ensure_edit(self):
        if self.edit is None:
            app = self.get_app()
            src = app.get("video", {})
            if not isinstance(src, dict):
                src = {}

            # Deep copy to keep nested widget edits isolated until Apply.
            self.edit = json.loads(json.dumps(src))
            self._ensure_stream_defaults(self.edit)
            self._ensure_widgets_defaults(self.edit)

    def _draw_stream_editor(self, *, v: Dict[str, Any]) -> None:
        # --- Source select (sticks immediately in self.edit) ---
        srcs = ["videotest", "udp", "rtsp"]
        try:
            cur_src = srcs.index(v.get("source", "videotest"))
        except ValueError:
            cur_src = 0
            v["source"] = "videotest"

        changed_src, cur_src = imgui.combo("Source", cur_src, srcs)
        if changed_src:
            v["source"] = srcs[cur_src]

        imgui.separator()

        # -------------------------------
        #  Source-specific configuration
        # -------------------------------
        src = v.get("source", "videotest")

        if src == "videotest":
            fps_items = ["30/1", "60/1", "90/1"]
            try:
                curfps = fps_items.index(v.get("framerate", "60/1"))
            except ValueError:
                curfps = 1
                v["framerate"] = "60/1"
            changed_fps, curfps = imgui.combo("FPS", curfps, fps_items)
            if changed_fps:
                v["framerate"] = fps_items[curfps]

            imgui.separator()
            imgui.text_disabled("Info: UDP and RTSP settings are not applicable for the test source.")

        elif src == "udp":
            imgui.text("UDP Receiver")
            imgui.separator()

            port = int(v.get("udp_port", 5600))
            changed_port, port = imgui.input_int("UDP Port", port)
            if changed_port:
                v["udp_port"] = max(1, min(65535, port))

            codec_opts = ["h265", "h264"]
            try:
                cc = codec_opts.index(v.get("codec", "h265"))
            except ValueError:
                cc = 0
                v["codec"] = "h265"
            changed_c, cc = imgui.combo("RTP Codec", cc, codec_opts)
            if changed_c:
                v["codec"] = codec_opts[cc]

            imgui.separator()
            imgui.text_disabled("Info: Frame rate & resolution come from the sender. RTSP URL is not applicable.")

        else:  # rtsp
            changed_url, url = imgui.input_text("RTSP URL", v.get("rtsp_url", ""), 512)
            if changed_url:
                v["rtsp_url"] = url

            is_http = str(v.get("rtsp_url", "")).lower().startswith("http://") or str(v.get("rtsp_url", "")).lower().startswith("https://")
            if is_http:
                imgui.text_disabled("HTTP URL detected: treated as MJPEG (codec options not used).")
            else:
                codecs = ["auto", "h264", "h265"]
                try:
                    cidx = codecs.index(str(v.get("rtsp_codec", "auto")).lower())
                except Exception:
                    cidx = 0
                    v["rtsp_codec"] = "auto"
                changed_c, cidx = imgui.combo("RTSP Codec", cidx, codecs)
                if changed_c:
                    v["rtsp_codec"] = codecs[cidx]
                imgui.text_disabled("Tip: use 'auto' unless the stream fails to decode.")

            autodetect = bool(v.get("rtsp_autodetect", True))
            changed_auto, autodetect = imgui.checkbox("Autodetect stream parameters (recommended)", autodetect)
            if changed_auto:
                v["rtsp_autodetect"] = autodetect

            if not autodetect:
                imgui.spacing()
                imgui.text("Manual parameters (used if autodetect fails):")

                fps_items = ["30/1", "60/1", "90/1"]
                try:
                    curfps = fps_items.index(v.get("framerate", "60/1"))
                except ValueError:
                    curfps = 1
                    v["framerate"] = "60/1"
                changed_fps, curfps = imgui.combo("FPS", curfps, fps_items)
                if changed_fps:
                    v["framerate"] = fps_items[curfps]

                w = int(v.get("width", 1280))
                changed_w, w = imgui.input_int("Width", w)
                if changed_w:
                    v["width"] = max(160, min(7680, w))

                h = int(v.get("height", 720))
                changed_h, h = imgui.input_int("Height", h)
                if changed_h:
                    v["height"] = max(120, min(4320, h))

                fmt = v.get("format", "NV12")
                changed_fmt, fmt = imgui.input_text("Format", fmt, 32)
                if changed_fmt:
                    v["format"] = fmt

            imgui.separator()
            imgui.text_disabled("Info: UDP settings are not applicable in RTSP mode.")

        imgui.separator()
        flip_h = bool(v.get("flip_h", False))
        flip_v = bool(v.get("flip_v", False))
        ch_fh, flip_h = imgui.checkbox("Flip Horizontal", flip_h)
        if ch_fh:
            v["flip_h"] = flip_h
        ch_fv, flip_v = imgui.checkbox("Flip Vertical", flip_v)
        if ch_fv:
            v["flip_v"] = flip_v

    def draw_window(self, win_w, win_h):
        self._ensure_edit()
        v = self.edit  # shorthand

        imgui.set_next_window_position(40, 40)
        imgui.set_next_window_size(win_w - 80, win_h - 160)
        imgui.begin("Video Settings", True, imgui.WINDOW_NO_COLLAPSE)

        imgui.text("Main Video Overlay")
        imgui.push_id("main_video")
        self._draw_stream_editor(v=v)
        imgui.pop_id()

        imgui.separator()
        imgui.text("Video Stream Widgets (OSD mode only)")
        imgui.text_disabled("Click a widget in OSD to swap it with fullscreen (display-only).")

        widgets = v.get("widgets", [])
        if not isinstance(widgets, list):
            widgets = []
            v["widgets"] = widgets
        self._ensure_widgets_defaults(v)
        widgets = v["widgets"]

        for idx in range(3):
            wcfg = widgets[idx]
            imgui.push_id(f"widget_{idx}")
            opened = imgui.collapsing_header(f"Widget {idx + 1}", flags=imgui.TREE_NODE_DEFAULT_OPEN)
            if isinstance(opened, tuple):
                opened = opened[0]
            if opened:
                en = bool(wcfg.get("enabled", False))
                changed, en = imgui.checkbox("Enabled", en)
                if changed:
                    wcfg["enabled"] = bool(en)

                x = int(wcfg.get("x", 0))
                y = int(wcfg.get("y", 0))
                changed_x, x = imgui.input_int("x", x)
                if changed_x:
                    wcfg["x"] = max(0, int(x))
                changed_y, y = imgui.input_int("y", y)
                if changed_y:
                    wcfg["y"] = max(0, int(y))

                sp = wcfg.get("size_px", [360, 202])
                try:
                    ww_ = int(sp[0])
                    hh_ = int(sp[1])
                except Exception:
                    ww_, hh_ = 360, 202

                changed_w, ww_ = imgui.input_int("width", int(ww_))
                changed_h, hh_ = imgui.input_int("height", int(hh_))
                if changed_w or changed_h:
                    ww_ = max(32, min(4096, int(ww_)))
                    hh_ = max(32, min(4096, int(hh_)))
                    wcfg["size_px"] = [ww_, hh_]

                imgui.separator()
                imgui.text("Widget Stream")
                st = wcfg.get("stream", {})
                if not isinstance(st, dict):
                    st = {}
                    wcfg["stream"] = st
                self._ensure_stream_defaults(st)
                self._draw_stream_editor(v=st)

            imgui.pop_id()

        imgui.separator()

        if imgui.button("Apply & Rebuild", 220, 0):
            # write back to app config and rebuild
            app = self.get_app()
            app["video"] = json.loads(json.dumps(v))
            self.set_app(app)
            self.apply_cb()
            # keep edit so user sees current state; or reset, your call

        imgui.same_line()
        if imgui.button("Cancel", 140, 0):
            # discard edits and reload from saved app config
            app = self.get_app()
            src = app.get("video", {})
            self.edit = json.loads(json.dumps(src)) if isinstance(src, dict) else {}
            self._ensure_stream_defaults(self.edit)
            self._ensure_widgets_defaults(self.edit)

        imgui.end()

# --- replace your existing ControlView in gui.py with this one ---

import socket, json as _json, time
import imgui

class ControlView:
    """
    Sends JSON control messages over UDP to the drone CTRL_PORT.

    Expects app_config like:
    {
      "control": {
        "ctrl_ip": "127.0.0.1",
        "ctrl_port": 6020,
        "dest_ip": "",
        "dest_port": 5600,
        "telemetry_ip": "",
        "telemetry_port": 6021,
        "mavlink_ip": "",
        "mavlink_port": 14550,
        "bitrate_kbps": 12000,
        "rc_mode": "CBR",
        "gop": 60,
        "fps": 60,
        "width": 1280,
        "height": 720,
        "cam_res_id": 3,
        "stream_method": "UDP_UNICAST"
      }
    }
    """

    def __init__(self, app_config_getter):
        self.get_cfg = app_config_getter
        self._st = None  # working copy for edits in the UI

    # ---------- internals ----------
    def _ensure_state(self):
        if self._st is not None:
            return
        app = self.get_cfg()
        ctrl = dict(app.get("control", {}))
        # sensible defaults
        ctrl.setdefault("ctrl_ip", "127.0.0.1")
        ctrl.setdefault("ctrl_port", 6020)
        ctrl.setdefault("dest_ip", "")
        ctrl.setdefault("dest_port", 5600)
        ctrl.setdefault("telemetry_ip", "")
        ctrl.setdefault("telemetry_port", 6021)
        ctrl.setdefault("mavlink_ip", "")
        ctrl.setdefault("mavlink_port", 14550)
        ctrl.setdefault("bitrate_kbps", 12000)
        ctrl.setdefault("rc_mode", "CBR")
        ctrl.setdefault("gop", 60)
        ctrl.setdefault("fps", 60)
        ctrl.setdefault("width", 1280)
        ctrl.setdefault("height", 720)
        ctrl.setdefault("cam_res_id", 3)  # 3=HD720, 2=HD1080 (matches your server comments)
        ctrl.setdefault("stream_method", "UDP_UNICAST")
        ctrl.setdefault("hole_detector_enabled", True)
        self._st = ctrl

    def _udp_send(self, ip, port, obj):
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.sendto(_json.dumps(obj).encode("utf-8"), (ip, int(port)))
            s.close()
        except Exception as e:
            print("[CTRL][GS] UDP send failed:", e)

    def _save_to_disk(self):
        try:
            from utils import save_json
            app = self.get_cfg()
            app["control"] = dict(self._st)
            save_json("app_config.json", app)
            print("[CTRL][GS] control config saved to app_config.json")
        except Exception as e:
            print("[CTRL][GS] save failed:", e)

    # ---------- UI ----------
    def draw_window(self, win_w, win_h):
        self._ensure_state()
        st = self._st

        imgui.set_next_window_position(40, 40)
        imgui.set_next_window_size(win_w - 80, win_h - 160)
        imgui.begin("Control (UDP JSON -> Drone CTRL_PORT)", True,
                    imgui.WINDOW_NO_COLLAPSE)

        # Connection
        imgui.text("Control Target")
        changed, ip = imgui.input_text("CTRL IP", st["ctrl_ip"], 128)
        if changed: st["ctrl_ip"] = ip
        changed, prt = imgui.input_int("CTRL Port", int(st["ctrl_port"]))
        if changed: st["ctrl_port"] = max(1, min(65535, prt))

        if imgui.button("Ping", 120, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"ping": int(time.time())})
        imgui.same_line()
        if imgui.button("Force IDR", 140, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"idr": True})

        changed, en = imgui.checkbox("Hole detector (3D acquire)", bool(st.get("hole_detector_enabled", True)))
        if changed:
            st["hole_detector_enabled"] = bool(en)
            self._udp_send(
                st["ctrl_ip"],
                st["ctrl_port"],
                {
                    "type": "hole_enable",
                    "ts": int(time.time() * 1000),
                    "enable": int(1 if bool(en) else 0),
                },
            )

        imgui.separator()

        # Destination (UDP video)
        imgui.text("Video Destination (UDP mode)")
        if imgui.button("Video to Sender", 200, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"video_to_sender": True})

        changed, dip = imgui.input_text("dest_ip", st.get("dest_ip",""), 128)
        if changed: st["dest_ip"] = dip
        changed, dpt = imgui.input_int("dest_port", int(st.get("dest_port",5600)))
        if changed: st["dest_port"] = max(1, min(65535, dpt))

        if imgui.button("Apply Destination", 200, 0):
            msg = {}
            if st.get("dest_ip"):  msg["dest_ip"] = st["dest_ip"]
            if st.get("dest_port"): msg["dest_port"] = int(st["dest_port"])
            if msg:
                self._udp_send(st["ctrl_ip"], st["ctrl_port"], msg)

        imgui.same_line()
        # Stream method (kept simple for now; your server supports UDP_UNICAST in UDP mode)
        methods = ["UDP_UNICAST"]
        cur_idx = max(0, methods.index(st.get("stream_method","UDP_UNICAST"))
                         if st.get("stream_method","UDP_UNICAST") in methods else 0)
        changed, cur_idx = imgui.combo("stream_method", cur_idx, methods)
        if changed: st["stream_method"] = methods[cur_idx]
        imgui.same_line()
        if imgui.button("Set Stream Method", 200, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"stream_method": st["stream_method"]})

        imgui.separator()

        # Runtime encoder controls
        imgui.text("Encoder (Runtime)")
        changed, kbps = imgui.input_int("bitrate_kbps", int(st["bitrate_kbps"]))
        if changed: st["bitrate_kbps"] = max(1, min(9000000, int(kbps)))
        if imgui.button("Set Bitrate", 150, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"bitrate_kbps": int(st["bitrate_kbps"])})

        modes = ["CBR", "VBR"]
        m_idx = 0 if str(st.get("rc_mode","CBR")).upper()=="CBR" else 1
        changed, m_idx = imgui.combo("rc_mode", m_idx, modes)
        if changed: st["rc_mode"] = modes[m_idx]
        imgui.same_line()
        if imgui.button("Set RC Mode", 150, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"rc_mode": st["rc_mode"]})

        changed, g = imgui.input_int("gop (frames)", int(st["gop"]))
        if changed: st["gop"] = max(1, min(10000, int(g)))
        imgui.same_line()
        if imgui.button("Set GOP", 120, 0):
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], {"gop": int(st["gop"])})

        imgui.separator()

        # Structural controls (rebuild)
        imgui.text("Structural (Rebuild on drone)")
        fps_options = [30, 60, 90]
        try: fps_idx = fps_options.index(int(st["fps"]))
        except Exception: fps_idx = 1
        changed, fps_idx = imgui.combo("fps", fps_idx, [str(x) for x in fps_options])
        if changed: st["fps"] = fps_options[fps_idx]

        changed, w = imgui.input_int("width", int(st["width"]))
        if changed: st["width"] = max(160, min(3840, int(w)))
        changed, h = imgui.input_int("height", int(st["height"]))
        if changed: st["height"] = max(120, min(2160, int(h)))

        # cam_res_id hint: 3=HD720; 2=HD1080
        res_labels = ["HD1080 (2)", "HD720 (3)"]
        id_map = [2, 3]
        try:
            ridx = id_map.index(int(st["cam_res_id"]))
        except Exception:
            ridx = 1  # default to 3/HD720
        changed, ridx = imgui.combo("cam_res_id", ridx, res_labels)
        if changed: st["cam_res_id"] = id_map[ridx]

        if imgui.button("Apply Structural", 200, 0):
            msg = {
                "fps": int(st["fps"]),
                "width": int(st["width"]),
                "height": int(st["height"]),
                "cam_res_id": int(st["cam_res_id"])
            }
            self._udp_send(st["ctrl_ip"], st["ctrl_port"], msg)

        imgui.separator()

        # Telemetry / MAVLink routing (UDP mode)
        imgui.text("Telemetry & MAVLink (UDP out from drone)")
        changed, tip = imgui.input_text("telemetry_ip", st.get("telemetry_ip",""), 128)
        if changed: st["telemetry_ip"] = tip
        changed, tpt = imgui.input_int("telemetry_port", int(st.get("telemetry_port",6021)))
        if changed: st["telemetry_port"] = max(1, min(65535, int(tpt)))

        changed, mip = imgui.input_text("mavlink_ip", st.get("mavlink_ip",""), 128)
        if changed: st["mavlink_ip"] = mip
        changed, mpt = imgui.input_int("mavlink_port", int(st.get("mavlink_port",14550)))
        if changed: st["mavlink_port"] = max(1, min(65535, int(mpt)))

        if imgui.button("Apply Telemetry", 180, 0):
            msg = {}
            if st.get("telemetry_ip"):   msg["telemetry_ip"] = st["telemetry_ip"]
            if st.get("telemetry_port"): msg["telemetry_port"] = int(st["telemetry_port"])
            if msg: self._udp_send(st["ctrl_ip"], st["ctrl_port"], msg)

        imgui.same_line()
        if imgui.button("Apply MAVLink", 160, 0):
            msg = {}
            if st.get("mavlink_ip"):
                # New key (preferred) + legacy key for backwards compatibility.
                msg["mav_mirror_ip"] = st["mavlink_ip"]
                msg["mavlink_ip"] = st["mavlink_ip"]
            if st.get("mavlink_port"):
                msg["mavlink_port"] = int(st["mavlink_port"])
            if msg: self._udp_send(st["ctrl_ip"], st["ctrl_port"], msg)

        imgui.separator()

        if imgui.button("Save Control Config", 220, 0):
            self._save_to_disk()

        imgui.end()
