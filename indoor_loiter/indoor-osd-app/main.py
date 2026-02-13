#!/usr/bin/env python3
import os, time, json, socket
import gi
gi.require_version('Gst','1.0')
from gi.repository import Gst

import glfw
import OpenGL.GL as gl
import imgui
from imgui.integrations.glfw import GlfwRenderer
from typing import Optional, Dict, Any, List, Tuple

from utils import (
    load_json, save_json,
    create_empty_texture, update_texture_rgb,
    CursorManager,
)
from stream import VideoWorker
from mavlink import TelemetryWorker, JsonTelemetryWorker
from osd import OSDView
from gui import AppUI, OsdLayoutEditor, VideoSettings, ControlView, MODE_OSD
from maps import MapConfig, MapWidget


TARGET_FPS = 60


def _cfg_ip_or_none(v: Any) -> Optional[str]:
    if v is None:
        return None
    try:
        s = str(v).strip()
    except Exception:
        return None
    if s == "":
        return None
    if s.lower() == "auto":
        return None
    return s


def _bind_ip(v: Any) -> str:
    # Empty/"auto" means bind all interfaces.
    ip = _cfg_ip_or_none(v)
    return ip if ip is not None else "0.0.0.0"


def _dest_ip(v: Any) -> Optional[str]:
    ip = _cfg_ip_or_none(v)
    if ip in (None, "", "0.0.0.0"):
        return None
    return ip


def _local_ip_for_peer(peer_ip: str, peer_port: int) -> Optional[str]:
    """
    Return the local IPv4 address the OS would use to reach (peer_ip, peer_port).
    Uses a UDP "connect" (no packets sent) so it works without a handshake.
    """
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((str(peer_ip), int(peer_port)))
        local_ip = s.getsockname()[0]
        return str(local_ip)
    except Exception:
        return None
    finally:
        try:
            s.close()
        except Exception:
            pass


def _ensure_video_stream_defaults(cfg: Dict[str, Any]) -> Dict[str, Any]:
    if cfg is None or not isinstance(cfg, dict):
        cfg = {}
    cfg.setdefault("source", "videotest")  # videotest | udp | rtsp
    cfg.setdefault("framerate", "60/1")
    cfg.setdefault("udp_port", 5600)
    cfg.setdefault("codec", "h265")
    cfg.setdefault("rtsp_url", "")
    cfg.setdefault("rtsp_autodetect", True)
    cfg.setdefault("rtsp_codec", "auto")  # auto | h264 | h265 (non-HTTP RTSP only)
    cfg.setdefault("width", 1280)
    cfg.setdefault("height", 720)
    cfg.setdefault("format", "NV12")
    cfg.setdefault("flip_h", False)
    cfg.setdefault("flip_v", False)
    return cfg


def _ensure_video_widgets(video_cfg: Dict[str, Any], count: int = 3) -> Dict[str, Any]:
    if video_cfg is None or not isinstance(video_cfg, dict):
        video_cfg = {}

    _ensure_video_stream_defaults(video_cfg)

    widgets = video_cfg.get("widgets", [])
    if not isinstance(widgets, list):
        widgets = []

    # Normalize list length.
    while len(widgets) < count:
        widgets.append({})
    widgets = widgets[:count]

    # Fill defaults per widget.
    for idx in range(count):
        w = widgets[idx]
        if not isinstance(w, dict):
            w = {}
            widgets[idx] = w
        w.setdefault("enabled", False)
        w.setdefault("x", 1540)
        w.setdefault("y", 40 + idx * 232)
        sp = w.get("size_px")
        if not (isinstance(sp, (list, tuple)) and len(sp) == 2):
            sp = [360, 202]
            w["size_px"] = sp
        else:
            try:
                sp[0] = int(sp[0])
                sp[1] = int(sp[1])
            except Exception:
                w["size_px"] = [360, 202]

        stream = w.get("stream", {})
        if not isinstance(stream, dict):
            stream = {}
            w["stream"] = stream
        _ensure_video_stream_defaults(stream)

    video_cfg["widgets"] = widgets
    return video_cfg


class _VideoSlot:
    def __init__(self):
        self.worker: Optional[VideoWorker] = None
        self.cfg_sig: Optional[str] = None
        self.tex: Optional[int] = None
        self.size: Tuple[int, int] = (0, 0)

    def stop_worker(self) -> None:
        try:
            if self.worker:
                self.worker.stop()
        except Exception:
            pass
        try:
            if self.worker:
                self.worker.join(timeout=1.0)
        except Exception:
            pass
        self.worker = None
        self.cfg_sig = None

    def delete_texture(self) -> None:
        try:
            if self.tex:
                gl.glDeleteTextures(int(self.tex))
        except Exception:
            pass
        self.tex = None
        self.size = (0, 0)

    def stop(self, *, delete_texture: bool = True) -> None:
        self.stop_worker()
        if delete_texture:
            self.delete_texture()


class App:
    def __init__(self):
        try:
            Gst.init(None)
        except Exception:
            pass

        base = os.path.dirname(__file__)
        self.osd_path = os.path.join(base, "osd_config.json")
        self.app_path = os.path.join(base, "app_config.json")
        self.map_path = os.path.join(base, "map_config.json")

        self.osd_cfg = load_json(self.osd_path)
        self.app_cfg = load_json(self.app_path)
        _ensure_video_widgets(self.app_cfg.get("video", {}))

        # Map config & widget
        self.map_cfg = MapConfig.load()
        self.map_widget = MapWidget(self.map_cfg)

        c = self.app_cfg.get("control", {})
        if not isinstance(c, dict):
            c = {}

        ctrl_ip = _cfg_ip_or_none(c.get("ctrl_ip"))
        try:
            ctrl_port = int(c.get("ctrl_port", 0) or 0)
        except Exception:
            ctrl_port = 0

        # Best-effort: detect our local IP toward the control server.
        self.local_ip: Optional[str] = None
        if ctrl_ip and ctrl_port > 0:
            self.local_ip = _local_ip_for_peer(ctrl_ip, ctrl_port)

        # MAVLink + JSON telemetry
        endpoint = self.osd_cfg.get("mavlink",{}).get("endpoint")
        if not endpoint:
            mip = _bind_ip(c.get("mavlink_ip"))
            mpt = int(c.get("mavlink_port", 14550) or 14550)
            endpoint = f"udpin:{mip}:{mpt}"
        self.telem = TelemetryWorker(endpoint)
        self.telem.start()

        # JSON telemetry listener
        snap = self.telem.get()
        jt_ip   = _bind_ip(c.get("telemetry_ip"))
        jt_port = int(c.get("telemetry_port", 6021) or 6021)
        dbg = self.app_cfg.get("debug", {}) if isinstance(self.app_cfg.get("debug", {}), dict) else {}
        json_log_cfg = dbg.get("json_logging", None)
        self.json_telem = JsonTelemetryWorker(jt_ip, jt_port, snap, log_enabled=False, log_cfg=json_log_cfg)

        # Optional control destination (for send_command)
        if ctrl_ip is not None and ctrl_port > 0:
            try:
                self.json_telem.set_control_dest(str(ctrl_ip), int(ctrl_port))
            except Exception as e:
                print("[JSON] failed to set control dest:", e)

        self.json_telem.start()

        # On startup, ask the server to send telemetry/video/mavlink mirror to this GUI.
        # If the relevant IP fields are empty/"auto"/0.0.0.0, we use the detected local IP.
        try:
            self._send_startup_routing(c, ctrl_ip=ctrl_ip, ctrl_port=ctrl_port, local_ip=self.local_ip)
        except Exception as e:
            print("[CTRL] startup routing failed:", e)

        # Video slots:
        # - slot 0: fullscreen (unscaled)
        # - slots 1..3: PiP widgets (downscaled to widget size)
        self._video_perm: List[int] = [0, 1, 2, 3]  # slot->stream index (0=main, 1..3=widget streams)
        self._video_slots: List[_VideoSlot] = [_VideoSlot(), _VideoSlot(), _VideoSlot(), _VideoSlot()]
        self._video_mode_osd: bool = True  # AppUI starts in OSD
        self._pending_swap_widget_idx: Optional[int] = None
        self._ensure_video_slots(mode_osd=True, force=True)

        # OSD
        self.osd_view = OSDView(lambda: self.osd_cfg, lambda: self._video_slots[0].worker.get_live_fps() if self._video_slots[0].worker else 0.0)

        # GUI panes
        self.video_settings = VideoSettings(lambda: self.app_cfg, lambda c: self._set_app_cfg(c), self._apply_video_settings)
        self.control_view = ControlView(lambda: self.app_cfg)

        # UI (mode switching + windows)
        self.layout_editor = OsdLayoutEditor(lambda: self.osd_cfg, self._set_osd_cfg, self.osd_path)
        self.ui = AppUI(self.osd_view, self.map_widget, self.video_settings, self.control_view, self.layout_editor)

        self.video_tex: Optional[int] = None
        self.video_size: Tuple[int,int] = (0,0)
        self.cursor_mgr = None

    def _send_startup_routing(self, c: Dict[str, Any], *, ctrl_ip: Optional[str], ctrl_port: int, local_ip: Optional[str]) -> None:
        if not ctrl_ip or ctrl_port <= 0:
            return

        def send_retry(msg: Dict[str, Any], attempts: int = 40, delay_s: float = 0.05) -> None:
            for _ in range(attempts):
                if self.json_telem.send_command(msg):
                    return
                time.sleep(delay_s)

        # Desired destination IP for drone->GUI links.
        auto_ip = local_ip

        telemetry_ip = _dest_ip(c.get("telemetry_ip")) or auto_ip
        try:
            telemetry_port = int(c.get("telemetry_port", 6021) or 6021)
        except Exception:
            telemetry_port = 6021

        mav_mirror_ip = _dest_ip(c.get("mav_mirror_ip")) or _dest_ip(c.get("mavlink_ip")) or auto_ip

        video_cfg = self.app_cfg.get("video", {})
        if not isinstance(video_cfg, dict):
            video_cfg = {}
        try:
            video_listen_port = int(video_cfg.get("udp_port", 5600) or 5600)
        except Exception:
            video_listen_port = 5600
        dest_port = video_listen_port
        dest_ip = _dest_ip(c.get("dest_ip")) or auto_ip

        if telemetry_ip:
            send_retry({"telemetry_ip": telemetry_ip, "telemetry_port": int(telemetry_port)})
        if mav_mirror_ip:
            send_retry({"mav_mirror_ip": mav_mirror_ip, "mavlink_ip": mav_mirror_ip})
        if dest_ip:
            send_retry({"dest_ip": dest_ip, "dest_port": int(dest_port)})

    def _set_app_cfg(self, cfg):
        self.app_cfg = cfg
        try:
            save_json(self.app_path, cfg)
            print("[APP] app_config saved")
        except Exception as e:
            print("[APP] save app_config failed:", e)

    def _set_osd_cfg(self, cfg):
        self.osd_cfg = cfg

    def _apply_video_settings(self):
        # Reset runtime swaps and rebuild.
        self._video_perm = [0, 1, 2, 3]
        self._pending_swap_widget_idx = None
        self._ensure_video_slots(mode_osd=(self.ui.mode == MODE_OSD), force=True)

    def _video_stream_cfg(self, stream_idx: int) -> Dict[str, Any]:
        vcfg = self.app_cfg.get("video", {})
        _ensure_video_widgets(vcfg)

        if stream_idx == 0:
            base = dict(vcfg)
            base.pop("widgets", None)
            return base

        widx = stream_idx - 1
        widgets = vcfg.get("widgets", [])
        if not (0 <= widx < len(widgets)):
            return {}
        w = widgets[widx] if isinstance(widgets[widx], dict) else {}
        stream = w.get("stream", {})
        return dict(stream) if isinstance(stream, dict) else {}

    def _video_slot_target_scale(self, slot_idx: int) -> Optional[Tuple[int, int]]:
        if slot_idx == 0:
            return None
        vcfg = self.app_cfg.get("video", {})
        _ensure_video_widgets(vcfg)
        w = vcfg.get("widgets", [])[slot_idx - 1]
        sp = w.get("size_px", [0, 0])
        try:
            ow = int(sp[0])
            oh = int(sp[1])
        except Exception:
            return None
        if ow > 0 and oh > 0:
            return (ow, oh)
        return None

    def _video_widget_enabled(self, widget_idx: int) -> bool:
        vcfg = self.app_cfg.get("video", {})
        _ensure_video_widgets(vcfg)
        w = vcfg.get("widgets", [])[widget_idx]
        try:
            return bool(w.get("enabled", False))
        except Exception:
            return False

    def _video_build_worker_cfg(self, *, stream_idx: int, scale_to: Optional[Tuple[int, int]]) -> Dict[str, Any]:
        cfg = self._video_stream_cfg(stream_idx)
        _ensure_video_stream_defaults(cfg)
        cfg.pop("widgets", None)

        if scale_to is not None:
            cfg["out_width"] = int(scale_to[0])
            cfg["out_height"] = int(scale_to[1])
        else:
            cfg.pop("out_width", None)
            cfg.pop("out_height", None)

        return cfg

    def _video_slot_sig(self, cfg: Dict[str, Any]) -> str:
        try:
            return json.dumps(cfg, sort_keys=True)
        except Exception:
            return str(cfg)

    def _ensure_video_slot_running(self, slot_idx: int, *, cfg: Dict[str, Any], force: bool) -> None:
        slot = self._video_slots[slot_idx]
        sig = self._video_slot_sig(cfg)
        if (not force) and slot.worker and slot.cfg_sig == sig:
            return

        slot.stop()
        slot.cfg_sig = sig
        slot.worker = VideoWorker(cfg)
        slot.worker.start()

    def _ensure_video_slots(self, *, mode_osd: bool, force: bool = False) -> None:
        # Always keep fullscreen slot running:
        # - In OSD: follow runtime mapping (swaps)
        # - Else: force main stream (no swaps outside OSD)
        full_stream_idx = self._video_perm[0] if mode_osd else 0
        self._ensure_video_slot_running(
            0,
            cfg=self._video_build_worker_cfg(stream_idx=full_stream_idx, scale_to=None),
            force=force,
        )

        # PiP widget slots (only in OSD mode, only if enabled)
        for widx in range(3):
            slot_idx = 1 + widx
            if mode_osd and self._video_widget_enabled(widx):
                stream_idx = self._video_perm[slot_idx]
                scale_to = self._video_slot_target_scale(slot_idx)
                self._ensure_video_slot_running(
                    slot_idx,
                    cfg=self._video_build_worker_cfg(stream_idx=stream_idx, scale_to=scale_to),
                    force=force,
                )
            else:
                self._video_slots[slot_idx].stop()

    def _video_set_mode(self, *, mode_osd: bool) -> None:
        if mode_osd == self._video_mode_osd:
            return

        # Leaving OSD: reset swaps so other modes always use the main overlay stream.
        if not mode_osd:
            self._video_perm = [0, 1, 2, 3]
            self._pending_swap_widget_idx = None

        self._video_mode_osd = mode_osd
        # Don't force-restart pipelines on mode changes; only restart if the stream config actually changes.
        self._ensure_video_slots(mode_osd=mode_osd, force=False)

    def request_video_widget_swap(self, widget_idx: int) -> None:
        self._pending_swap_widget_idx = int(widget_idx)

    def _apply_pending_video_swap(self) -> None:
        if self._pending_swap_widget_idx is None:
            return
        if not self._video_mode_osd:
            self._pending_swap_widget_idx = None
            return

        widx = int(self._pending_swap_widget_idx)
        self._pending_swap_widget_idx = None
        if not (0 <= widx < 3) or (not self._video_widget_enabled(widx)):
            return

        slot_idx = 1 + widx
        self._video_perm[0], self._video_perm[slot_idx] = self._video_perm[slot_idx], self._video_perm[0]
        # Only restart slots whose stream config changed.
        self._ensure_video_slots(mode_osd=True, force=False)

    # ----------------- window / GL -----------------
    def init_window(self):
        if not glfw.init(): raise RuntimeError("glfw init failed")
        monitor = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(monitor)
        self.window = glfw.create_window(mode.size.width, mode.size.height, "ESP710 OSD App", monitor, None)
        glfw.make_context_current(self.window)
        glfw.swap_interval(0)

        imgui.create_context()
        self.impl = GlfwRenderer(self.window, attach_callbacks=True)

        io = imgui.get_io()
        io.font_global_scale = 2.0
        self.impl.refresh_font_texture()

        style = imgui.get_style()
        style.frame_padding = (14, 10)
        style.item_spacing = (16, 12)

        self.cursor_mgr = CursorManager(self.window, auto_hide=False)

    def _draw_pid_tune(self, win_w, win_h):
        """
        PID Debug window:
        - almost fullscreen (leaves space for the wizard footer)
        - stacked plots spanning width (CX, CY, Yaw, Pitch, Thrust)
        - uses new pid_debug JSON parsed into Telemetry by mavlink.JsonTelemetryWorker
        """
        import imgui
        from utils import plot_line  # local import to avoid touching global imports

        # Leave bottom space for wizard buttons (~70px footer + margin)
        bottom_reserved = 90
        left_margin, top_margin, right_margin = 10, 10, 10

        # Optional video background (unchanged)
        if getattr(self, "video_tex", None):
            self.draw_fullscreen(self.video_tex, *self.video_size)

        # Window placement & sizing
        win_x = left_margin
        win_y = top_margin
        win_wi = max(320, win_w - (left_margin + right_margin))
        win_hi = max(200, win_h - bottom_reserved - top_margin)

        imgui.set_next_window_position(win_x, win_y)
        imgui.set_next_window_size(win_wi, win_hi)
        imgui.begin(
            "PID Tune",
            False,
            imgui.WINDOW_NO_COLLAPSE | imgui.WINDOW_NO_RESIZE | imgui.WINDOW_NO_MOVE
        )

        snap = self.telem.get()

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
                    xx.append(a);
                    yy.append(b)
            return xx[-200:], yy[-200:]

        xs_cx, cxs = fv(xs, cxs)
        xs_cy, cys = fv(xs, cys)
        xs_yw, yws = fv(xs, yws)
        xs_pt, pts = fv(xs, pts)
        xs_th, ths = fv(xs, ths)

        # Header lines
        rc_txt = "—" if gate_rc is None else ("ON" if gate_rc else "OFF")
        ok_txt = "—" if gate_mode is None else ("YES" if gate_mode else "NO")
        imgui.text(f"Tracking state: {pid_state}")
        imgui.same_line()
        imgui.text(f"  RC Gate: {rc_txt}   |   Gate OK: {ok_txt} ")
        imgui.same_line()
        imgui.text(
            (f"yaw: {yaw_cmd:+.3f}" if yaw_cmd is not None else "yaw: —") + "   "
            + (f"pitch: {pitch_cmd:+.3f}" if pitch_cmd is not None else "pitch: —") + "   "
            + (f"thrust: {thrust_cmd:+.3f}" if thrust_cmd is not None else "thrust: —")
        )

        imgui.text(
            (f"cx_px: {cx_px:.1f}" if cx_px is not None else "cx_px: —") + "   "
            + (f"cy_px: {cy_px:.1f}" if cy_px is not None else "cy_px: —") + "   |   "
            + (f"ex_px: {ex_px:.1f}" if ex_px is not None else "ex_px: —") + "   "
            + (f"ey_px: {ey_px:.1f}" if ey_px is not None else "ey_px: —") + "   |   "
            + (f"nx: {nx:.3f}" if nx is not None else "nx: —") + "   "
            + (f"ny: {ny:.3f}" if ny is not None else "ny: —")
        )

        # Draw stacked plots (span the window width)
        dl = imgui.get_foreground_draw_list()
        # Use the current cursor as top-left for plots
        origin_x, origin_y = imgui.get_cursor_screen_pos()

        # Account for some inner padding
        inner_pad = 10
        plot_x = origin_x + inner_pad
        plot_w = win_wi - (inner_pad * 2) - 20
        row_h = 120
        row_gap = 49

        # Row 1: CX
        plot_y = origin_y + 22
        plot_line(dl, plot_x, plot_y, plot_w, row_h, xs_cx, cxs, title="CX (px)", fmt_val="{:+.1f}", show_minmax=True)

        # Row 2: CY
        plot_y += row_h + row_gap
        plot_line(dl, plot_x, plot_y, plot_w, row_h, xs_cy, cys, title="CY (px)", fmt_val="{:+.1f}", show_minmax=True)

        # Row 3: Yaw
        plot_y += row_h + row_gap
        plot_line(dl, plot_x, plot_y, plot_w, row_h, xs_yw, yws, title="Yaw cmd", fmt_val="{:+.3f}", show_minmax=True)

        # Row 4: Pitch
        plot_y += row_h + row_gap
        plot_line(dl, plot_x, plot_y, plot_w, row_h, xs_pt, pts, title="Pitch cmd", fmt_val="{:+.3f}", show_minmax=True)

        # Row 5: Thrust
        plot_y += row_h + row_gap
        plot_line(dl, plot_x, plot_y, plot_w, row_h, xs_th, ths, title="Thrust cmd", fmt_val="{:+.3f}",
                  show_minmax=True)

        # Advance ImGui cursor so later widgets (if any) don’t overlap drawings
        total_h = (plot_y + row_h) - origin_y
        imgui.dummy(plot_w, total_h + 6)

        imgui.end()

    def shutdown(self):
        # Stop workers first (best-effort)
        try:
            for slot in getattr(self, "_video_slots", []) or []:
                try:
                    slot.stop(delete_texture=False)
                except Exception:
                    pass
        except Exception:
            pass
        try:
            self.json_telem.stop()
        except Exception:
            pass
        try:
            self.telem.stop()
        except Exception:
            pass

        # Join threads (best-effort; keep timeouts short)
        # (Video slots already joined inside slot.stop().)
        try:
            self.json_telem.join(timeout=1.0)
        except Exception:
            pass
        try:
            self.telem.join(timeout=1.0)
        except Exception:
            pass

        # GL/ImGui cleanup requires a current context.
        try:
            if getattr(self, "window", None):
                glfw.make_context_current(self.window)
        except Exception:
            pass

        # Stop map worker + free its textures
        try:
            self.map_widget.shutdown()
        except Exception:
            pass

        # Free app-created textures
        try:
            for slot in getattr(self, "_video_slots", []) or []:
                try:
                    slot.delete_texture()
                except Exception:
                    pass
        except Exception:
            pass
        try:
            if getattr(self, "osd_view", None):
                self.osd_view.icons.shutdown()
        except Exception:
            pass

        # Renderer/window cleanup
        try:
            if getattr(self, "impl", None):
                self.impl.shutdown()
        except Exception:
            pass
        try:
            imgui.destroy_context()
        except Exception:
            pass
        try:
            if getattr(self, "window", None):
                glfw.destroy_window(self.window)
        except Exception:
            pass
        try:
            glfw.terminate()
        except Exception:
            pass

    # ----------------- main loop -----------------
    def run(self):
        try:
            self.init_window()
            last = time.time()
            frame = 1.0 / float(TARGET_FPS)

            while not glfw.window_should_close(self.window):
                glfw.poll_events()
                self.impl.process_inputs()

                # video mode transitions (widgets only in OSD mode)
                self._video_set_mode(mode_osd=(self.ui.mode == MODE_OSD))

                # video texture updates (fullscreen + enabled widgets)
                for slot in self._video_slots:
                    if not slot.worker:
                        continue
                    latest = slot.worker.get_latest()
                    if not latest:
                        continue
                    w, h, b = latest
                    if (slot.tex is None) or (slot.size != (w, h)):
                        if slot.tex:
                            try:
                                gl.glDeleteTextures(int(slot.tex))
                            except Exception:
                                pass
                        slot.tex = create_empty_texture(w, h)
                        slot.size = (w, h)
                    try:
                        update_texture_rgb(slot.tex, w, h, b)
                    except Exception as e:
                        print("[VIDEO] texture update error:", e)

                # back-compat fields used by some UI code
                self.video_tex = self._video_slots[0].tex
                self.video_size = self._video_slots[0].size

                # sizes: window (ImGui coords) and framebuffer (pixels)
                win_w, win_h = glfw.get_window_size(self.window)
                fb_w, fb_h = glfw.get_framebuffer_size(self.window)

                # OSD-only PiP widget draw data
                video_widgets = []
                tracking_enabled = True
                if self.ui.mode == MODE_OSD:
                    vcfg = self.app_cfg.get("video", {})
                    _ensure_video_widgets(vcfg)
                    widgets_cfg = vcfg.get("widgets", [])
                    for idx in range(3):
                        wcfg = widgets_cfg[idx]
                        enabled = bool(wcfg.get("enabled", False))
                        sp = wcfg.get("size_px", [0, 0])
                        try:
                            ww_ = int(sp[0])
                            hh_ = int(sp[1])
                        except Exception:
                            ww_, hh_ = 0, 0
                        video_widgets.append(
                            {
                                "idx": idx,
                                "enabled": enabled,
                                "x": float(wcfg.get("x", 0)),
                                "y": float(wcfg.get("y", 0)),
                                "w": float(ww_),
                                "h": float(hh_),
                                "tex": self._video_slots[1 + idx].tex if enabled else None,
                                "label": f"V{idx + 1}",
                            }
                        )

                    tracking_enabled = (self._video_perm[0] == 0)

                imgui.new_frame()
                self.ui.draw(
                    window=self.window,
                    win_w=win_w,
                    win_h=win_h,
                    fb_w=fb_w,
                    fb_h=fb_h,
                    video_tex=self.video_tex,
                    video_widgets=video_widgets,
                    tracking_enabled=tracking_enabled,
                    request_video_widget_swap=self.request_video_widget_swap,
                    telem=self.telem.get(),
                    send_command=self.json_telem.send_command,
                )

                # Process runtime video swaps requested by the OSD view.
                self._apply_pending_video_swap()

                # render
                gl.glViewport(0, 0, *glfw.get_framebuffer_size(self.window))
                gl.glClearColor(0, 0, 0, 1)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT)
                imgui.render()
                self.impl.render(imgui.get_draw_data())
                glfw.swap_buffers(self.window)

                # pacing
                dt = time.time() - last
                sleep_left = frame - dt
                if sleep_left > 0:
                    time.sleep(sleep_left)
                last = time.time()
        finally:
            self.shutdown()


def main():
    App().run()

if __name__=="__main__":
    main()
