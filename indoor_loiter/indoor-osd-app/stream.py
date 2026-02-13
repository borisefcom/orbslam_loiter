#!/usr/bin/env python3
import threading, time, os
from typing import Dict, Any, Optional, Tuple

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GObject

# Ensure GStreamer is initialized once in this process
Gst.init(None)


def _flip_chain(cfg: Dict[str, Any]) -> str:
    """Return a videoflip stage chain based on config flags."""
    vcfg = dict(cfg or {})
    flip_h = bool(vcfg.get("flip_h", False))
    flip_v = bool(vcfg.get("flip_v", False))
    parts = []
    if flip_h:
        parts.append("videoflip method=horizontal-flip")
    if flip_v:
        parts.append("videoflip method=vertical-flip")
    return (" ! ".join(parts)) if parts else ""


class VideoWorker(threading.Thread):
    """
    Pulls frames via GStreamer and exposes the latest RGB frame for OpenGL upload.

    Config schema (app_config.json -> "video"):
      {
        "source": "videotest" | "udp" | "rtsp",
        "framerate": "60/1",              # for videotest only
        "udp_port": 5600,                 # for udp
        "codec": "h265" | "h264",         # for udp
        "rtsp_url": "rtsp://..." | "http://... (MJPEG)",
        "rtsp_codec": "auto" | "h264" | "h265",  # for non-HTTP RTSP only (auto uses uridecodebin)
        "flip_h": false,
        "flip_v": false,

        # Optional output scaling (used for PiP widgets).
        # When set, the pipeline scales to exactly this size before appsink.
        "out_width": 0,
        "out_height": 0
      }
    """
    def __init__(self, cfg: Dict[str, Any]):
        super().__init__(daemon=True)
        self.cfg = dict(cfg or {})
        self._latest: Optional[Tuple[int, int, bytes]] = None
        self._lock = threading.Lock()
        self._running = True

        self._pipe = None
        self._appsink = None
        self._frame_count = 0
        self._fps_last_time = time.monotonic()
        self._live_fps = 0.0
    # ---------------------- Public API ----------------------
    def stop(self):
        self._running = False

    def get_latest(self):
        with self._lock:
            return self._latest

    def get_live_fps(self) -> float:
        return float(self._live_fps)

    # ---------------------- Internals -----------------------
    def _on_new_sample(self, sink):
        sample = sink.emit("pull-sample")
        if not sample:
            return Gst.FlowReturn.OK

        buf = sample.get_buffer()
        caps = sample.get_caps()
        s = caps.get_structure(0)
        w = int(s.get_value("width"))
        h = int(s.get_value("height"))

        # --- Copy pixels ---
        ok, info = buf.map(Gst.MapFlags.READ)
        if ok:
            with self._lock:
                self._latest = (w, h, bytes(info.data))
            buf.unmap(info)

        # --- FPS counter ---
        self._frame_count += 1
        now = time.monotonic()
        if now - self._fps_last_time >= 1.0:
            self._live_fps = float(self._frame_count) / (now - self._fps_last_time)
            self._frame_count = 0
            self._fps_last_time = now

        return Gst.FlowReturn.OK

    def _appsink_tail(self) -> str:
        # Always drop old buffers; never sync to clock
        return "appsink name=appsink emit-signals=true max-buffers=1 drop=true sync=false"

    def _lowlat_queue(self) -> str:
        # Queue that only holds one buffer and drops the rest
        return "queue max-size-buffers=1 max-size-time=0 max-size-bytes=0 leaky=downstream"

    def _common_color_chain(self) -> str:
        # Convert to RGB for the OpenGL texture upload path, optionally scaling first.
        caps = "video/x-raw,format=RGB"
        try:
            ow = int(self.cfg.get("out_width", 0) or 0)
        except Exception:
            ow = 0
        try:
            oh = int(self.cfg.get("out_height", 0) or 0)
        except Exception:
            oh = 0
        if ow > 0 and oh > 0:
            caps = f"{caps},width={ow},height={oh}"

        return f"videoconvert ! videoscale ! {caps} ! {self._lowlat_queue()}"

    def _build_pipeline_str(self) -> str:
        src = str(self.cfg.get("source", "videotest")).lower().strip()
        flip = _flip_chain(self.cfg)

        # 1) Pure local test — no seeking, live timestamps, leaky queues
        if src == "videotest":
            fr = str(self.cfg.get("framerate", "60/1"))
            return (
                "videotestsrc is-live=true pattern=snow do-timestamp=true ! "
                f"video/x-raw,framerate={fr} ! {self._lowlat_queue()} ! "
                f"{self._common_color_chain()} "
                f"{('! ' + flip) if flip else ''} ! "
                f"{self._appsink_tail()}"
            )

        # 2) RTP/UDP — minimize latency via jitterbuffer & leaky queues
        if src == "udp":
            port = int(self.cfg.get("udp_port", 5600))
            codec = str(self.cfg.get("codec", "h265")).lower()
            enc = "H265" if codec == "h265" else "H264"
            depay = "rtph265depay" if codec == "h265" else "rtph264depay"
            parse = "h265parse" if codec == "h265" else "h264parse"
            dec   = "avdec_h265"  if codec == "h265" else "avdec_h264"

            return (
                f"udpsrc port={port} caps=\"application/x-rtp, media=video, encoding-name={enc}, payload=96, clock-rate=90000\" ! "
                "rtpjitterbuffer latency=0 mode=0 max-dropout-time=0 max-misorder-time=0 do-lost=true ! "
                f"{depay} ! {parse} ! {dec} ! "
                f"{self._lowlat_queue()} ! "
                f"{self._common_color_chain()} "
                f"{('! ' + flip) if flip else ''} ! "
                f"{self._appsink_tail()}"
            )

        # 3) RTSP or HTTP MJPEG handled under the same 'rtsp' selector.
        #    If URL starts with http/https: treat it as MJPEG (multipart), never use uridecodebin.
        if src == "rtsp":
            url = str(self.cfg.get("rtsp_url", "")).strip()
            if url.lower().startswith("http://") or url.lower().startswith("https://"):
                # --- HTTP MJPEG (no seeking; no Range headers) ---
                # souphttpsrc -> multipartdemux -> jpegdec -> (leaky) -> convert -> appsink
                # We explicitly avoid uridecodebin here so it won't probe/seek.
                ua = "OSDApp/1.0"
                return (
                    f"souphttpsrc is-live=true do-timestamp=true keep-alive=true user-agent=\"{ua}\" location=\"{url}\" ! "
                    "multipartdemux ! image/jpeg ! "
                    "jpegdec ! "
                    f"{self._lowlat_queue()} ! "
                    f"{self._common_color_chain()} "
                    f"{('! ' + flip) if flip else ''} ! "
                    f"{self._appsink_tail()}"
                )
            else:
                # --- RTSP (keep simple for now) ---
                # uridecodebin typically does not seek for RTSP; we still keep sink unsynced and leaky.
                # If we need even lower latency we can switch to explicit rtspsrc + pad-added to set latency≈0–50.
                rtsp_codec = str(self.cfg.get("rtsp_codec", "auto")).lower().strip()
                if rtsp_codec in ("h264", "h265"):
                    depay = "rtph265depay" if rtsp_codec == "h265" else "rtph264depay"
                    parse = "h265parse" if rtsp_codec == "h265" else "h264parse"
                    dec   = "avdec_h265"  if rtsp_codec == "h265" else "avdec_h264"
                    return (
                        f"rtspsrc location=\"{url}\" latency=0 ! "
                        f"{depay} ! {parse} ! {dec} ! "
                        f"{self._lowlat_queue()} ! "
                        f"{self._common_color_chain()} "
                        f"{('! ' + flip) if flip else ''} ! "
                        f"{self._appsink_tail()}"
                    )

                return (
                    f"uridecodebin uri=\"{url}\" use-buffering=false ! "
                    f"{self._lowlat_queue()} ! "
                    f"{self._common_color_chain()} "
                    f"{('! ' + flip) if flip else ''} ! "
                    f"{self._appsink_tail()}"
                )

        raise ValueError(f"Unknown video source: {src}")

    def _build_and_play(self):
        # Build via parse_launch and hook appsink signal
        pipe_str = self._build_pipeline_str()
        self._pipe = Gst.parse_launch(pipe_str)
        self._appsink = self._pipe.get_by_name("appsink")
        if not self._appsink:
            raise RuntimeError("appsink not found in pipeline")

        # Always deliver newest frame only
        self._appsink.connect("new-sample", self._on_new_sample)

        # Go PLAYING
        self._pipe.set_state(Gst.State.PLAYING)

    def run(self):
        try:
            self._build_and_play()
            bus = self._pipe.get_bus()
            while self._running:
                msg = bus.timed_pop_filtered(100 * Gst.MSECOND,
                                             Gst.MessageType.ERROR | Gst.MessageType.EOS | Gst.MessageType.STATE_CHANGED)
                if not msg:
                    continue
                t = msg.type
                if t == Gst.MessageType.ERROR:
                    err, dbg = msg.parse_error()
                    print(f"[VIDEO][ERR] {err} {dbg or ''}".rstrip())
                    # keep running: try to continue; if source is fatal, pipeline will EOS
                elif t == Gst.MessageType.EOS:
                    print("[VIDEO] EOS"); break
                # STATE_CHANGED messages are noisy; we ignore unless debugging
        except Exception as e:
            print("[VIDEO] worker error:", e)
        finally:
            try:
                if self._pipe:
                    self._pipe.set_state(Gst.State.NULL)
            except Exception:
                pass
