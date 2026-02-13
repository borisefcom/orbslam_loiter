import os, json, math, time, threading, queue, io
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import imgui
import OpenGL.GL as gl
from PIL import Image
import requests

# --------- Small shared helpers (no GL here) ---------
def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _placeholder_tile(w: int = 256, h: int = 256) -> Image.Image:
    img = Image.new("RGBA", (w, h), (35, 35, 35, 255))
    return img

def _deg2num(lat_deg: float, lon_deg: float, zoom: int) -> Tuple[int, int]:
    lat_rad = math.radians(lat_deg)
    n = 2.0 ** zoom
    xtile = int((lon_deg + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
    return xtile, ytile

def _num2deg(xtile: int, ytile: int, zoom: int) -> Tuple[float, float]:
    """Return (lat_deg, lon_deg) for the NW corner of the given tile coordinate."""
    n = 2.0 ** int(zoom)
    lon_deg = float(xtile) / n * 360.0 - 180.0
    lat_rad = math.atan(math.sinh(math.pi * (1.0 - 2.0 * float(ytile) / n)))
    lat_deg = math.degrees(lat_rad)
    return float(lat_deg), float(lon_deg)

def _tile_url(style: str, z: int, x: int, y: int, key: str) -> str:
    # MapTiler common styles
    base = "https://api.maptiler.com/maps"
    if style == "streets":
        path = "streets"
    elif style == "satellite":
        path = "satellite"
    elif style == "outdoor":
        path = "outdoor"
    elif style == "topo":
        path = "topo"
    elif style == "toner":
        path = "toner"
    else:
        path = "streets"
    return f"{base}/{path}/256/{z}/{x}/{y}.png?key={key}"

@dataclass
class MapConfig:
    style: str = "satellite"       # streets | satellite | outdoor | topo | toner
    api_key: str = ""              # MapTiler key
    cache_dir: str = "tile_cache"  # base cache folder (per style subfolder is added)
    default_lat: float = 30.915260
    default_lon: float = 34.423723
    default_zoom: int = 17
    min_zoom: int = 0
    max_zoom: int = 20
    heading_up: bool = False
    auto_download: bool = False

    @staticmethod
    def load(path: str = "map_config.json") -> "MapConfig":
        try:
            with open(path, "r", encoding="utf-8") as f:
                js = json.load(f)
        except Exception:
            js = {}

        # Support both schemas:
        # - legacy: {style, api_key, cache:{dir}, defaults:{lat,lon,zoom}, heading_up}
        # - current: {selected_style, api_key, cache:{dir}, default_location:{lat,lon}, zoom, min_zoom, max_zoom, mode}
        if not isinstance(js, dict):
            js = {}

        style = js.get("selected_style", None)
        if not style:
            style = js.get("style", "satellite")

        default_loc = js.get("default_location", None)
        defaults = js.get("defaults", None)

        if isinstance(default_loc, dict):
            dlat = default_loc.get("lat", 30.915260)
            dlon = default_loc.get("lon", 34.423723)
        elif isinstance(defaults, dict):
            dlat = defaults.get("lat", 30.915260)
            dlon = defaults.get("lon", 34.423723)
        else:
            dlat, dlon = (30.915260, 34.423723)

        zoom = js.get("zoom", None)
        if zoom is None and isinstance(defaults, dict):
            zoom = defaults.get("zoom", 17)
        if zoom is None:
            zoom = 17

        min_zoom = js.get("min_zoom", 0)
        max_zoom = js.get("max_zoom", 20)

        # heading_up can be a boolean or implied by mode.
        heading_up = js.get("heading_up", None)
        if heading_up is None:
            mode = str(js.get("mode", "north_up")).lower().strip()
            heading_up = (mode == "heading_up")

        auto_download = bool(js.get("auto_download", False))

        try:
            min_zoom_i = int(min_zoom)
        except Exception:
            min_zoom_i = 0
        try:
            max_zoom_i = int(max_zoom)
        except Exception:
            max_zoom_i = 20
        if max_zoom_i < min_zoom_i:
            min_zoom_i, max_zoom_i = max_zoom_i, min_zoom_i

        try:
            zoom_i = int(zoom)
        except Exception:
            zoom_i = 14
        zoom_i = max(min_zoom_i, min(max_zoom_i, zoom_i))

        return MapConfig(
            style          = str(style),
            api_key        = js.get("api_key", ""),
            cache_dir      = js.get("cache", {}).get("dir", "tile_cache"),
            default_lat    = float(dlat),
            default_lon    = float(dlon),
            default_zoom   = zoom_i,
            min_zoom       = min_zoom_i,
            max_zoom       = max_zoom_i,
            heading_up     = bool(heading_up),
            auto_download  = bool(auto_download),
        )

    def save(self, path: str = "map_config.json") -> None:
        try:
            # Best-effort: preserve unknown keys by round-tripping.
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                data = {}
            if not isinstance(data, dict):
                data = {}

            if any(k in data for k in ("selected_style", "default_location", "zoom", "styles", "provider")):
                # Current schema
                data.setdefault("provider", "maptiler")
                data["api_key"] = self.api_key
                data["selected_style"] = self.style
                data["default_location"] = {"lat": self.default_lat, "lon": self.default_lon}
                data["zoom"] = int(self.default_zoom)
                data["min_zoom"] = int(self.min_zoom)
                data["max_zoom"] = int(self.max_zoom)
                data["mode"] = "heading_up" if self.heading_up else "north_up"
                data["auto_download"] = bool(self.auto_download)
                cache = data.get("cache", {})
                if not isinstance(cache, dict):
                    cache = {}
                cache["dir"] = self.cache_dir
                data["cache"] = cache
            else:
                # Legacy schema
                data = {
                    "style": self.style,
                    "api_key": self.api_key,
                    "cache": {"dir": self.cache_dir},
                    "defaults": {
                        "lat": self.default_lat,
                        "lon": self.default_lon,
                        "zoom": self.default_zoom,
                    },
                    "heading_up": self.heading_up,
                    "auto_download": bool(self.auto_download),
                }

            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

# --------- Map tile worker (CPU-only) ---------
class _TileWorker(threading.Thread):
    """Downloads a 3x3 patch around (lat,lon,z) and composites to a single PIL image.
       Communicates back via a queue with (w,h,pil_image,timestamp). No GL calls here."""
    def __init__(self, api_key: str, cache_root: str):
        super().__init__(daemon=True)
        self.api_key = api_key
        self.cache_root = cache_root
        self.req_queue: "queue.Queue[Tuple[str,float,float,int,int,int,bool]]" = queue.Queue()
        self.out_queue: "queue.Queue[Tuple[int,int,Image.Image,float]]" = queue.Queue()
        self._running = True

    def stop(self):
        self._running = False

    def request(self, style: str, lat: float, lon: float, z: int, w: int, h: int, *, allow_download: bool):
        # coalesce latest by dropping older
        while not self.req_queue.empty():
            try: self.req_queue.get_nowait()
            except Exception: break
        self.req_queue.put((style, lat, lon, z, w, h, bool(allow_download)))

    def _cache_path(self, style: str, z: int, x: int, y: int) -> str:
        p = os.path.join(self.cache_root, style, str(z), str(x))
        _ensure_dir(p)
        return os.path.join(p, f"{y}.png")

    def _fetch_tile(self, style: str, z: int, x: int, y: int, *, allow_download: bool) -> Image.Image:
        z = int(z)
        n = int(2 ** max(0, z))
        x = int(x) % max(1, n)
        y = max(0, min(max(0, n - 1), int(y)))
        cache_path = self._cache_path(style, z, x, y)
        if os.path.exists(cache_path):
            try:
                with Image.open(cache_path) as im:
                    return im.convert("RGBA")
            except Exception:
                pass

        if not bool(allow_download):
            return _placeholder_tile()

        if not str(self.api_key or "").strip():
            return _placeholder_tile()

        url = _tile_url(style, z, x, y, self.api_key)
        try:
            r = requests.get(url, timeout=3)
            if r.status_code == 200 and r.content:
                try:
                    with open(cache_path, "wb") as f:
                        f.write(r.content)
                except Exception:
                    pass
                try:
                    with Image.open(io.BytesIO(r.content)) as im:
                        return im.convert("RGBA")
                except Exception:
                    pass
        except Exception:
            pass
        return _placeholder_tile()

    def run(self):
        while self._running:
            try:
                style, lat, lon, z, w, h, allow_download = self.req_queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                # Build 3x3 patch
                cx, cy = _deg2num(lat, lon, z)
                patch = Image.new("RGBA", (256*3, 256*3), (40,40,40,255))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        tile = self._fetch_tile(style, z, cx + dx, cy + dy, allow_download=bool(allow_download))
                        patch.paste(tile, ((dx+1)*256, (dy+1)*256))
                # Scale to desired window content size
                patch = patch.resize((max(2,w), max(2,h)), Image.BILINEAR)
                self.out_queue.put((w, h, patch, time.time()))
            except Exception:
                # return placeholder to avoid stalling UI
                self.out_queue.put((w, h, _placeholder_tile(max(2,w), max(2,h)), time.time()))

# --------- GL helpers (safe to call only after context exists) ---------
def _pil_to_gl_texture(img: Image.Image) -> int:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA", 0, -1)
    w, h = img.size
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex

# --------- Map widget (lazy GL init) ---------
class MapWidget:
    def __init__(self, cfg: MapConfig):
        self.cfg = cfg
        # CPU thread for tile downloads
        self.worker = _TileWorker(api_key=cfg.api_key, cache_root=cfg.cache_dir)
        self.worker.start()

        # Lazy GL members (must be created after context exists)
        self.placeholder_tex: Optional[int] = None
        self.view_tex: Optional[int] = None
        self.view_dims: Tuple[int,int] = (0,0)

        # Last composited PIL image waiting to upload
        self._pending_img: Optional[Image.Image] = None
        self._pending_dims: Tuple[int,int] = (0,0)

        # UI state
        self.zoom = max(int(cfg.min_zoom), min(int(cfg.max_zoom), int(cfg.default_zoom)))
        self.style = cfg.style if cfg.style in ("streets", "satellite", "outdoor", "topo", "toner") else "satellite"
        self.auto_download = bool(getattr(cfg, "auto_download", False))

        # Bulk download state (background thread; no GL calls)
        self._dl_stop = threading.Event()
        self._dl_lock = threading.Lock()
        self._dl_thread: Optional[threading.Thread] = None
        self._dl_total: int = 0
        self._dl_done: int = 0
        self._dl_err: Optional[str] = None

    def shutdown(self):
        try:
            self._dl_stop.set()
        except Exception:
            pass
        try:
            th = self._dl_thread
            if th and th.is_alive():
                th.join(timeout=1.0)
        except Exception:
            pass

        try:
            self.worker.stop()
        except Exception:
            pass
        try:
            self.worker.join(timeout=1.0)
        except Exception:
            pass
        # Best-effort GL cleanup (requires a current context).
        try:
            if self.view_tex:
                gl.glDeleteTextures(int(self.view_tex))
        except Exception:
            pass
        try:
            if self.placeholder_tex:
                gl.glDeleteTextures(int(self.placeholder_tex))
        except Exception:
            pass
        self.view_tex = None
        self.placeholder_tex = None

    def _dl_running(self) -> bool:
        th = self._dl_thread
        return bool(th and th.is_alive())

    def _start_download_all_submaps(self, *, lat: float, lon: float) -> None:
        if self._dl_running():
            return

        # Clear previous status
        with self._dl_lock:
            self._dl_total = 0
            self._dl_done = 0
            self._dl_err = None

        self._dl_stop.clear()
        style = str(self.style)
        z0 = int(self.zoom)

        def _work():
            try:
                # Define the "map area" as the current 3x3 tile patch at the current zoom (the displayed view).
                cx, cy = _deg2num(float(lat), float(lon), z0)
                n0 = int(2 ** max(0, z0))
                x0 = max(0, min(n0 - 1, int(cx) - 1))
                x1 = max(0, min(n0 - 1, int(cx) + 1))
                y0 = max(0, min(n0 - 1, int(cy) - 1))
                y1 = max(0, min(n0 - 1, int(cy) + 1))

                lat_max, lon_min = _num2deg(int(x0), int(y0), z0)
                lat_min, lon_max = _num2deg(int(x1) + 1, int(y1) + 1, z0)

                z_min = int(getattr(self.cfg, "min_zoom", 0))
                z_max = int(getattr(self.cfg, "max_zoom", 20))
                if z_max < z_min:
                    z_min, z_max = z_max, z_min

                # Build a de-duped list of tiles to fetch.
                # Safety guard: prevent accidental massive downloads if the current zoom is too low.
                max_tiles = 20000
                jobs: List[Tuple[int, int, int]] = []
                seen: set = set()
                for z in range(z_min, z_max + 1):
                    n = int(2 ** max(0, int(z)))
                    xa, ya = _deg2num(float(lat_max), float(lon_min), int(z))
                    xb, yb = _deg2num(float(lat_min), float(lon_max), int(z))

                    xa = max(0, min(n - 1, int(xa)))
                    xb = max(0, min(n - 1, int(xb)))
                    ya = max(0, min(n - 1, int(ya)))
                    yb = max(0, min(n - 1, int(yb)))

                    if xb < xa:
                        xa, xb = xb, xa
                    if yb < ya:
                        ya, yb = yb, ya

                    for yy in range(int(ya), int(yb) + 1):
                        for xx in range(int(xa), int(xb) + 1):
                            k = (int(z), int(xx), int(yy))
                            if k in seen:
                                continue
                            seen.add(k)
                            jobs.append(k)
                            if len(jobs) > max_tiles:
                                raise RuntimeError(
                                    f"Refusing to download >{max_tiles} tiles (current view is too large). Zoom in and try again."
                                )

                with self._dl_lock:
                    self._dl_total = int(len(jobs))
                    self._dl_done = 0
                    self._dl_err = None

                # Download into the existing cache layout used by _TileWorker.
                for idx, (z, x, y) in enumerate(jobs):
                    if self._dl_stop.is_set():
                        break
                    try:
                        # Reuse the worker's fetch logic so caching stays consistent.
                        self.worker._fetch_tile(style, int(z), int(x), int(y), allow_download=True)  # type: ignore[attr-defined]
                    except Exception:
                        pass
                    if (idx % 4) == 0 or idx == (len(jobs) - 1):
                        with self._dl_lock:
                            self._dl_done = int(idx + 1)
            except Exception as e:
                with self._dl_lock:
                    self._dl_err = str(e)

        self._dl_thread = threading.Thread(target=_work, daemon=True, name="map-prefetch")
        self._dl_thread.start()

    def _ensure_gl(self):
        """Create placeholder GL texture once the context is available."""
        if self.placeholder_tex is None:
            ph = _placeholder_tile(64, 64)
            self.placeholder_tex = _pil_to_gl_texture(ph)

    def _drain_worker(self):
        """Pull the latest PIL image from the worker; keep only the newest."""
        got = False
        while True:
            try:
                w, h, pil_img, _ts = self.worker.out_queue.get_nowait()
            except queue.Empty:
                break
            self._pending_img = pil_img
            self._pending_dims = (w, h)
            got = True
        return got

    def _upload_pending(self):
        """Upload latest PIL to GL (on main thread)."""
        if self._pending_img is None:
            return
        img = self._pending_img.convert("RGBA")
        w, h = self._pending_dims
        # Reuse the existing texture when possible to avoid churn.
        if self.view_tex and self.view_dims == (w, h):
            try:
                data = img.tobytes("raw", "RGBA", 0, -1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, int(self.view_tex))
                try:
                    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                except Exception:
                    pass
                gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, int(w), int(h), gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            except Exception:
                # Fallback: re-create on any upload failure.
                try:
                    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
                except Exception:
                    pass
                try:
                    gl.glDeleteTextures(int(self.view_tex))
                except Exception:
                    pass
                self.view_tex = _pil_to_gl_texture(img)
        else:
            # Delete old (if any) and create a fresh texture for the new size.
            if self.view_tex:
                try:
                    gl.glDeleteTextures(int(self.view_tex))
                except Exception:
                    pass
            self.view_tex = _pil_to_gl_texture(img)
            self.view_dims = (w, h)
        self._pending_img = None

    def _request_tiles(self, lat: float, lon: float, w: int, h: int):
        self.worker.request(self.style, lat, lon, self.zoom, w, h, allow_download=bool(self.auto_download))

    def _content_avail(self) -> Tuple[float, float]:
        # ImGui 1.x (pyimgui) API: get_content_region_available()
        try:
            sz = imgui.get_content_region_available()
            return float(sz.x), float(sz.y)
        except AttributeError:
            # fallback for older pyimgui
            try:
                return imgui.get_content_region_avail()
            except AttributeError:
                # last resort: fill most of the window
                return 640.0, 360.0

    def draw_window(self, win_w: int, win_h: int, telem) -> None:
        # Create a content window
        imgui.set_next_window_position(40, 40)
        imgui.set_next_window_size(win_w - 80, win_h - 160)
        imgui.begin("GPS Map", True, imgui.WINDOW_NO_COLLAPSE)

        # Top controls
        style_items = [
            ("streets", "Streets"),
            ("satellite", "Satellite"),
            ("outdoor", "Outdoor"),
            ("topo", "Topography"),
            ("toner", "Toner"),
        ]
        style_ids = [s for (s, _lbl) in style_items]
        style_labels = [lbl for (_s, lbl) in style_items]
        idx = style_ids.index(self.style) if self.style in style_ids else 0
        changed, idx = imgui.combo("Style", idx, style_labels)
        if changed:
            self.style = style_ids[idx]
            # switching style -> request new tiles (cache is separate per style)
            # no need to clear cache here; worker uses style in cache path

        changed_ad, ad = imgui.checkbox("Auto download maps", bool(self.auto_download))
        if changed_ad:
            self.auto_download = bool(ad)

        imgui.same_line()
        if imgui.button("-", 40, 0):
            self.zoom = max(int(self.cfg.min_zoom), self.zoom - 1)
        imgui.same_line()
        imgui.text(f"Zoom: {self.zoom}")
        imgui.same_line()
        if imgui.button("+", 40, 0):
            self.zoom = min(int(self.cfg.max_zoom), self.zoom + 1)

        # Pick center position
        lat = telem.lat if getattr(telem, "lat", None) is not None else self.cfg.default_lat
        lon = telem.lon if getattr(telem, "lon", None) is not None else self.cfg.default_lon

        # Bulk cache control
        imgui.same_line()
        if imgui.button("Download all sumbaps", 220, 0):
            self._start_download_all_submaps(lat=float(lat), lon=float(lon))

        with self._dl_lock:
            dl_total = int(self._dl_total)
            dl_done = int(self._dl_done)
            dl_err = self._dl_err
        if dl_total > 0:
            imgui.same_line()
            if dl_done >= dl_total and not self._dl_running():
                imgui.text(f"Done ({dl_done}/{dl_total})")
            else:
                imgui.text(f"Downloading ({dl_done}/{dl_total})")
        if dl_err:
            imgui.text_colored(f"Map download error: {dl_err}", 1.0, 0.3, 0.3, 1.0)

        # Available area for the map image
        avail_w, avail_h = self._content_avail()
        avail_w = max(2, int(avail_w))
        avail_h = max(2, int(avail_h))

        # Ask worker to fetch a patch for current view
        self._request_tiles(lat, lon, avail_w, avail_h)

        # Ensure GL textures exist (after context)
        self._ensure_gl()

        # Drain worker & upload any newly composited map image
        if self._drain_worker():
            self._upload_pending()

        # Draw the map image
        img_x, img_y = imgui.get_cursor_screen_pos()
        if self.view_tex:
            imgui.image(int(self.view_tex), avail_w, avail_h, (0.0, 1.0), (1.0, 0.0))
        else:
            # draw placeholder once
            imgui.image(int(self.placeholder_tex), min(64, avail_w), min(64, avail_h), (0.0, 1.0), (1.0, 0.0))

        # Drone marker at center (GPS position is the map center in this widget).
        draw = imgui.get_window_draw_list()
        cx = float(img_x) + float(avail_w) / 2.0
        cy = float(img_y) + float(avail_h) / 2.0

        col_drone = imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 1.0)
        col_head = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.95)
        try:
            draw.add_circle_filled(cx, cy, 6.0, col_drone, 20)
        except TypeError:
            draw.add_circle_filled((cx, cy), 6.0, col_drone, 20)

        # Heading arrow (from MAVLink heading_deg; 0°=North/up).
        try:
            heading_deg = float(getattr(telem, "heading_deg", 0.0))
        except Exception:
            heading_deg = None
        if heading_deg is not None:
            ang = math.radians(float(heading_deg))
            dx = math.sin(ang)
            dy = -math.cos(ang)
            L = 18.0
            tip_x = cx + dx * L
            tip_y = cy + dy * L
            try:
                draw.add_line(cx, cy, tip_x, tip_y, col_head, 2.0)
            except TypeError:
                draw.add_line((cx, cy), (tip_x, tip_y), col_head, 2.0)

            # Small arrowhead.
            head_len = 7.0
            head_w = 4.0
            base_x = cx + dx * (L - head_len)
            base_y = cy + dy * (L - head_len)
            px, py = (-dy, dx)  # perpendicular
            lx = base_x + px * head_w
            ly = base_y + py * head_w
            rx = base_x - px * head_w
            ry = base_y - py * head_w
            try:
                draw.add_line(lx, ly, tip_x, tip_y, col_head, 2.0)
                draw.add_line(rx, ry, tip_x, tip_y, col_head, 2.0)
            except TypeError:
                draw.add_line((lx, ly), (tip_x, tip_y), col_head, 2.0)
                draw.add_line((rx, ry), (tip_x, tip_y), col_head, 2.0)

        imgui.end()
