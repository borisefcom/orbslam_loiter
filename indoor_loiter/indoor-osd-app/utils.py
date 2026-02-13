import os, time
import imgui, OpenGL.GL as gl

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
OSD_ASSETS_DIR = os.path.join(ASSETS_DIR, "osd")

# ---------- JSON I/O ----------
def load_json(path):
    import json
    with open(path,"r",encoding="utf-8") as f: return json.load(f)

def save_json(path, data):
    import json
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)

# ---------- PIL -> OpenGL texture ----------
def pil_to_gl_texture(img):
    from PIL import Image
    if not isinstance(img, Image.Image):
        raise TypeError("pil_to_gl_texture expects PIL.Image")
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA", 0, -1)
    w,h = img.size
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex

def create_empty_texture(w,h):
    tex = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, int(w), int(h), 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE, None)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tex

def update_texture_rgb(tex,w,h,bytes_):
    gl.glBindTexture(gl.GL_TEXTURE_2D, tex)
    # GL defaults to 4-byte rows; RGB rows are 3*w bytes and may be unaligned.
    try:
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
    except Exception:
        pass
    gl.glTexSubImage2D(gl.GL_TEXTURE_2D,0,0,0,int(w),int(h),gl.GL_RGB,gl.GL_UNSIGNED_BYTE,bytes_)
    gl.glBindTexture(gl.GL_TEXTURE_2D,0)

# ---------- ImGui draw-list helpers (v2-safe) ----------
def _rgba(col):
    if isinstance(col, int): return col
    r,g,b,a = col
    return imgui.get_color_u32_rgba(float(r),float(g),float(b),float(a))

def dl_add_image(dl, tex, x, y, w, h, flip_v=False):
    # pyimgui v2 signature:
    # add_image(texture_id, p_min:(x,y), p_max:(x,y), uv_min:(u,v)=(0,0), uv_max:(u,v)=(1,1), col=0xFFFFFFFF)
    p_min = (float(x), float(y))
    p_max = (float(x + w), float(y + h))
    if flip_v:
        uv_min = (0.0, 1.0)
        uv_max = (1.0, 0.0)
    else:
        uv_min = (0.0, 0.0)
        uv_max = (1.0, 1.0)
    dl.add_image(int(tex), p_min, p_max, uv_min, uv_max)

def dl_add_text(dl, x, y, color, text, font_scale=1.0):
    col = _rgba(color)
    # add_text signature in v2: (x, y, col, text)
    if font_scale != 1.0:
        imgui.push_style_var(imgui.STYLE_ALPHA, 1.0)  # placeholder (no font size change API), keep for symmetry
    dl.add_text(float(x), float(y), col, str(text))
    if font_scale != 1.0:
        imgui.pop_style_var(1)

def dl_add_rect_filled(dl, x1,y1,x2,y2, color, rounding=0.0):
    dl.add_rect_filled(float(x1), float(y1), float(x2), float(y2), _rgba(color), float(rounding))

def dl_add_line(dl, x1,y1,x2,y2, color, thickness=1.0):
    dl.add_line(float(x1), float(y1), float(x2), float(y2), _rgba(color), float(thickness))

def dl_add_circle_filled(dl, cx,cy, r, color, num_segments=24):
    dl.add_circle_filled(float(cx), float(cy), float(r), _rgba(color), int(num_segments))

def dl_add_circle(dl, cx,cy, r, color, num_segments=24, thickness=1.0):
    dl.add_circle(float(cx), float(cy), float(r), _rgba(color), int(num_segments), float(thickness))

# ---------- UI helpers ----------
def ui_button(label: str, extra_px: int = 60) -> bool:
    """
    Button that auto-sizes to label width using current font + padding.
    extra_px adds breathing room.
    """
    tw, th = imgui.calc_text_size(label)
    pad_x, pad_y = imgui.get_style().frame_padding
    return imgui.button(label, tw + pad_x * 2 + extra_px, th + pad_y * 2)

# --- Drop-in line plot with right-edge Min/Max labels at actual Y levels ---
def plot_line(dl, x, y, w, h, xs, ys, title="", fmt_val="{:.3f}", right_margin=86, show_minmax=True, y_min=None, y_max=None):
    """
    Immediate-mode line plot.
    - Draws frame + midline + polyline.
    - Shows Min/Max values on the RIGHT edge, *aligned to their Y levels*:
      Max is near the top, Min near the bottom (text sits inside the plot).
    - Back-compatible shape vs older helper.

    dl  : ImGui draw list
    x,y : top-left screen coords
    w,h : size in pixels
    xs  : list of timestamps (ms)
    ys  : list of values (floats)
    """

    import imgui
    # Colors
    col_border = imgui.get_color_u32_rgba(1, 1, 1, 0.70)
    col_axis   = imgui.get_color_u32_rgba(1, 1, 1, 0.15)
    col_line   = imgui.get_color_u32_rgba(0.30, 1.00, 0.60, 1.00)
    col_text   = imgui.get_color_u32_rgba(1, 1, 1, 0.85)
    col_tick   = imgui.get_color_u32_rgba(1, 1, 1, 0.45)

    # Optional title (above frame)
    if title:
        imgui.set_cursor_screen_pos((x, y - 25))
        imgui.text(title)

    # Frame
    dl.add_rect(x, y, x + w, y + h, col_border, 0.0, 0, 1.0)
    # Midline (horizontal)
    dl.add_line(x + 2, y + h * 0.5, x + w - 2, y + h * 0.5, col_axis, 1.0)

    # Empty series: still reserve right label area
    if not xs or not ys:
        if show_minmax:
            imgui.set_cursor_screen_pos((x + w - right_margin, y + 6))
            imgui.text_disabled("Min")
            imgui.set_cursor_screen_pos((x + w - right_margin, y + 24))
            imgui.text_disabled("Max")
        return

    # Domain/range
    xmin, xmax = float(xs[0]), float(xs[-1])
    if xmax <= xmin:
        xmax = xmin + 1.0
    data_ymin = min(ys)
    data_ymax = max(ys)

    # Optional fixed y-range (disables autoscale).
    ymin = data_ymin if y_min is None else float(y_min)
    ymax = data_ymax if y_max is None else float(y_max)
    if ymax <= ymin:
        ymax = ymin + 1e-6

    inv_dx = 1.0 / (xmax - xmin)
    inv_dy = 1.0 / (ymax - ymin)

    # Build polyline
    pts = []
    for xi, yi in zip(xs, ys):
        tx = (float(xi) - xmin) * inv_dx
        ty = (float(yi) - ymin) * inv_dy
        # Clamp to plot rect to avoid drawing outside when using fixed ranges.
        if ty < 0.0:
            ty = 0.0
        elif ty > 1.0:
            ty = 1.0
        px = x + tx * w
        py = y + (1.0 - ty) * h
        pts.append((px, py))
    if len(pts) >= 2:
        dl.add_polyline(pts, col_line, False, 2.0)

    # Right-edge Min/Max labels at actual Y positions
    if show_minmax:
        # Map values -> pixel Y
        def map_y(val: float) -> float:
            ty = (float(val) - ymin) * inv_dy
            # keep label mapping in-bounds if caller passes out-of-range values
            if ty < 0.0:
                ty = 0.0
            elif ty > 1.0:
                ty = 1.0
            return y + (1.0 - ty) * h

        # Guard rails so text stays inside the frame
        def clamp(v, lo, hi): return max(lo, min(hi, v))

        lbl_min = data_ymin if (y_min is None and y_max is None) else float(ymin)
        lbl_max = data_ymax if (y_min is None and y_max is None) else float(ymax)

        try:
            smin = fmt_val.format(lbl_min)
        except Exception:
            smin = f"{float(lbl_min):.3f}"
        try:
            smax = fmt_val.format(lbl_max)
        except Exception:
            smax = f"{float(lbl_max):.3f}"

        # Pixel Y for labels (nudge a bit away from top/bottom)
        top_pad = 6.0
        bot_pad = 6.0
        y_max_px = clamp(map_y(lbl_max), y + top_pad, y + h - bot_pad)
        y_min_px = clamp(map_y(lbl_min), y + top_pad, y + h - bot_pad)

        # Right column start
        x_lbl = x + w - right_margin + 4.0

        # Small ticks on the right edge for clarity
        dl.add_line(x + w - 6, y_max_px, x + w - 2, y_max_px, col_tick, 1.0)
        dl.add_line(x + w - 6, y_min_px, x + w - 2, y_min_px, col_tick, 1.0)

        # Text (use draw-list so we can place it exactly)
        dl.add_text(x_lbl, y_max_px - 30.0, col_text, f"{smax}")
        dl.add_text(x_lbl, y_min_px + 5.0,  col_text, f"{smin}")

# Back-compat alias if older code calls `_plot_line`
_plot_line = plot_line

class CursorManager:
    """
    Simple, no-auto-hide cursor manager so existing calls don't break.
    Shows the OS cursor and provides update()/draw() no-ops.
    """
    def __init__(self, window, auto_hide: bool = False, hide_after: float = 3.0):
        self.window = window
        try:
            glfw.set_input_mode(self.window, glfw.CURSOR, glfw.CURSOR_NORMAL)
        except Exception:
            pass

    def update(self, mode_is_config: bool = False):
        # Nothing to do – cursor always visible
        pass

    def draw(self, draw_list=None):
        # No custom cursor drawing
        pass
