"""
seam_test.py  —  Tile seam banding detector
============================================

For each noise threshold, renders two horizontally adjacent sub-panels
independently (exactly like two neighbouring tiles in the real render)
and stitches them. If the threshold causes banding, a visible step will
appear at the centre join. If the stitch is seamless, that threshold is
safe to use.

Output
------
  <OUTPUT_DIR>/<name>/seam_comparison.jpg

  ┌────┬─────┬────┬─────┬────┬─────┬────┬─────┬────┬─────┬────┬─────┐
  │ L  │  R  │ L  │  R  │ L  │  R  │ L  │  R  │ L  │  R  │ L  │  R  │
  │    th 0.100    │    th 0.050    │    th 0.010    │   th 0.005  …  │
  └─────────────── ┴─────────────── ┴─────────────── ┴───────────────┘
    seamless = good   step visible = too loose

Usage
-----
  Drag-and-drop .blend onto  SEAM TEST.bat
  — or —
  blender --background scene.blend --python seam_test.py
"""

import bpy
import os
import time
import subprocess
import sys
import importlib
import site as _site

# ── Auto-install Pillow ───────────────────────────────────────────────────────
_user_site = _site.getusersitepackages()
if _user_site not in sys.path:
    sys.path.insert(0, _user_site)

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError:
    print("  Pillow not found — installing into Blender's Python…")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
    importlib.invalidate_caches()
    from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None


# ── CONFIG ────────────────────────────────────────────────────────────────────

LONG_EDGE    = 120_000   # must match the actual render — used to compute zoom
COMPARISON_W = 7680      # total width of the output image
PANEL_H      = 1080      # height of the output image

# Thresholds to test. Each becomes one L+R pair.
NOISE_THRESHOLDS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]

# Fixed-samples reference pair at the end (adaptive OFF). Set to None to disable.
FIXED_SAMPLES = 2048

MIN_SAMPLES = 128    # adaptive sampling floor
MAX_SAMPLES = 4096   # adaptive sampling ceiling

# Scene point where the simulated tile boundary falls.
# Pick a dense, well-lit area — that's where banding is most likely.
# fx = horizontal fraction (0.0=left … 1.0=right)
# fy = vertical fraction   (0.0=top  … 1.0=bottom)
SEAM_FX = 0.50
SEAM_FY = 0.50

OUTPUT_DIR   = "//seam_test/"   # relative to .blend, or absolute path
RENDER_NAME  = ""               # sub-folder name; "" = use .blend filename
FILE_FORMAT  = "JPEG"
JPEG_QUALITY = 92

USE_GPU        = True
COMPUTE_DEVICE = "OPTIX"   # "OPTIX" (RTX), "CUDA", "HIP", "METAL", "CPU"

# ── END CONFIG ────────────────────────────────────────────────────────────────


def _fmt_hms(seconds):
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:   return f"{h}h {m:02d}m {sec:02d}s"
    if m:   return f"{m}m {sec:02d}s"
    return f"{sec}s"


def save_state(scene, cam_obj):
    r   = scene.render
    cam = cam_obj.data
    st  = dict(
        res_x         = r.resolution_x,
        res_y         = r.resolution_y,
        res_pct       = r.resolution_percentage,
        filepath      = r.filepath,
        fmt           = r.image_settings.file_format,
        quality       = r.image_settings.quality,
        use_border    = r.use_border,
        use_crop      = r.use_crop_to_border,
        use_comp      = r.use_compositing,
        use_seq       = r.use_sequencer,
        bmin_x        = r.border_min_x,
        bmax_x        = r.border_max_x,
        bmin_y        = r.border_min_y,
        bmax_y        = r.border_max_y,
        cam_type      = cam.type,
        focal_length  = cam.lens,
        ortho_scale   = cam.ortho_scale,
        shift_x       = cam.shift_x,
        shift_y       = cam.shift_y,
        sensor_fit    = cam.sensor_fit,
        cam_location  = cam_obj.location.copy(),
        cycles_device = scene.cycles.device if hasattr(scene, "cycles") else None,
    )
    if hasattr(scene, "cycles"):
        st["cycles_samples"]           = scene.cycles.samples
        st["cycles_time_limit"]        = scene.cycles.time_limit
        st["cycles_use_adaptive"]      = scene.cycles.use_adaptive_sampling
        st["cycles_adaptive_threshold"]= scene.cycles.adaptive_threshold
        st["cycles_adaptive_min"]      = scene.cycles.adaptive_min_samples
    return st


def restore_state(scene, cam_obj, st):
    r   = scene.render
    cam = cam_obj.data
    cam.type                     = st["cam_type"]
    cam.lens                     = st["focal_length"]
    cam.ortho_scale              = st["ortho_scale"]
    cam.shift_x                  = st["shift_x"]
    cam.shift_y                  = st["shift_y"]
    cam.sensor_fit               = st["sensor_fit"]
    r.resolution_x               = st["res_x"]
    r.resolution_y               = st["res_y"]
    r.resolution_percentage      = st["res_pct"]
    r.filepath                   = st["filepath"]
    r.image_settings.file_format = st["fmt"]
    r.image_settings.quality     = st["quality"]
    r.use_border                 = st["use_border"]
    r.use_crop_to_border         = st["use_crop"]
    r.use_compositing            = st["use_comp"]
    r.use_sequencer              = st["use_seq"]
    r.border_min_x               = st["bmin_x"]
    r.border_max_x               = st["bmax_x"]
    r.border_min_y               = st["bmin_y"]
    r.border_max_y               = st["bmax_y"]
    cam_obj.location             = st["cam_location"]
    if st["cycles_device"] is not None:
        scene.cycles.device = st["cycles_device"]
    if hasattr(scene, "cycles") and "cycles_samples" in st:
        scene.cycles.samples              = st["cycles_samples"]
        scene.cycles.time_limit           = st["cycles_time_limit"]
        scene.cycles.use_adaptive_sampling= st["cycles_use_adaptive"]
        scene.cycles.adaptive_threshold   = st["cycles_adaptive_threshold"]
        scene.cycles.adaptive_min_samples = st["cycles_adaptive_min"]


def apply_zoom(cam, N):
    cam.sensor_fit = "HORIZONTAL"
    if cam.type == "PERSP":
        cam.lens *= N
    elif cam.type == "ORTHO":
        cam.ortho_scale /= N
    else:
        raise RuntimeError(f"Camera type '{cam.type}' not supported.")


def setup_gpu(scene, compute_device):
    if not hasattr(scene, "cycles"):
        return
    prefs  = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.compute_device_type = compute_device
    cprefs.get_devices()
    for dev in cprefs.devices:
        dev.use = True
    scene.cycles.device = "GPU"
    print(f"  Render device : GPU  [{compute_device}]")


def apply_shift(cam_obj, cam, Z, fx, fy, aspect_y,
                orig_loc, orig_ortho_scale,
                framing_vec=None, orig_shift_x=0.0, orig_shift_y=0.0):
    if cam.type == "PERSP":
        cam.shift_x = Z * (fx - 0.5) + orig_shift_x * Z
        cam.shift_y = Z * (0.5 - fy) * aspect_y + orig_shift_y * Z * aspect_y
    else:
        import mathutils
        mat   = cam_obj.matrix_world.to_3x3()
        right = mat.col[0].normalized()
        up    = mat.col[1].normalized()
        dx = (fx - 0.5) * orig_ortho_scale
        dy = (0.5 - fy) * orig_ortho_scale * aspect_y
        fv = framing_vec if framing_vec is not None else mathutils.Vector((0.0, 0.0, 0.0))
        cam_obj.location = (
            orig_loc
            + dx * mathutils.Vector(right)
            + dy * mathutils.Vector(up)
            + fv
        )


def _load_font(size):
    for path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ):
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            pass
    return ImageFont.load_default()


def _draw_label(img, text, x0, w, h):
    draw = ImageDraw.Draw(img)
    font = _load_font(max(18, w // 22))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = x0 + (w - tw) // 2
    y    = h - th - max(12, h // 60)
    for ox, oy in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)):
        draw.text((x+ox, y+oy), text, fill=(0, 0, 0), font=font)
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_seam_test(
    long_edge        = LONG_EDGE,
    comparison_w     = COMPARISON_W,
    panel_h          = PANEL_H,
    noise_thresholds = NOISE_THRESHOLDS,
    fixed_samples    = FIXED_SAMPLES,
    min_samples      = MIN_SAMPLES,
    max_samples      = MAX_SAMPLES,
    seam_fx          = SEAM_FX,
    seam_fy          = SEAM_FY,
    output_dir       = OUTPUT_DIR,
    render_name      = RENDER_NAME,
    file_format      = FILE_FORMAT,
    jpeg_quality     = JPEG_QUALITY,
    use_gpu          = USE_GPU,
    compute_device   = COMPUTE_DEVICE,
):
    scene   = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("No active camera in the scene.")
    cam = cam_obj.data
    if cam.type not in ("PERSP", "ORTHO"):
        raise RuntimeError(f"Camera type '{cam.type}' not supported.")
    if not hasattr(scene, "cycles"):
        raise RuntimeError("Scene render engine is not Cycles.")

    r = scene.render

    # ── Dimensions ───────────────────────────────────────────────────────────
    aspect  = (r.resolution_x * r.pixel_aspect_x) / (r.resolution_y * r.pixel_aspect_y)
    total_W = long_edge if aspect >= 1.0 else max(1, round(long_edge * aspect))
    total_H = max(1, round(long_edge / aspect)) if aspect >= 1.0 else long_edge

    cols = [(th, False) for th in noise_thresholds]
    if fixed_samples is not None:
        cols.append((fixed_samples, True))
    n_cols = len(cols)

    # Each pair occupies panel_w pixels; each L/R sub-panel = seam_pw pixels.
    panel_w  = comparison_w // n_cols
    seam_pw  = panel_w // 2
    Z        = total_W / seam_pw     # zoom so 1 sub-panel px = 1 final render px
    aspect_y = panel_h / seam_pw

    # L is centred half a sub-panel to the left of seam_fx, R to the right.
    # Together they span exactly one panel_w of scene content meeting at seam_fx.
    seam_half = 0.5 / Z

    # ── Output ───────────────────────────────────────────────────────────────
    name    = (render_name
               or bpy.path.basename(
                   bpy.context.blend_data.filepath).replace(".blend", "")
               or "render")
    abs_out = os.path.join(bpy.path.abspath(output_dir), name)
    os.makedirs(abs_out, exist_ok=True)
    ext = "jpg" if file_format == "JPEG" else "png"

    print(f"\n{'═'*64}")
    print(f"  seam_test.py  —  tile boundary banding detector")
    print(f"{'─'*64}")
    print(f"  Camera      : {cam_obj.name}  [{cam.type}]")
    print(f"  Full image  : {total_W:,} × {total_H:,} px")
    print(f"  Sub-panel   : {seam_pw} × {panel_h} px  (zoom ×{Z:.1f})")
    print(f"  Seam centre : fx={seam_fx}  fy={seam_fy}")
    print(f"  Thresholds  : {noise_thresholds}")
    if fixed_samples:
        print(f"  Fixed ref   : {fixed_samples} samples")
    print(f"  Smp range   : min {min_samples} → max {max_samples}")
    print(f"  Output      : {abs_out}")
    print(f"  Total panels: {n_cols * 2}  ({n_cols} pairs × L+R)")
    print(f"{'═'*64}\n")

    state = save_state(scene, cam_obj)
    orig_shift_x     = state["shift_x"]
    orig_shift_y     = state["shift_y"]
    orig_ortho_scale = state["ortho_scale"]
    orig_loc         = state["cam_location"]
    framing_vec      = None

    strip_w = panel_w * n_cols
    strip   = Image.new("RGB", (strip_w, panel_h))

    try:
        if use_gpu:
            setup_gpu(scene, compute_device)

        r.resolution_percentage      = 100
        r.resolution_x               = seam_pw
        r.resolution_y               = panel_h
        r.use_border                 = False
        r.use_crop_to_border         = False
        r.use_compositing            = False
        r.use_sequencer              = False
        r.image_settings.file_format = file_format
        if file_format == "JPEG":
            r.image_settings.quality = jpeg_quality

        # ORTHO: bake framing shift
        if cam.type == "ORTHO":
            cam.shift_x = 0.0
            cam.shift_y = 0.0
            if orig_shift_x != 0.0 or orig_shift_y != 0.0:
                import mathutils as _mu
                _mat   = cam_obj.matrix_world.to_3x3()
                _right = _mat.col[0].normalized()
                _up    = _mat.col[1].normalized()
                framing_vec = (
                    orig_shift_x * orig_ortho_scale * _mu.Vector(_right)
                    + orig_shift_y * orig_ortho_scale * _mu.Vector(_up)
                )

        apply_zoom(cam, Z)

        for col_i, (val, is_fixed) in enumerate(cols):
            pair_x0 = col_i * panel_w
            short   = f"{val} smp" if is_fixed else f"th {val:.3f}"

            if is_fixed:
                scene.cycles.use_adaptive_sampling = False
                scene.cycles.samples               = val
                scene.cycles.time_limit            = 0.0
            else:
                scene.cycles.use_adaptive_sampling = True
                scene.cycles.adaptive_threshold    = val
                scene.cycles.adaptive_min_samples  = min_samples
                scene.cycles.samples               = max_samples
                scene.cycles.time_limit            = 0.0

            elapsed_total = 0.0

            for side, fx_c in (("L", seam_fx - seam_half), ("R", seam_fx + seam_half)):
                apply_shift(
                    cam_obj, cam, Z, fx_c, seam_fy, aspect_y,
                    orig_loc, orig_ortho_scale,
                    framing_vec=framing_vec,
                    orig_shift_x=orig_shift_x,
                    orig_shift_y=orig_shift_y,
                )
                tmp = os.path.join(abs_out, f"_tmp_{col_i}_{side}.{ext}")
                r.filepath = tmp[:-(len(ext) + 1)]

                print(f"  {short:16s}  {side}  …", end="", flush=True)
                t0      = time.time()
                bpy.ops.render.render(write_still=True)
                elapsed = time.time() - t0
                elapsed_total += elapsed
                print(f"  {_fmt_hms(elapsed)}")

                img = Image.open(tmp)
                if img.size != (seam_pw, panel_h):
                    img = img.resize((seam_pw, panel_h), Image.LANCZOS)
                paste_x = pair_x0 if side == "L" else pair_x0 + seam_pw
                strip.paste(img, (paste_x, 0))
                img.close()
                try: os.remove(tmp)
                except OSError: pass

            _draw_label(strip, f"{short}  |  {_fmt_hms(elapsed_total)}",
                        pair_x0, panel_w, panel_h)

        out_path = os.path.join(abs_out, "seam_comparison.jpg")
        strip.save(out_path, quality=jpeg_quality, optimize=True)
        strip.close()

        print(f"\n{'═'*64}")
        print(f"  ✓  seam_comparison.jpg")
        print(f"     {strip_w} × {panel_h} px")
        print(f"     {out_path}")
        print(f"{'═'*64}\n")

    finally:
        restore_state(scene, cam_obj, state)
        print("  Camera and render settings restored.")


if __name__ == "__main__":
    run_seam_test()
