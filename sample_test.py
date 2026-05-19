"""
sample_test.py  —  Sample count comparison renderer
====================================================

Renders a set of scene regions at increasing sample counts, then stitches
the results into a single comparison image to help choose the right sample
count before committing to a multi-hour full render.

Output
------
One file:  <OUTPUT_DIR>/<name>/sample_comparison.jpg

  ┌──────────┬──────────┬──────────┬──────────┐  ← center strip
  │  32 smp  │  64 smp  │ 128 smp  │ 256 smp  │
  │  0m 12s  │  0m 24s  │  0m 48s  │  1m 36s  │
  ├──────────┼──────────┼──────────┼──────────┤  ← shadows strip
  │          │          │          │          │
  ├──────────┼──────────┼──────────┼──────────┤  ← corner strip
  │          │          │          │          │
  └──────────┴──────────┴──────────┴──────────┘

Panel width  =  COMPARISON_W // len(SAMPLE_COUNTS)  (auto — more samples = thinner panels)
Panel height =  PANEL_H (fixed)
Total image  =  COMPARISON_W  ×  PANEL_H × len(REGIONS)

Usage
-----
  Drag-and-drop .blend onto  SAMPLE TEST.bat
  — or —
  blender --background scene.blend --python sample_test.py
"""

import bpy
import os
import time
import datetime
import subprocess
import sys

# ── Auto-install Pillow into Blender's Python if missing ─────────────────────
# Blender's embedded Python omits the user site-packages directory where pip
# defaults to installing when the system site-packages is not writable.
# Step 1: add user site-packages to sys.path so a pre-existing install is found.
# Step 2: only if still missing, run pip (installs into that same user dir).
import importlib, site as _site
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

LONG_EDGE      = 120_000    # must match the actual render — used to compute zoom
                            # (determines pixel density: 1 panel pixel = 1 final pixel)
COMPARISON_W   = 7680       # total width of the stitched output image (fixed)
PANEL_H        = 1080       # height of each panel in pixels (fixed)
                            # panel_W = COMPARISON_W // len(SAMPLE_COUNTS)  — auto

# Sample counts to compare — remove or add values as needed.
# More values → thinner panels, same total width.
SAMPLE_COUNTS  = [32, 64, 128, 256, 512, 1024, 2048, 4096]

# Per-panel time limit in seconds (Cycles native time_limit).
# Cycles stops rendering when this is reached, even if samples are not done.
# 0 = no limit.
MAX_TIME_SECS  = 0        # 4 minutes

OUTPUT_DIR     = "//sample_test/"   # relative to .blend file, or absolute path
RENDER_NAME    = ""                 # sub-folder name; "" = use .blend filename
FILE_FORMAT    = "JPEG"
JPEG_QUALITY   = 92

USE_GPU        = True
COMPUTE_DEVICE = "OPTIX"           # "OPTIX" (RTX), "CUDA", "HIP", "METAL", "CPU"

NOISE_THRESHOLD = 0.001              # Adaptive sampling threshold (0.0 = disabled, fixed samples)
                                   # Typical values: 0.1 (fast), 0.01 (default), 0.001 (clean)

# Regions to render.
# Each entry: (name, fx, fy)
#   fx  =  horizontal fraction of the full image  (0.0 = left,   1.0 = right)
#   fy  =  vertical fraction                      (0.0 = top,    1.0 = bottom)
# The script zooms the camera to place (fx, fy) at the center of the panel.
REGIONS = [
    ("center",  0.50, 0.50),   # center of the image
    ("shadows", 0.05, 0.05),   # top-left corner — shadowed areas
    ("corner",  0.95, 0.95),   # bottom-right corner
]

# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── Helpers (copied from render_tiles_unlimited.py) ───────────────────────────

def _fmt_hms(seconds):
    """Format a duration as  Xh Ym Zs  (omits leading zero units)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def save_state(scene, cam_obj):
    """Capture every setting this script will modify."""
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
    return st


def restore_state(scene, cam_obj, st):
    """Restore all settings captured by save_state()."""
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
        scene.cycles.device      = st["cycles_device"]
    if hasattr(scene, "cycles") and "cycles_samples" in st:
        scene.cycles.samples             = st["cycles_samples"]
        scene.cycles.time_limit          = st["cycles_time_limit"]
        scene.cycles.use_adaptive_sampling= st["cycles_use_adaptive"]
        scene.cycles.adaptive_threshold  = st["cycles_adaptive_threshold"]


def apply_zoom(cam, N):
    """
    Zoom the camera in by factor N so one panel fills the full horizontal frame.
    sensor_fit = HORIZONTAL so lens / ortho_scale always refer to the horizontal axis.
    """
    cam.sensor_fit = "HORIZONTAL"
    if cam.type == "PERSP":
        cam.lens *= N
    elif cam.type == "ORTHO":
        cam.ortho_scale /= N
    else:
        raise RuntimeError(f"Camera type '{cam.type}' is not supported (PERSP or ORTHO only).")


def setup_gpu(scene, compute_device):
    if not hasattr(scene, "cycles"):
        print("  Render device : GPU requested but engine is not Cycles — skipped")
        return
    prefs  = bpy.context.preferences
    cprefs = prefs.addons["cycles"].preferences
    cprefs.compute_device_type = compute_device
    cprefs.get_devices()
    for dev in cprefs.devices:
        dev.use = True
    scene.cycles.device = "GPU"
    print(f"  Render device : GPU  [{compute_device}]")


# ── Region shift ──────────────────────────────────────────────────────────────

def apply_region_shift(cam_obj, cam, Z, fx, fy, aspect_y,
                       orig_loc, orig_ortho_scale,
                       framing_vec=None, orig_shift_x=0.0, orig_shift_y=0.0):
    """
    Point the camera at fraction (fx, fy) of the full image, at zoom Z.

    PERSP — uses shift_x / shift_y (same formula as render_tiles_unlimited.py
            but with continuous fractions instead of integer tile indices):
        shift_x = Z * (fx - 0.5) + orig_shift_x * Z
        shift_y = Z * (0.5 - fy) * aspect_y + orig_shift_y * Z * aspect_y

    ORTHO — offsets camera location along its local right/up world vectors
            (shift parameters shear a tilted ORTHO camera, so we avoid them).
    """
    if cam.type == "PERSP":
        cam.shift_x = Z * (fx - 0.5) + orig_shift_x * Z
        cam.shift_y = Z * (0.5 - fy) * aspect_y + orig_shift_y * Z * aspect_y
    else:  # ORTHO
        import mathutils
        mat   = cam_obj.matrix_world.to_3x3()
        right = mat.col[0].normalized()
        up    = mat.col[1].normalized()
        # orig_ortho_scale = full-scene width in world units (before apply_zoom)
        dx = (fx - 0.5) * orig_ortho_scale
        dy = (0.5 - fy) * orig_ortho_scale * aspect_y
        fv = framing_vec if framing_vec is not None else mathutils.Vector((0.0, 0.0, 0.0))
        cam_obj.location = (
            orig_loc
            + dx * mathutils.Vector(right)
            + dy * mathutils.Vector(up)
            + fv
        )


# ── Label ─────────────────────────────────────────────────────────────────────

def _draw_label(img, text, panel_x, panel_w, panel_h):
    """
    Burn 'N smp  |  Xm Ys' centered at the bottom of a panel area in img.
    White text with black outline for readability on any background.
    """
    draw = ImageDraw.Draw(img)

    font = None
    font_size = max(20, panel_w // 22)   # scale with panel width
    for font_path in (
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ):
        try:
            font = ImageFont.truetype(font_path, size=font_size)
            break
        except OSError:
            pass
    if font is None:
        font = ImageFont.load_default()

    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = panel_x + (panel_w - tw) // 2
    y    = panel_h - th - max(12, panel_h // 60)

    # Black outline
    for ox, oy in ((-2, 0), (2, 0), (0, -2), (0, 2),
                   (-1, -1), (1, -1), (-1, 1), (1, 1)):
        draw.text((x + ox, y + oy), text, fill=(0, 0, 0), font=font)
    # White fill
    draw.text((x, y), text, fill=(255, 255, 255), font=font)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_sample_test(
    long_edge       = LONG_EDGE,
    comparison_w    = COMPARISON_W,
    panel_h         = PANEL_H,
    sample_counts   = SAMPLE_COUNTS,
    max_time_secs   = MAX_TIME_SECS,
    output_dir      = OUTPUT_DIR,
    render_name     = RENDER_NAME,
    file_format     = FILE_FORMAT,
    jpeg_quality    = JPEG_QUALITY,
    use_gpu         = USE_GPU,
    compute_device  = COMPUTE_DEVICE,
    noise_threshold = NOISE_THRESHOLD,
    regions         = REGIONS,
):
    scene   = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("No active camera in the scene.")
    cam = cam_obj.data
    if cam.type not in ("PERSP", "ORTHO"):
        raise RuntimeError(f"Camera type '{cam.type}' is not supported — use PERSP or ORTHO.")

    r = scene.render

    # ── Dimensions (must be read BEFORE modifying resolution) ─────────────────
    aspect  = (r.resolution_x * r.pixel_aspect_x) / (r.resolution_y * r.pixel_aspect_y)
    total_W = long_edge if aspect >= 1.0 else max(1, round(long_edge * aspect))
    total_H = max(1, round(long_edge / aspect)) if aspect >= 1.0 else long_edge

    n       = len(sample_counts)
    panel_w = comparison_w // n
    Z       = total_W / panel_w          # zoom factor — same pixel density as final render
    aspect_y = panel_h / panel_w         # vertical step correction (same as tile_aspect_y)

    # ── Output path ───────────────────────────────────────────────────────────
    name    = (render_name
               or bpy.path.basename(
                   bpy.context.blend_data.filepath).replace(".blend", "")
               or "render")
    abs_out = os.path.join(bpy.path.abspath(output_dir), name)
    os.makedirs(abs_out, exist_ok=True)

    ext = "jpg" if file_format == "JPEG" else "png"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  sample_test.py  —  sample count comparison")
    print(f"{'─' * 64}")
    print(f"  Camera       : {cam_obj.name}  [{cam.type}]")
    print(f"  Full image   : {total_W:,} × {total_H:,} px  (long edge {long_edge:,})")
    print(f"  Panel size   : {panel_w} × {panel_h} px  (zoom ×{Z:.1f})")
    print(f"  Samples      : {sample_counts}")
    print(f"  Time limit   : {_fmt_hms(max_time_secs) if max_time_secs else 'none'} per panel")
    print(f"  Noise thresh : {noise_threshold if noise_threshold > 0 else 'disabled (fixed samples)'}")
    print(f"  Regions      : {[r[0] for r in regions]}")
    print(f"  Output       : {abs_out}")
    total_renders = len(regions) * n
    print(f"  Total panels : {total_renders}")
    print(f"{'═' * 64}\n")

    # ── Save state ────────────────────────────────────────────────────────────
    state = save_state(scene, cam_obj)

    orig_shift_x    = state["shift_x"]
    orig_shift_y    = state["shift_y"]
    orig_ortho_scale = state["ortho_scale"]   # full-scene width before zoom (ORTHO)
    orig_loc        = state["cam_location"]
    framing_vec     = None

    try:
        # ── GPU ───────────────────────────────────────────────────────────────
        if use_gpu:
            setup_gpu(scene, compute_device)

        # ── Adaptive sampling ─────────────────────────────────────────────────
        if hasattr(scene, "cycles"):
            if noise_threshold > 0:
                scene.cycles.use_adaptive_sampling = True
                scene.cycles.adaptive_threshold    = noise_threshold
                print(f"  Adaptive sampling : ON  (threshold {noise_threshold})")
            else:
                scene.cycles.use_adaptive_sampling = False
                print(f"  Adaptive sampling : OFF  (fixed sample counts)")

        # ── Render settings ───────────────────────────────────────────────────
        r.resolution_x          = panel_w
        r.resolution_y          = panel_h
        r.resolution_percentage = 100
        r.use_border            = False
        r.use_crop_to_border    = False
        r.use_compositing       = False
        r.use_sequencer         = False
        r.image_settings.file_format = file_format
        if file_format == "JPEG":
            r.image_settings.quality = jpeg_quality

        # ── Zoom camera so one panel = full frame ──────────────────────────────
        apply_zoom(cam, Z)

        # ── ORTHO: bake framing shift into a world-space offset ────────────────
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
                print(f"  Framing shift : baked into world-space offset ({orig_shift_x:+.4f}, {orig_shift_y:+.4f})")

        # ── Region × sample loop ──────────────────────────────────────────────
        region_strips = []   # PIL Images, kept in memory until final stack
        all_times     = []   # flat list of every panel's elapsed time

        for region_name, fx, fy in regions:
            print(f"\n  ── Region: {region_name}  (fx={fx}, fy={fy}) ──")

            apply_region_shift(
                cam_obj, cam, Z, fx, fy, aspect_y,
                orig_loc, orig_ortho_scale,
                framing_vec=framing_vec,
                orig_shift_x=orig_shift_x,
                orig_shift_y=orig_shift_y,
            )

            tmp_paths  = []
            time_taken = []

            for samples in sample_counts:
                scene.cycles.samples    = samples
                scene.cycles.time_limit = float(max_time_secs) if max_time_secs > 0 else 0.0

                tmp_path = os.path.join(abs_out,
                                        f"_tmp_{region_name}_{samples:05d}smp.{ext}")
                # Blender appends the extension — strip it from filepath
                r.filepath = tmp_path[:-(len(ext) + 1)]

                print(f"  [{region_name}]  {samples:>5} smp  …", end="", flush=True)
                t0      = time.time()
                bpy.ops.render.render(write_still=True)
                elapsed = time.time() - t0

                tmp_paths.append(tmp_path)
                time_taken.append(elapsed)
                all_times.append(elapsed)
                print(f"  {_fmt_hms(elapsed)}")

            # ── Stitch this region's strip ─────────────────────────────────────
            strip_w = panel_w * n
            strip   = Image.new("RGB", (strip_w, panel_h))

            for i, (path, smp, elapsed) in enumerate(
                    zip(tmp_paths, sample_counts, time_taken)):
                panel_img = Image.open(path)
                # Ensure panel is exactly panel_w × panel_h (Blender may round)
                if panel_img.size != (panel_w, panel_h):
                    panel_img = panel_img.resize((panel_w, panel_h), Image.LANCZOS)
                strip.paste(panel_img, (i * panel_w, 0))
                panel_img.close()

                label = f"{smp} smp  |  {_fmt_hms(elapsed)}"
                _draw_label(strip, label, i * panel_w, panel_w, panel_h)

            region_strips.append((region_name, strip))

            # Delete temp files immediately — no longer needed
            for p in tmp_paths:
                try:
                    os.remove(p)
                except OSError:
                    pass

        # ── Stack all region strips vertically into the final image ────────────
        final_h = panel_h * len(regions)
        final   = Image.new("RGB", (strip_w, final_h))

        for i, (region_name, strip) in enumerate(region_strips):
            final.paste(strip, (0, i * panel_h))
            strip.close()

        out_path = os.path.join(abs_out, "sample_comparison.jpg")
        save_kwargs = {"quality": jpeg_quality, "optimize": True}
        final.save(out_path, **save_kwargs)
        final.close()

        print(f"\n{'═' * 64}")
        print(f"  ✓  sample_comparison.jpg")
        print(f"     {strip_w} × {final_h} px  ({n} samples × {len(regions)} regions)")
        print(f"     Total render time : {_fmt_hms(sum(all_times))}")
        print(f"     {out_path}")
        print(f"{'═' * 64}\n")

    finally:
        # Always restore — even on Ctrl+C or error
        restore_state(scene, cam_obj, state)
        print("  Camera and render settings restored.")


if __name__ == "__main__":
    run_sample_test()
