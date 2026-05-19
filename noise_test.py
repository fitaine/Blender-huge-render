"""
noise_test.py  —  Adaptive noise threshold comparison renderer
==============================================================

Renders scene regions at increasing Cycles noise thresholds (adaptive
sampling) and stitches the results into a single comparison image.

Adds an optional seam-test row that renders two horizontally adjacent
sub-panels independently (exactly like two neighbouring tiles in the
full render) and stitches them, so you can inspect the simulated
tile boundary for banding or noise-level discontinuities.

Output
------
One file:  <OUTPUT_DIR>/<name>/noise_comparison.jpg

  ┌──────────┬──────────┬──────────┬──────────┬──────────┬──────────┐
  │ th 0.100 │ th 0.050 │ th 0.010 │ th 0.005 │ th 0.001 │ 2048 fx  │   ← region row
  │   12s    │   18s    │   45s    │  1m 12s  │  2m 10s  │  4m 20s  │
  ├──────────┼──────────┼──────────┼──────────┼──────────┼──────────┤
  │          │          │          │          │          │          │   ← shadows row
  ├────┬─────┼────┬─────┼────┬─────┼────┬─────┼────┬─────┼────┬─────┤
  │  L │  R  │  L │  R  │  L │  R  │  L │  R  │  L │  R  │  L │  R  │   ← seam row
  └────┴─────┴────┴─────┴────┴─────┴────┴─────┴────┴─────┴────┴─────┘
       ↑ seam      ↑ seam      ↑ seam      ↑ seam      ↑ seam

  The red line in each seam pair is the simulated tile boundary.
  L and R are adjacent regions rendered independently at the same threshold.
  If banding exists it will appear as a noise-level step across the red line.

Usage
-----
  Drag-and-drop .blend onto  NOISE TEST.bat
  — or —
  blender --background scene.blend --python noise_test.py
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
                         # (ensures 1 panel pixel = 1 pixel in the final render)
COMPARISON_W = 7680      # total width of the stitched output image
PANEL_H      = 1080      # height of each row in pixels

# Noise thresholds to compare. Lower = more samples = cleaner but slower.
# Blender default is 0.01. Add or remove values as needed.
NOISE_THRESHOLDS = [0.10, 0.05, 0.02, 0.01, 0.005, 0.001]

# Adaptive sampling bounds.
MIN_SAMPLES = 128    # floor — prevents gross under-sampling of "easy" tiles
MAX_SAMPLES = 4096   # ceiling — hard cap (Cycles stops even if not converged)

# Fixed-samples reference column at the end (adaptive OFF, exact sample count).
# Set to None to disable.
FIXED_SAMPLES = 2048

# Regions to render — same format as sample_test.py.
# (name, fx, fy): fx/fy are fractions of the full image (0.0=left/top, 1.0=right/bottom).
REGIONS = [
    ("center",  0.50, 0.50),
    ("shadows", 0.05, 0.05),
    ("corner",  0.95, 0.95),
]

# Seam test: simulates two adjacent tiles rendered independently.
# The seam row renders L (left sub-panel) and R (right sub-panel) separately
# at each threshold, then stitches them with a red line at the boundary.
# L and R are geometrically adjacent — together they span the same scene width
# as one main panel, split exactly at SEAM_REGION's (fx, fy).
SEAM_TEST   = True
SEAM_REGION = ("center", 0.50, 0.50)   # scene point where the seam will fall

OUTPUT_DIR   = "//noise_test/"   # relative to .blend, or absolute path
RENDER_NAME  = ""                # sub-folder name; "" = use .blend filename
FILE_FORMAT  = "JPEG"
JPEG_QUALITY = 92

USE_GPU        = True
COMPUTE_DEVICE = "OPTIX"   # "OPTIX" (RTX), "CUDA", "HIP", "METAL", "CPU"

# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── Helpers (adapted from sample_test.py) ─────────────────────────────────────

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


def _reset_cam_zoom(cam, cam_obj, st):
    """Undo any zoom by resetting lens/ortho_scale/location to saved state."""
    cam.lens         = st["focal_length"]
    cam.ortho_scale  = st["ortho_scale"]
    cam.shift_x      = st["shift_x"]
    cam.shift_y      = st["shift_y"]
    cam_obj.location = st["cam_location"]


def apply_zoom(cam, N):
    cam.sensor_fit = "HORIZONTAL"
    if cam.type == "PERSP":
        cam.lens *= N
    elif cam.type == "ORTHO":
        cam.ortho_scale /= N
    else:
        raise RuntimeError(f"Camera type '{cam.type}' not supported (PERSP or ORTHO only).")


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


def apply_region_shift(cam_obj, cam, Z, fx, fy, aspect_y,
                       orig_loc, orig_ortho_scale,
                       framing_vec=None, orig_shift_x=0.0, orig_shift_y=0.0):
    """
    Point the zoomed camera at scene fraction (fx, fy).
    Identical logic to sample_test.py / render_tiles_unlimited.py.
    """
    if cam.type == "PERSP":
        cam.shift_x = Z * (fx - 0.5) + orig_shift_x * Z
        cam.shift_y = Z * (0.5 - fy) * aspect_y + orig_shift_y * Z * aspect_y
    else:  # ORTHO
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


def _set_sampling(scene, val, is_fixed, min_samples, max_samples):
    """Configure Cycles adaptive sampling or fixed sample count."""
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


def _col_label_short(val, is_fixed):
    return f"{val} smp" if is_fixed else f"th {val:.3f}"


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


def _draw_label(img, text, x0, w, h, color=(255, 255, 255)):
    """Burn text centred at the bottom of the panel area."""
    draw = ImageDraw.Draw(img)
    font = _load_font(max(18, w // 22))
    bbox = draw.textbbox((0, 0), text, font=font)
    tw   = bbox[2] - bbox[0]
    th   = bbox[3] - bbox[1]
    x    = x0 + (w - tw) // 2
    y    = h - th - max(12, h // 60)
    for ox, oy in ((-2,0),(2,0),(0,-2),(0,2),(-1,-1),(1,-1),(-1,1),(1,1)):
        draw.text((x+ox, y+oy), text, fill=(0,0,0), font=font)
    draw.text((x, y), text, fill=color, font=font)


def _draw_seam_line(img, x, h):
    """Red vertical line marking the simulated tile boundary."""
    draw = ImageDraw.Draw(img)
    draw.line([(x, 0), (x, h - 1)], fill=(220, 30, 30), width=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_noise_test(
    long_edge        = LONG_EDGE,
    comparison_w     = COMPARISON_W,
    panel_h          = PANEL_H,
    noise_thresholds = NOISE_THRESHOLDS,
    min_samples      = MIN_SAMPLES,
    max_samples      = MAX_SAMPLES,
    fixed_samples    = FIXED_SAMPLES,
    regions          = REGIONS,
    seam_test        = SEAM_TEST,
    seam_region      = SEAM_REGION,
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
        raise RuntimeError(f"Camera type '{cam.type}' not supported — use PERSP or ORTHO.")
    if not hasattr(scene, "cycles"):
        raise RuntimeError("Scene render engine is not Cycles.")

    r = scene.render

    # ── Dimensions ───────────────────────────────────────────────────────────
    aspect  = (r.resolution_x * r.pixel_aspect_x) / (r.resolution_y * r.pixel_aspect_y)
    total_W = long_edge if aspect >= 1.0 else max(1, round(long_edge * aspect))
    total_H = max(1, round(long_edge / aspect)) if aspect >= 1.0 else long_edge

    # Columns: thresholds + optional fixed reference
    cols = [(th, False) for th in noise_thresholds]
    if fixed_samples is not None:
        cols.append((fixed_samples, True))
    n_cols   = len(cols)
    panel_w  = comparison_w // n_cols
    Z        = total_W / panel_w          # zoom factor — 1 panel px = 1 final px
    aspect_y = panel_h / panel_w

    # Seam sub-panel geometry (half width of a main panel → 2× zoom)
    seam_pw  = panel_w // 2              # width of each L or R sub-panel
    Z_seam   = total_W / seam_pw         # ≈ 2 × Z
    ay_seam  = panel_h / seam_pw         # aspect_y for seam sub-panels
    seam_half = 0.5 / Z_seam            # half-extent of one sub-panel in image fractions
    # L is centred at (fx − seam_half), R at (fx + seam_half)
    # Together they span [fx−1/Z_seam, fx+1/Z_seam] = one full Z_seam panel
    # Geometrically identical to two adjacent tiles in the real render.

    # ── Output path ───────────────────────────────────────────────────────────
    name    = (render_name
               or bpy.path.basename(
                   bpy.context.blend_data.filepath).replace(".blend", "")
               or "render")
    abs_out = os.path.join(bpy.path.abspath(output_dir), name)
    os.makedirs(abs_out, exist_ok=True)
    ext = "jpg" if file_format == "JPEG" else "png"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═'*64}")
    print(f"  noise_test.py  —  adaptive threshold comparison")
    print(f"{'─'*64}")
    print(f"  Camera       : {cam_obj.name}  [{cam.type}]")
    print(f"  Full image   : {total_W:,} × {total_H:,} px  (long edge {long_edge:,})")
    print(f"  Panel size   : {panel_w} × {panel_h} px  (zoom ×{Z:.1f})")
    print(f"  Thresholds   : {noise_thresholds}")
    if fixed_samples:
        print(f"  Fixed ref    : {fixed_samples} samples")
    print(f"  Smp range    : min {min_samples} → max {max_samples}")
    print(f"  Seam test    : {'yes — seam sub-panel ' + str(seam_pw) + ' px  (zoom ×' + f'{Z_seam:.1f})' if seam_test else 'no'}")
    print(f"  Regions      : {[rg[0] for rg in regions]}")
    print(f"  Output       : {abs_out}")
    n_renders = len(regions) * n_cols + (2 * n_cols if seam_test else 0)
    print(f"  Total panels : {n_renders}")
    print(f"{'═'*64}\n")

    # ── Save full state ───────────────────────────────────────────────────────
    state = save_state(scene, cam_obj)
    orig_shift_x     = state["shift_x"]
    orig_shift_y     = state["shift_y"]
    orig_ortho_scale = state["ortho_scale"]
    orig_loc         = state["cam_location"]
    framing_vec      = None

    strip_w    = panel_w * n_cols
    all_strips = []   # list of (label, PIL strip)

    try:
        if use_gpu:
            setup_gpu(scene, compute_device)

        # Common render settings (not zoom / resolution — those vary per section)
        r.resolution_percentage      = 100
        r.use_border                 = False
        r.use_crop_to_border         = False
        r.use_compositing            = False
        r.use_sequencer              = False
        r.image_settings.file_format = file_format
        if file_format == "JPEG":
            r.image_settings.quality = jpeg_quality

        # ORTHO: bake original framing shift into a world-space vector
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
                print(f"  Framing shift : baked ({orig_shift_x:+.4f}, {orig_shift_y:+.4f})")

        # ══ Section 1: main region rows ══════════════════════════════════════

        apply_zoom(cam, Z)
        r.resolution_x = panel_w
        r.resolution_y = panel_h

        for reg_name, fx, fy in regions:
            print(f"\n  ── Region: {reg_name}  (fx={fx}, fy={fy}) ──")

            apply_region_shift(
                cam_obj, cam, Z, fx, fy, aspect_y,
                orig_loc, orig_ortho_scale,
                framing_vec=framing_vec,
                orig_shift_x=orig_shift_x,
                orig_shift_y=orig_shift_y,
            )

            strip = Image.new("RGB", (strip_w, panel_h))

            for col_i, (val, is_fixed) in enumerate(cols):
                short = _col_label_short(val, is_fixed)
                _set_sampling(scene, val, is_fixed, min_samples, max_samples)

                tmp = os.path.join(abs_out,
                                   f"_tmp_{reg_name}_{col_i}.{ext}")
                r.filepath = tmp[:-(len(ext) + 1)]

                print(f"  [{reg_name}]  {short:16s}  …", end="", flush=True)
                t0      = time.time()
                bpy.ops.render.render(write_still=True)
                elapsed = time.time() - t0
                print(f"  {_fmt_hms(elapsed)}")

                img = Image.open(tmp)
                if img.size != (panel_w, panel_h):
                    img = img.resize((panel_w, panel_h), Image.LANCZOS)
                strip.paste(img, (col_i * panel_w, 0))
                img.close()

                label = f"{short}  |  {_fmt_hms(elapsed)}"
                _draw_label(strip, label, col_i * panel_w, panel_w, panel_h)
                try: os.remove(tmp)
                except OSError: pass

            all_strips.append((reg_name, strip))

        # ══ Section 2: seam test row ══════════════════════════════════════════

        if seam_test:
            reg_name, fx, fy = seam_region
            print(f"\n  ── Seam test (region: {reg_name}, fx={fx}) ──")
            print(f"     L centre: fx={fx - seam_half:.4f}  "
                  f"R centre: fx={fx + seam_half:.4f}  "
                  f"seam at fx={fx:.4f}")

            # Switch to seam zoom (undo Z, apply Z_seam)
            _reset_cam_zoom(cam, cam_obj, state)
            if cam.type == "ORTHO":
                cam.shift_x = 0.0
                cam.shift_y = 0.0
            apply_zoom(cam, Z_seam)
            r.resolution_x = seam_pw
            r.resolution_y = panel_h

            strip = Image.new("RGB", (strip_w, panel_h))

            for col_i, (val, is_fixed) in enumerate(cols):
                short    = _col_label_short(val, is_fixed)
                pair_x0  = col_i * panel_w
                seam_x   = pair_x0 + seam_pw   # pixel x of the seam line in strip

                _set_sampling(scene, val, is_fixed, min_samples, max_samples)

                elapsed_total = 0.0

                for side, fx_c in (("L", fx - seam_half), ("R", fx + seam_half)):
                    apply_region_shift(
                        cam_obj, cam, Z_seam, fx_c, fy, ay_seam,
                        orig_loc, orig_ortho_scale,
                        framing_vec=framing_vec,
                        orig_shift_x=orig_shift_x,
                        orig_shift_y=orig_shift_y,
                    )
                    tmp = os.path.join(abs_out,
                                       f"_tmp_seam_{col_i}_{side}.{ext}")
                    r.filepath = tmp[:-(len(ext) + 1)]

                    print(f"  [seam]  {short:16s}  {side}  …", end="", flush=True)
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

                # Label in salmon to distinguish seam row from main rows
                # (no seam line drawn — the point is to NOT see any boundary)
                label = f"{short}  |  {_fmt_hms(elapsed_total)}"
                _draw_label(strip, label,
                            pair_x0, panel_w, panel_h,
                            color=(255, 140, 140))

            all_strips.append(("seam_test", strip))

        # ══ Stack strips and save ════════════════════════════════════════════

        final_h = panel_h * len(all_strips)
        final   = Image.new("RGB", (strip_w, final_h))
        for i, (_, strip) in enumerate(all_strips):
            final.paste(strip, (0, i * panel_h))
            strip.close()

        out_path = os.path.join(abs_out, "noise_comparison.jpg")
        final.save(out_path, quality=jpeg_quality, optimize=True)
        final.close()

        print(f"\n{'═'*64}")
        print(f"  ✓  noise_comparison.jpg")
        print(f"     {strip_w} × {final_h} px  "
              f"({n_cols} thresholds × {len(regions)} regions"
              f"{' + seam row' if seam_test else ''})")
        print(f"     {out_path}")
        print(f"{'═'*64}\n")

    finally:
        restore_state(scene, cam_obj, state)
        print("  Camera and render settings restored.")


if __name__ == "__main__":
    run_noise_test()
