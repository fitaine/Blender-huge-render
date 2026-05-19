"""
render_tiles_65k.py  —  Tiled render via render region
=======================================================

Sets the scene to a 65 000-px long-edge resolution, then renders one
tile at a time using Blender's built-in render border (use_border +
use_crop_to_border).  Works unchanged for ORTHO and PERSP cameras and
any render ratio — the camera is never touched.

Usage
-----
  • Blender scripting editor: set CONFIG below, then Run Script.
  • CLI:
        blender --background my_scene.blend --python render_tiles_65k.py

Output
------
  <OUTPUT_DIR>/<render_name>/
      tile_r0000_c0000.jpg   ← top-left in image coordinates
      tile_r0000_c0001.jpg
      ...
      manifest.json

After rendering, stitch tiles with VIPS / ImageMagick, then run
generate_tiles.py as usual.  manifest.json carries the full dimensions.
"""

import bpy
import os
import json
import math
import time


def _fmt(seconds):
    """Format a duration in seconds as HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ── CONFIG ─────────────────────────────────────────────────────────────────────

LONG_EDGE     = 65_000     # pixels on the long dimension of the assembled image
TILE_W        = 4096       # tile width  in pixels  (power of 2 recommended)
TILE_H        = 4096       # tile height in pixels
OUTPUT_DIR    = "//tiles/" # relative to .blend file, or absolute path
RENDER_NAME   = ""         # sub-folder; "" = use .blend filename
FILE_FORMAT   = "JPEG"     # "JPEG" or "PNG"
JPEG_QUALITY  = 92         # ignored for PNG
SKIP_EXISTING = True       # True = resume interrupted renders

# GPU rendering — set to True to force GPU Compute (Cycles only).
# COMPUTE_DEVICE: "CUDA" (older NVIDIA), "OPTIX" (RTX series, fastest),
#                 "HIP" (AMD), "METAL" (Apple), "CPU" (force CPU)
USE_GPU        = True
COMPUTE_DEVICE = "OPTIX"   # change to "CUDA" if OptiX is unavailable

# Optional explicit size override — set both to bypass the auto-aspect logic.
# Values are rounded UP to the nearest tile multiple automatically.
# Set to 0 to use LONG_EDGE + scene aspect ratio instead (default behaviour).
TOTAL_W       = 0          # e.g. 65_536  (0 = auto)
TOTAL_H       = 0          # e.g. 63_488  (0 = auto)

# ── END CONFIG ─────────────────────────────────────────────────────────────────


def compute_grid(res_x, res_y, pax, pay, long_edge, tile_w, tile_h,
                 override_w=0, override_h=0):
    """
    Returns (total_W, total_H, cols, rows).

    total_W / total_H are the EXACT desired pixel dimensions — edge tiles will
    simply be smaller than tile_w / tile_h (just like VIPS, DZI, Photoshop).
    No snapping to tile multiples: the grid just needs enough columns/rows to
    cover the full image.

    If override_w and override_h are both > 0 those values are used directly.
    Otherwise the scene aspect ratio is applied to long_edge.
    """
    if override_w > 0 and override_h > 0:
        total_W = override_w
        total_H = override_h
    else:
        aspect  = (res_x * pax) / (res_y * pay)   # W / H pixel ratio
        if aspect >= 1.0:
            total_W = long_edge
            total_H = max(1, round(long_edge / aspect))
        else:
            total_H = long_edge
            total_W = max(1, round(long_edge * aspect))

    cols = math.ceil(total_W / tile_w)
    rows = math.ceil(total_H / tile_h)
    return total_W, total_H, cols, rows


def render_tiles(
    long_edge    = LONG_EDGE,
    tile_w       = TILE_W,
    tile_h       = TILE_H,
    output_dir   = OUTPUT_DIR,
    render_name  = RENDER_NAME,
    file_format  = FILE_FORMAT,
    jpeg_quality = JPEG_QUALITY,
    skip_existing = SKIP_EXISTING,
    total_w      = TOTAL_W,
    total_h      = TOTAL_H,
    use_gpu      = USE_GPU,
    compute_device = COMPUTE_DEVICE,
):
    scene   = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("No active camera in the scene.")

    r = scene.render

    # ── grid from scene aspect (or explicit override) ─────────────────────────
    total_W, total_H, cols, rows = compute_grid(
        r.resolution_x, r.resolution_y,
        r.pixel_aspect_x, r.pixel_aspect_y,
        long_edge, tile_w, tile_h,
        override_w=total_w, override_h=total_h,
    )
    total = cols * rows

    # ── output path ───────────────────────────────────────────────────────────
    name    = (render_name
               or bpy.path.basename(bpy.context.blend_data.filepath).replace(".blend", "")
               or "render")
    abs_out = os.path.join(bpy.path.abspath(output_dir), name)
    os.makedirs(abs_out, exist_ok=True)

    print(f"\n{'═' * 64}")
    print(f"  Camera      : {cam_obj.name}  [{cam_obj.data.type}]")
    print(f"  Final image : {total_W:,} × {total_H:,} px"
          f"  (long edge = {max(total_W, total_H):,})")
    print(f"  Grid        : {cols} cols × {rows} rows  =  {total} tiles")
    print(f"  Tile size   : {tile_w} × {tile_h} px")
    print(f"  Output      : {abs_out}")

    ext = "jpg" if file_format == "JPEG" else "png"
    already_done = sum(
        1 for _r in range(rows) for _c in range(cols)
        if os.path.exists(
            os.path.join(abs_out, f"tile_r{(rows-1)-_r:04d}_c{_c:04d}.{ext}")
        )
    )
    remaining = total - already_done
    if already_done == 0:
        print(f"  Session     : fresh start — {total} tiles to render")
    else:
        print(f"  Session     : resuming — {already_done}/{total} done,"
              f" {remaining} remaining")
    print(f"{'═' * 64}\n")

    # ── save original render state ────────────────────────────────────────────
    orig = dict(
        res_x      = r.resolution_x,
        res_y      = r.resolution_y,
        res_pct    = r.resolution_percentage,
        filepath   = r.filepath,
        fmt        = r.image_settings.file_format,
        quality    = r.image_settings.quality,
        use_border = r.use_border,
        use_crop   = r.use_crop_to_border,
        bmin_x     = r.border_min_x,
        bmax_x     = r.border_max_x,
        bmin_y     = r.border_min_y,
        bmax_y     = r.border_max_y,
        cycles_device = scene.cycles.device if hasattr(scene, "cycles") else None,
    )

    # ── GPU setup ─────────────────────────────────────────────────────────────
    if use_gpu and hasattr(scene, "cycles"):
        prefs     = bpy.context.preferences
        cprefs    = prefs.addons["cycles"].preferences
        cprefs.compute_device_type = compute_device
        # Activate all available devices of the chosen type
        cprefs.get_devices()
        for dev in cprefs.devices:
            dev.use = True
        scene.cycles.device = "GPU"
        print(f"  Render device : GPU  [{compute_device}]")
    elif use_gpu:
        print("  Render device : GPU requested but scene uses non-Cycles engine — skipped")

    # ── apply full-image resolution (exact, not snapped to tile multiples) ────
    r.resolution_x          = total_W
    r.resolution_y          = total_H
    r.resolution_percentage = 100
    r.use_border            = True
    r.use_crop_to_border    = True
    r.image_settings.file_format = file_format
    if file_format == "JPEG":
        r.image_settings.quality = jpeg_quality

    # ── manifest (written before loop so it exists even if interrupted) ────────
    manifest = {
        "scene"      : name,
        "camera"     : cam_obj.name,
        "cam_type"   : cam_obj.data.type,
        "total_W"    : total_W,
        "total_H"    : total_H,
        "tile_W"     : tile_w,
        "tile_H"     : tile_h,
        "cols"       : cols,
        "rows"       : rows,
        "format"     : file_format,
        "long_edge"  : max(total_W, total_H),
        "tiles_total": total,
        "tiles_done" : already_done,
        "note"       : "row/col in filename = image coordinates (row 0 = top-left)",
    }
    manifest_path = os.path.join(abs_out, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── render loop ───────────────────────────────────────────────────────────
    errors        = []
    done          = 0
    render_times  = []          # durations of completed tiles (seconds)
    session_start = time.time()

    # Sort tiles center-outward so the main subject (usually centered) appears
    # first — makes it easy to judge exposure/colour before the full render finishes.
    center_r = (rows - 1) / 2.0
    center_c = (cols - 1) / 2.0
    tile_order = sorted(
        ((row, col) for row in range(rows) for col in range(cols)),
        key=lambda t: (t[0] - center_r) ** 2 + (t[1] - center_c) ** 2,
    )

    for row, col in tile_order:
        # Pixel-accurate border fractions — edge tiles are naturally
        # smaller (clamped to 1.0), matching how VIPS / DZI tiling works.
        bmin_x = (col       * tile_w) / total_W
        bmax_x = min((col + 1) * tile_w, total_W) / total_W
        bmin_y = (row       * tile_h) / total_H
        bmax_y = min((row + 1) * tile_h, total_H) / total_H

        # File naming: row 0 = top of image (flip Blender's Y-up)
        img_row  = (rows - 1) - row
        filename = f"tile_r{img_row:04d}_c{col:04d}.{ext}"
        filepath = os.path.join(abs_out, filename)

        done += 1
        if skip_existing and os.path.exists(filepath):
            print(f"  [{done:>4}/{total}] SKIP  {filename}")
            continue

        r.border_min_x = bmin_x
        r.border_max_x = bmax_x
        r.border_min_y = bmin_y
        r.border_max_y = bmax_y

        # Blender appends the extension; strip it from the path
        r.filepath = filepath[: -(len(ext) + 1)]

        print(f"  [{done:>4}/{total}]  r{img_row:04d} c{col:04d}"
              f"  border x[{bmin_x:.4f}–{bmax_x:.4f}]"
              f"  y[{bmin_y:.4f}–{bmax_y:.4f}]"
              f"  →  {filename}")

        try:
            tile_start = time.time()
            bpy.ops.render.render(write_still=True)
            tile_elapsed = time.time() - tile_start

            render_times.append(tile_elapsed)
            already_done += 1
            manifest["tiles_done"] = already_done
            with open(manifest_path, "w") as f:
                json.dump(manifest, f, indent=2)

            recent       = render_times[-10:]     # rolling window — last 10 tiles
            avg          = sum(recent) / len(recent)
            remaining    = total - done          # tiles still in queue
            eta          = avg * remaining
            elapsed      = time.time() - session_start
            print(f"         ↳  tile {tile_elapsed:5.1f}s  |"
                  f"  avg {avg:5.1f}s  |"
                  f"  elapsed {_fmt(elapsed)}  |"
                  f"  ETA {_fmt(eta)}")

        except Exception as exc:
            print(f"         !! ERROR: {exc}")
            errors.append(filename)

    # ── manifest — final update ───────────────────────────────────────────────
    if errors:
        manifest["errors"] = errors
    manifest["tiles_done"] = already_done
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── restore ───────────────────────────────────────────────────────────────
    r.resolution_x          = orig["res_x"]
    r.resolution_y          = orig["res_y"]
    r.resolution_percentage = orig["res_pct"]
    r.filepath              = orig["filepath"]
    r.image_settings.file_format = orig["fmt"]
    r.image_settings.quality     = orig["quality"]
    r.use_border            = orig["use_border"]
    r.use_crop_to_border    = orig["use_crop"]
    r.border_min_x          = orig["bmin_x"]
    r.border_max_x          = orig["bmax_x"]
    r.border_min_y          = orig["bmin_y"]
    r.border_max_y          = orig["bmax_y"]
    if orig["cycles_device"] is not None:
        scene.cycles.device = orig["cycles_device"]

    # ── summary ───────────────────────────────────────────────────────────────
    ok = total - len(errors)
    print(f"\n{'═' * 64}")
    print(f"  {ok}/{total} tiles rendered  |  {len(errors)} errors")
    if errors:
        print("  Failed:", ", ".join(errors))
    print(f"  manifest  →  {manifest_path}")
    print(f"{'═' * 64}\n")

    return manifest


if __name__ == "__main__":
    render_tiles()
