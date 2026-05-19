"""
render_tiles_unlimited.py  —  Tiled render via camera shift
============================================================

Renders a giant image at any resolution by shifting the camera's projection
plane one tile at a time.  Unlike render_tiles_65k.py, this script never sets
scene.render.resolution_x/y to the full image size, so Blender's 65 536 px
per-axis ceiling does not apply.

Technique
---------
  • Blender's render resolution is set to TILE_W × TILE_H for every call.
  • The camera is zoomed in by factor N so one tile fills the full frame:
      PERSP  →  camera.data.lens       ×= N
      ORTHO  →  camera.data.ortho_scale /= N
  • camera.data.shift_x / shift_y slide the projection plane to select which
    tile is rendered.  The physical camera never moves or rotates.
  • All camera and render settings are saved before the loop and restored in a
    finally block — the .blend file on disk is never written, so the camera is
    always safe to open in Blender regardless of how the script was stopped.

Tile naming
-----------
  tile_r0000_c0000.jpg  →  top-left of the full image
  tile_r{N-1}_c{N-1}   →  bottom-right

Output is fully compatible with stitch_to_dzi.py (Mode A) and
stitch_render_tiles.py; the manifest.json format is identical.

Usage 
-----
  Set LONG_EDGE and other CONFIG values below, then:

    blender --background scene.blend --python render_tiles_unlimited.py

  Or drag-and-drop onto RENDER TILES UNLIMITED.bat.

Limits
------
  GPU VRAM (scene complexity) and render time only — no Blender integer cap.
"""

import bpy
import os
import json
import math
import time
import datetime


# ── CONFIG ──────────────────────────────────────────────────────────────────

LONG_EDGE      = 120_000    # pixels on the long dimension of the assembled image
TILE_LONG      = 4096       # tile long-edge in pixels (power of 2 recommended)
                            # tile short-edge is derived from the image aspect ratio
OUTPUT_DIR     = "//tiles/" # relative to .blend file, or absolute path
RENDER_NAME    = ""         # sub-folder name;  "" = use .blend filename
FILE_FORMAT    = "JPEG"      # "JPEG" or "PNG"
JPEG_QUALITY   = 95         # ignored for PNG
SKIP_EXISTING  = True       # True = skip tiles already on disk (resume support)

# GPU rendering — Cycles only.
# COMPUTE_DEVICE: "OPTIX" (RTX, fastest), "CUDA" (older NVIDIA),
#                 "HIP" (AMD), "METAL" (Apple), "CPU"
USE_GPU        = True
COMPUTE_DEVICE = "OPTIX"

# Optional explicit size override — set both to bypass LONG_EDGE + auto-aspect.
# Set to 0 to use LONG_EDGE + scene aspect ratio (default behaviour).
TOTAL_W        = 0          # e.g. 120_000  (0 = auto)
TOTAL_H        = 0          # e.g.  80_400  (0 = auto)

# Preview mode — run before committing to a multi-hour full render.
# Two complementary checks (controlled by PREVIEW_GRID):
#
#   PREVIEW_GRID = 1  →  single full-frame tile (fast, ~1 min).
#                        N is forced to 1, so no shift is applied.
#                        Validates framing (FOV, camera aim) ONLY.
#                        Will NOT catch shift formula bugs (e.g. wrong vertical
#                        step for non-square images) — the bug that caused the
#                        Avoriaz staircase would have passed this check.
#
#   PREVIEW_GRID = N  →  N×N coarse grid at low res, stitched into one image.
#                        Uses the EXACT same apply_shift code path and tile
#                        aspect ratio as the full render.  Catches both framing
#                        AND shift formula bugs.  Recommended: PREVIEW_GRID = 5
#                        (25 tiles, typically 5–20 min depending on scene).
#
# Always use PREVIEW_GRID > 1 before a long render on a new scene or config.
#
PREVIEW_MODE    = False    # True = run preview, then stop
PREVIEW_RES     = 1920      # output preview width in pixels (height auto from aspect)
PREVIEW_GRID    = 5         # 1 = single full-frame tile; >1 = N×N coarse grid
PREVIEW_SAMPLES = 128         # Cycles samples per preview tile (low for speed)

# ── END CONFIG ──────────────────────────────────────────────────────────────


def _fmt(seconds):
    """Format a duration in seconds as HH:MM:SS."""
    s = int(seconds)
    return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"


# ── Grid ────────────────────────────────────────────────────────────────────

def compute_grid(res_x, res_y, pax, pay, long_edge, tile_long,
                 override_w=0, override_h=0):
    """
    Returns (total_W, total_H, tile_W, tile_H, N_cols, N_rows).

    The tile short-edge is derived from the image aspect ratio so that
    N_cols ≈ N_rows — which is required for the single-focal-length zoom to
    work correctly on both axes with sensor_fit = HORIZONTAL.
    """
    if override_w > 0 and override_h > 0:
        total_W = override_w
        total_H = override_h
    else:
        aspect = (res_x * pax) / (res_y * pay)   # W / H
        if aspect >= 1.0:                          # landscape or square
            total_W = long_edge
            total_H = max(1, round(long_edge / aspect))
        else:                                      # portrait
            total_H = long_edge
            total_W = max(1, round(long_edge * aspect))

    # Tile short-edge matches image aspect so N_cols == N_rows
    if total_W >= total_H:                         # landscape or square
        tile_W = tile_long
        tile_H = max(1, round(tile_long * total_H / total_W))
    else:                                          # portrait
        tile_H = tile_long
        tile_W = max(1, round(tile_long * total_W / total_H))

    N_cols = math.ceil(total_W / tile_W)
    N_rows = math.ceil(total_H / tile_H)

    return total_W, total_H, tile_W, tile_H, N_cols, N_rows


# ── State save / restore ────────────────────────────────────────────────────

def save_state(scene, cam_obj):
    """Capture every setting this script will modify."""
    r   = scene.render
    cam = cam_obj.data
    return dict(
        # render
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
        # camera data
        cam_type      = cam.type,
        focal_length  = cam.lens,
        ortho_scale   = cam.ortho_scale,
        shift_x       = cam.shift_x,
        shift_y       = cam.shift_y,
        sensor_fit    = cam.sensor_fit,
        # camera object (location modified for ORTHO)
        cam_location  = cam_obj.location.copy(),
        # cycles
        cycles_device = scene.cycles.device if hasattr(scene, "cycles") else None,
    )


def restore_state(scene, cam_obj, st):
    """Restore all settings captured by save_state()."""
    r   = scene.render
    cam = cam_obj.data
    # Camera type first — Blender ignores lens on ORTHO and ortho_scale on PERSP
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


# ── Camera zoom & shift ──────────────────────────────────────────────────────

def resolve_sensor_fit(cam, res_x, res_y):
    """
    Return 'HORIZONTAL' or 'VERTICAL', resolving AUTO from the render resolution.
    Blender resolves AUTO as HORIZONTAL when res_x >= res_y (landscape/square)
    and VERTICAL when res_y > res_x (portrait).
    """
    fit = cam.sensor_fit
    if fit == "AUTO":
        return "HORIZONTAL" if res_x >= res_y else "VERTICAL"
    return fit  # already 'HORIZONTAL' or 'VERTICAL'


def apply_zoom(cam, N_cols, N_rows, resolved_fit):
    """
    Zoom the camera in so one tile fills the full frame on the fit axis.
    sensor_fit is NOT forced — it is respected as resolved by resolve_sensor_fit().

    HORIZONTAL: zoom by N_cols so the horizontal sensor covers one tile width.
    VERTICAL:   zoom by N_rows so the vertical sensor covers one tile height.
    """
    if cam.type == "PERSP":
        cam.lens *= N_cols if resolved_fit == "HORIZONTAL" else N_rows
    elif cam.type == "ORTHO":
        cam.ortho_scale /= N_cols if resolved_fit == "HORIZONTAL" else N_rows
    else:
        raise RuntimeError(
            f"Camera type '{cam.type}' is not supported — use PERSP or ORTHO."
        )


def apply_shift(cam_obj, cam, N_cols, N_rows, col, row,
                orig_loc, tile_W_world, tile_H_world,
                framing_vec=None, orig_shift_x=0.0, orig_shift_y=0.0,
                tile_aspect_y=1.0, resolved_fit="HORIZONTAL"):
    """
    Position the camera to render tile (col, row).

    PERSP — lens shift (shift_x / shift_y):
      shift_x/shift_y are always in sensor_width fractions, but which axis
      the sensor_width anchors depends on resolved_fit:

      HORIZONTAL (landscape / explicit HORIZONTAL):
        sensor_width anchors the horizontal axis → zoom by N_cols.
        shift_x step = 1.0 (one tile width per column)
        shift_y step = tile_H/tile_W (scale to sensor-width units)
        orig shifts scaled by N_cols.

      VERTICAL (portrait AUTO or explicit VERTICAL):
        sensor_width anchors the vertical axis → zoom by N_rows.
        shift_y step = 1.0 (one tile height per row)
        shift_x step = tile_W/tile_H (scale to sensor-height units)
        orig shifts scaled by N_rows.

    ORTHO — camera location offset:
      shift_x / shift_y on a tilted ORTHO camera move the projection in
      camera-local space, which has a world-vertical component when the camera
      is not axis-aligned.  This creates a shear/staircase in the DZI output.
      Solution: move cam_obj.location along the camera's local right/up axes
      in world space — unambiguous for any camera orientation.
      Original framing shifts are zeroed out before the loop and passed in as
      framing_vec (a world-space Vector) so the user's intended crop is preserved.
    """
    if cam.type == "PERSP":
        if resolved_fit == "HORIZONTAL":
            cam.shift_x = (col + 0.5 - N_cols / 2.0) + orig_shift_x * N_cols
            cam.shift_y = (N_rows / 2.0 - row - 0.5) * tile_aspect_y + orig_shift_y * N_cols
        else:  # VERTICAL — sensor_width anchors the vertical axis
            cam.shift_x = (col + 0.5 - N_cols / 2.0) * (1.0 / tile_aspect_y) + orig_shift_x * N_rows
            cam.shift_y = (N_rows / 2.0 - row - 0.5) + orig_shift_y * N_rows
    else:  # ORTHO
        import mathutils
        mat   = cam_obj.matrix_world.to_3x3()
        right = mat.col[0].normalized()   # camera local X  =  right (unit vector)
        up    = mat.col[1].normalized()   # camera local Y  =  up    (unit vector)
        dx = (col  + 0.5 - N_cols / 2.0) * tile_W_world
        dy = (N_rows / 2.0 - row  - 0.5) * tile_H_world  # positive = move up = row 0 top
        fv = framing_vec if framing_vec is not None else mathutils.Vector((0.0, 0.0, 0.0))
        cam_obj.location = (
            orig_loc
            + dx * mathutils.Vector(right)
            + dy * mathutils.Vector(up)
            + fv
        )


# ── GPU setup ────────────────────────────────────────────────────────────────

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


# ── Per-scene timing file ────────────────────────────────────────────────────

def load_timing(timing_path):
    """Return timing dict from the timing file, or {} if absent / unreadable."""
    if not os.path.exists(timing_path):
        return {}
    try:
        with open(timing_path) as f:
            return json.load(f)
    except Exception:
        return {}


def save_timing(timing_path, avg_seconds, total_seconds,
                tile_W, tile_H, compute_device, scene_name):
    data = {
        "avg_seconds_per_tile"  : round(avg_seconds, 2),
        "total_render_seconds"  : round(total_seconds, 1),
        "tile_W"                : tile_W,
        "tile_H"                : tile_H,
        "compute_device"        : compute_device,
        "scene"                 : scene_name,
        "timestamp"             : datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    with open(timing_path, "w") as f:
        json.dump(data, f, indent=2)


def _backfill_tile_times(tile_dir, tile_times, N_rows, N_cols):
    """
    Estimate render time for tiles that have no recorded time, using file
    modification timestamps.  Gap between consecutive tiles (sorted by mtime)
    ≈ render time of the later tile.  Gaps > 1 hour are treated as session
    breaks and skipped.  Returns the number of tiles backfilled.
    """
    GAP_THRESHOLD = 3600
    tiles_on_disk = []
    for row in range(N_rows):
        for col in range(N_cols):
            base = os.path.join(tile_dir, f"tile_r{row:04d}_c{col:04d}")
            for x in (".jpg", ".jpeg", ".png"):
                p = base + x
                if os.path.exists(p):
                    tiles_on_disk.append((os.path.getmtime(p), os.path.basename(p)))
                    break
    tiles_on_disk.sort()
    backfilled = 0
    for i in range(1, len(tiles_on_disk)):
        _, fname = tiles_on_disk[i]
        if fname in tile_times:
            continue
        gap = tiles_on_disk[i][0] - tiles_on_disk[i - 1][0]
        if 0 < gap < GAP_THRESHOLD:
            tile_times[fname] = round(gap, 1)
            backfilled += 1
    return backfilled


# ── Settings snapshot ────────────────────────────────────────────────────────

def _save_snapshot(path, state, cam_obj, total_W, total_H, tile_W, tile_H, samples):
    """
    Save critical render settings to render_snapshot.json at the start of the
    first session.  Used by _check_snapshot() on every subsequent resume to
    detect settings changes that would corrupt the stitched image.
    """
    import math as _math
    rot  = cam_obj.rotation_euler
    snap = {
        "camera_name":  cam_obj.name,
        "cam_type":     state["cam_type"],
        "focal_length": round(state["focal_length"], 3) if state["cam_type"] == "PERSP" else None,
        "ortho_scale":  round(state["ortho_scale"],  4) if state["cam_type"] == "ORTHO" else None,
        "shift_x":      round(state["shift_x"], 4),
        "shift_y":      round(state["shift_y"], 4),
        "cam_location": [round(state["cam_location"].x, 3),
                         round(state["cam_location"].y, 3),
                         round(state["cam_location"].z, 3)],
        "cam_rotation": [round(_math.degrees(rot.x), 3),
                         round(_math.degrees(rot.y), 3),
                         round(_math.degrees(rot.z), 3)],
        "samples":      samples,
        "adaptive_sampling": getattr(bpy.context.scene.cycles, "use_adaptive_sampling", None),
        "adaptive_threshold": getattr(bpy.context.scene.cycles, "adaptive_threshold", None),
        "adaptive_min_samples": getattr(bpy.context.scene.cycles, "adaptive_min_samples", None),
        "frame":        bpy.context.scene.frame_current,
        "total_W":      total_W,
        "total_H":      total_H,
        "tile_W":       tile_W,
        "tile_H":       tile_H,
        "file_format":  state["fmt"],
        "saved_at":     datetime.datetime.now(datetime.timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(snap, f, indent=2)


def _check_snapshot(path, state, cam_obj, total_W, total_H, tile_W, tile_H, samples):
    """
    Compare current settings against the saved snapshot.
    Returns a list of (field, saved_value, current_value) tuples for each mismatch.
    Returns [] if no snapshot exists (old render folders) or all settings match.
    """
    import math as _math
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            snap = json.load(f)
    except Exception:
        return []

    rot     = cam_obj.rotation_euler
    current = {
        "camera_name":  cam_obj.name,
        "cam_type":     state["cam_type"],
        "focal_length": state["focal_length"] if state["cam_type"] == "PERSP" else None,
        "ortho_scale":  state["ortho_scale"]  if state["cam_type"] == "ORTHO" else None,
        "shift_x":      state["shift_x"],
        "shift_y":      state["shift_y"],
        "cam_location": [state["cam_location"].x,
                         state["cam_location"].y,
                         state["cam_location"].z],
        "cam_rotation": [_math.degrees(rot.x),
                         _math.degrees(rot.y),
                         _math.degrees(rot.z)],
        "samples":      samples,
        "total_W":      total_W,
        "total_H":      total_H,
        "tile_W":       tile_W,
        "tile_H":       tile_H,
        "file_format":  state["fmt"],
    }

    mismatches = []

    # Exact-match fields
    for field in ("camera_name", "cam_type", "samples", "file_format",
                  "total_W", "total_H", "tile_W", "tile_H"):
        if field not in snap:
            continue
        if snap[field] != current[field]:
            mismatches.append((field, snap[field], current[field]))

    # Float fields with tolerance
    for field, tol in (("focal_length", 1e-3), ("ortho_scale", 1e-3),
                       ("shift_x", 1e-3),      ("shift_y",     1e-3)):
        sv = snap.get(field)
        cv = current[field]
        if sv is None and cv is None:
            continue
        if sv is None or cv is None or abs(sv - cv) > tol:
            mismatches.append((field, sv, round(cv, 4) if cv is not None else None))

    # Vector fields — checked per axis
    for field, tol in (("cam_location", 0.1), ("cam_rotation", 0.01)):
        sv = snap.get(field)
        cv = current[field]
        if sv is None:
            continue
        for i, axis in enumerate(("x", "y", "z")):
            if abs(sv[i] - cv[i]) > tol:
                mismatches.append((f"{field}.{axis}", round(sv[i], 3), round(cv[i], 3)))

    return mismatches


# ── Scene statistics ─────────────────────────────────────────────────────────

def _collect_scene_stats(scene, cam_obj):
    """
    Collect scene metadata for the statistics file.
    Must be called BEFORE any render settings or camera are modified.
    Returns a dict; missing values are omitted (no crash on unusual setups).
    """
    import math as _math
    import mathutils
    stats = {}

    # ── Camera ────────────────────────────────────────────────────────────────
    try:
        cam = cam_obj.data
        cam_stats = {"name": cam_obj.name, "type": cam.type}

        if cam.type == "PERSP":
            cam_stats["focal_length_mm"] = round(cam.lens, 2)
            cam_stats["fov_deg"]         = round(_math.degrees(cam.angle), 1)
            cam_stats["sensor_mm"]       = round(cam.sensor_width, 1)
        elif cam.type == "ORTHO":
            cam_stats["ortho_scale"]     = round(cam.ortho_scale, 3)

        loc = cam_obj.location
        cam_stats["location"] = (round(loc.x, 1), round(loc.y, 1), round(loc.z, 1))
        cam_stats["altitude_m"] = round(loc.z, 1)

        # Euler angles → degrees (XYZ order)
        rot = cam_obj.rotation_euler
        cam_stats["rotation_deg"] = (
            round(_math.degrees(rot.x), 1),
            round(_math.degrees(rot.y), 1),
            round(_math.degrees(rot.z), 1),
        )
        # Tilt (pitch): angle below horizontal.  0° = looking straight down,
        # 90° = looking at the horizon.  Derived from the X Euler angle.
        tilt_from_nadir = abs(_math.degrees(rot.x))   # 0° = nadir, 90° = horizon
        cam_stats["tilt_from_nadir_deg"] = round(tilt_from_nadir, 1)

        cam_stats["clip_start"] = cam.clip_start
        cam_stats["clip_end"]   = cam.clip_end

        if cam.shift_x != 0.0 or cam.shift_y != 0.0:
            cam_stats["shift"] = (round(cam.shift_x, 4), round(cam.shift_y, 4))

        stats["camera"] = cam_stats
    except Exception:
        pass

    # Cycles render samples
    if hasattr(scene, "cycles"):
        stats["samples"] = scene.cycles.samples

    # Lamp objects visible to render
    n_lamps = sum(
        1 for obj in scene.objects
        if obj.type == "LIGHT" and not obj.hide_render
    )

    # Emissive mesh/curve objects: any material with an active Emission node
    def _has_emission(obj):
        if obj.hide_render:
            return False
        for slot in obj.material_slots:
            mat = slot.material
            if mat is None or not mat.use_nodes:
                continue
            for node in mat.node_tree.nodes:
                if node.type == "EMISSION":
                    # Check strength input — skip if 0 (disabled)
                    strength_input = node.inputs.get("Strength")
                    if strength_input is not None:
                        # Connected socket: assume non-zero
                        if strength_input.is_linked:
                            return True
                        if strength_input.default_value > 0:
                            return True
        return False

    n_emissive = sum(
        1 for obj in scene.objects
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"}
        and _has_emission(obj)
    )

    stats["n_lights"]   = n_lamps
    stats["n_emissive"] = n_emissive

    # Point / mesh stats — evaluate geometry nodes output
    depsgraph   = bpy.context.evaluated_depsgraph_get()
    total_verts = 0
    bbox_pts    = []

    for obj in scene.objects:
        if obj.hide_render or obj.type != "MESH":
            continue
        try:
            ev = obj.evaluated_get(depsgraph)
            # Vertex count from evaluated data (works for GeoNodes point clouds)
            total_verts += len(ev.data.vertices)
            # World-space bounding box (8 corners — fast, no vertex iteration)
            mat = ev.matrix_world
            for corner in ev.bound_box:
                bbox_pts.append(mat @ mathutils.Vector(corner))
        except Exception:
            pass

    if total_verts:
        stats["n_points"] = total_verts

    if bbox_pts:
        xs = [p.x for p in bbox_pts]
        ys = [p.y for p in bbox_pts]
        zs = [p.z for p in bbox_pts]
        stats["terrain_width"]   = max(xs) - min(xs)   # Blender units (metres for IGN)
        stats["terrain_depth"]   = max(ys) - min(ys)
        stats["terrain_height"]  = max(zs) - min(zs)
        stats["terrain_surface"] = stats["terrain_width"] * stats["terrain_depth"]

    return stats


# ── Render statistics ────────────────────────────────────────────────────────

def _fmt_hms(seconds):
    """Format seconds as  Xh Ym Zs  (omits leading zero units)."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {sec:02d}s"
    if m:
        return f"{m}m {sec:02d}s"
    return f"{sec}s"


def _write_stats(stats_path, name, cam_name, cam_type, compute_device,
                 total_W, total_H, tile_W, tile_H, N_cols, N_rows,
                 render_times, session_seconds, total_render_seconds,
                 tiles_done, tiles_total, errors, abs_out, ext,
                 completed_at, scene_stats=None):
    """Write a human-readable render statistics file."""
    sep  = "─" * 66
    sep2 = "═" * 66

    lines = []
    a = lines.append

    a(sep2)
    a(f"  RENDER STATISTICS  —  {name}")
    a(sep2)
    a("")

    # Status
    if tiles_done >= tiles_total and not errors:
        status = f"COMPLETE  ({tiles_done} / {tiles_total} tiles)"
    elif errors:
        status = f"COMPLETE WITH ERRORS  ({tiles_done} / {tiles_total} tiles,  {len(errors)} errors)"
    else:
        status = f"PARTIAL  ({tiles_done} / {tiles_total} tiles)"
    a(f"  Status         :  {status}")
    a(f"  Finished       :  {completed_at.strftime('%Y-%m-%d  %H:%M:%S')}")
    a("")

    # Image
    a(sep)
    a("  IMAGE")
    a(sep)
    mpx = total_W * total_H / 1_000_000
    gpx = mpx / 1000
    a(f"  Resolution     :  {total_W:,} × {total_H:,} px")
    a(f"  Size           :  {mpx:,.0f} Mpx  ({gpx:.2f} Gpx)")
    a(f"  Tile grid      :  {N_cols} × {N_rows}  =  {N_cols * N_rows} tiles")
    a(f"  Tile size      :  {tile_W:,} × {tile_H:,} px  each")
    a(f"  Method         :  camera_shift  [{cam_type}]  —  no resolution cap")
    a("")

    # Timing
    a(sep)
    a("  TIMING")
    a(sep)
    a(f"  Total render   :  {_fmt_hms(total_render_seconds)}  (all sessions combined)")
    a(f"  This session   :  {_fmt_hms(session_seconds)}")
    a(f"  Tiles rendered :  {tiles_done} / {tiles_total}")
    if errors:
        a(f"  Errors         :  {len(errors)}  →  {', '.join(errors)}")
    a("")

    # Per-tile
    if render_times:
        sorted_t = sorted(render_times)
        n        = len(sorted_t)
        fastest  = sorted_t[0]
        slowest  = sorted_t[-1]
        avg_t    = sum(sorted_t) / n
        median_t = (sorted_t[n // 2] if n % 2
                    else (sorted_t[n // 2 - 1] + sorted_t[n // 2]) / 2)
        a(sep)
        a(f"  PER-TILE  (this session — {n} tiles rendered)")
        a(sep)
        a(f"  Fastest        :  {fastest:.1f} s  ({_fmt_hms(fastest)})")
        a(f"  Slowest        :  {slowest:.1f} s  ({_fmt_hms(slowest)})")
        a(f"  Average        :  {avg_t:.1f} s  ({_fmt_hms(avg_t)})")
        a(f"  Median         :  {median_t:.1f} s")
        a("")

    # Scene
    ss = scene_stats or {}
    if ss:
        def _m(v):
            """Format a metre value: m if < 2 km, km otherwise."""
            return f"{v:,.0f} m" if v < 2000 else f"{v / 1000:.2f} km"
        a(sep)
        a("  SCENE")
        a(sep)
        if "n_points" in ss:
            pts = ss["n_points"]
            a(f"  Points (balls) :  {pts:,}  ({pts / 1_000_000:.2f} M)")
        if "samples" in ss:
            a(f"  Render samples :  {ss['samples']}")
        if "n_lights" in ss:
            n_em = ss.get("n_emissive", 0)
            a(f"  Lamps          :  {ss['n_lights']}")
            a(f"  Emissive objs  :  {n_em}  (meshes/curves with Emission shader)")
        if "terrain_width" in ss:
            w = ss["terrain_width"];  d = ss["terrain_depth"]
            h = ss["terrain_height"]; surf = ss["terrain_surface"]
            a(f"  Terrain width  :  {_m(w)}")
            a(f"  Terrain depth  :  {_m(d)}")
            a(f"  Terrain height :  {_m(h)}")
            surf_km2 = surf / 1_000_000
            a(f"  Ground surface :  {surf_km2:.2f} km²"
              + (f"  ({surf:,.0f} m²)" if surf_km2 < 10 else ""))
        a("")

    # Camera
    cs = (scene_stats or {}).get("camera", {})
    if cs:
        def _clip(v):
            return f"{v:,.1f} m" if v < 2000 else f"{v / 1000:.1f} km"
        a(sep)
        a("  CAMERA")
        a(sep)
        a(f"  Name           :  {cs.get('name', cam_name)}")
        a(f"  Type           :  {cs.get('type', cam_type)}")
        if "focal_length_mm" in cs:
            a(f"  Focal length   :  {cs['focal_length_mm']} mm")
        if "fov_deg" in cs:
            a(f"  Field of view  :  {cs['fov_deg']}°  (horizontal)")
        if "sensor_mm" in cs:
            a(f"  Sensor width   :  {cs['sensor_mm']} mm")
        if "ortho_scale" in cs:
            a(f"  Ortho scale    :  {cs['ortho_scale']}")
        loc = cs.get("location")
        if loc:
            a(f"  Location       :  X {loc[0]:,.1f}  Y {loc[1]:,.1f}  Z {loc[2]:,.1f}")
            a(f"  Altitude       :  {cs['altitude_m']:,.1f} m")
        rot = cs.get("rotation_deg")
        if rot:
            a(f"  Rotation       :  X {rot[0]}°   Y {rot[1]}°   Z {rot[2]}°")
            a(f"  Tilt           :  {cs['tilt_from_nadir_deg']}°  from nadir"
              "  (0° = straight down,  90° = horizon)")
        if "clip_start" in cs:
            a(f"  Clip range     :  {_clip(cs['clip_start'])}  →  {_clip(cs['clip_end'])}")
        if "shift" in cs:
            a(f"  Framing shift  :  x={cs['shift'][0]:+.4f}  y={cs['shift'][1]:+.4f}")
        a("")

    # Hardware
    a(sep)
    a("  HARDWARE")
    a(sep)
    a(f"  Compute        :  {compute_device}")
    a("")

    # Files
    a(sep)
    a("  FILES")
    a(sep)
    a(f"  Output folder  :  {abs_out}")
    try:
        tile_files  = [f for f in os.listdir(abs_out) if f.startswith("tile_") and f.endswith(f".{ext}")]
        total_bytes = sum(os.path.getsize(os.path.join(abs_out, f)) for f in tile_files)
        if total_bytes >= 1_073_741_824:
            size_str = f"{total_bytes / 1_073_741_824:.2f} GB"
        else:
            size_str = f"{total_bytes / 1_048_576:.1f} MB"
        a(f"  Tile files     :  {len(tile_files)} files  /  {size_str}")
    except Exception:
        pass
    a("")
    a(sep2)

    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ── Grid preview stitch ──────────────────────────────────────────────────────

def _stitch_grid_preview(tiles_dir, N_cols, N_rows, tile_W, tile_H, ext, out_path):
    """
    Stitch all N_cols×N_rows preview tiles into a single PNG using bpy image API.
    Called only from PREVIEW_MODE with PREVIEW_GRID > 1.

    Blender stores image pixels bottom-row-first (y=0 = bottom), so each tile
    array is flipped before placement, and the assembled canvas is flipped back
    before assigning to bpy.data.images.
    """
    import numpy as np

    canvas_W = N_cols * tile_W
    canvas_H = N_rows * tile_H
    canvas   = np.zeros((canvas_H, canvas_W, 4), dtype=np.float32)

    for row in range(N_rows):
        for col in range(N_cols):
            fp = os.path.join(tiles_dir, f"tile_r{row:04d}_c{col:04d}.{ext}")
            if not os.path.exists(fp):
                print(f"  ⚠  preview tile missing: {fp}")
                continue
            t    = bpy.data.images.load(fp)
            t_px = (np.array(t.pixels, dtype=np.float32)
                      .reshape(tile_H, tile_W, 4))
            t_px = np.flipud(t_px)               # Blender y=0 is bottom → flip to top-first
            canvas[row * tile_H:(row + 1) * tile_H,
                   col * tile_W:(col + 1) * tile_W] = t_px
            bpy.data.images.remove(t)

    out_img = bpy.data.images.new("_grid_preview", canvas_W, canvas_H, alpha=False)
    out_img.pixels = np.flipud(canvas).flatten().tolist()  # flip back for Blender
    out_img.filepath_raw = out_path
    out_img.file_format  = "PNG"
    out_img.save()
    bpy.data.images.remove(out_img)


# ── Main ────────────────────────────────────────────────────────────────────

def render_tiles(
    long_edge      = LONG_EDGE,
    tile_long      = TILE_LONG,
    output_dir     = OUTPUT_DIR,
    render_name    = RENDER_NAME,
    file_format    = FILE_FORMAT,
    jpeg_quality   = JPEG_QUALITY,
    skip_existing  = SKIP_EXISTING,
    total_w        = TOTAL_W,
    total_h        = TOTAL_H,
    use_gpu        = USE_GPU,
    compute_device = COMPUTE_DEVICE,
    preview_mode   = PREVIEW_MODE,
    preview_res    = PREVIEW_RES,
    preview_grid   = PREVIEW_GRID,
    preview_samples= PREVIEW_SAMPLES,
):
    scene   = bpy.context.scene
    cam_obj = scene.camera
    if cam_obj is None:
        raise RuntimeError("No active camera in the scene.")
    cam = cam_obj.data
    if cam.type not in ("PERSP", "ORTHO"):
        raise RuntimeError(
            f"Camera type '{cam.type}' is not supported — use PERSP or ORTHO."
        )

    r = scene.render

    # ── Scene stats (collected before any settings are modified) ──────────────
    scene_stats = _collect_scene_stats(scene, cam_obj)

    # ── Grid ──────────────────────────────────────────────────────────────────
    total_W, total_H, tile_W, tile_H, N_cols, N_rows = compute_grid(
        r.resolution_x, r.resolution_y,
        r.pixel_aspect_x, r.pixel_aspect_y,
        long_edge, tile_long,
        override_w=total_w, override_h=total_h,
    )
    total = N_cols * N_rows

    # ── Preview mode override ────────────────────────────────────────────────
    if preview_mode:
        if preview_grid <= 1:
            # Single full-frame tile: validates framing (FOV, camera aim) only.
            # N is forced to 1 → apply_shift is called with (row=0, col=0) → no
            # shift applied → this mode CANNOT catch shift formula bugs.
            N_cols  = 1
            N_rows  = 1
            total   = 1
            tile_W  = preview_res
            tile_H  = max(1, round(preview_res * total_H / total_W))
            total_W = tile_W
            total_H = tile_H
        else:
            # Coarse N×N grid: renders PREVIEW_GRID×PREVIEW_GRID tiles at low
            # resolution using the exact same apply_shift code path and tile
            # aspect ratio as the full render.  Catches framing AND shift bugs.
            # total_W / total_H are kept as full scene dimensions so the shift
            # formula covers the full scene range.
            N_cols = preview_grid
            N_rows = preview_grid
            total  = N_cols * N_rows
            tile_W = max(8, preview_res // N_cols)
            tile_H = max(8, round(tile_W * total_H / total_W))

    # ── Output path ───────────────────────────────────────────────────────────
    name    = (render_name
               or bpy.path.basename(
                   bpy.context.blend_data.filepath).replace(".blend", "")
               or "render")
    abs_out = os.path.join(bpy.path.abspath(output_dir), name)
    os.makedirs(abs_out, exist_ok=True)

    timing_path   = os.path.join(abs_out, "render_timing.json")
    manifest_path = os.path.join(abs_out, "manifest.json")
    ext = "jpg" if file_format == "JPEG" else "png"

    # ── Preview tile directory ─────────────────────────────────────────────────
    # Preview tiles live in a timestamped subfolder so they are never mixed with
    # real render tiles, never overwritten by a later preview run, and never
    # picked up by the resume counter or stitch_to_dzi.py.
    if preview_mode:
        ts       = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        tile_dir = os.path.join(abs_out, f"_preview_{ts}")
        os.makedirs(tile_dir, exist_ok=True)
    else:
        tile_dir = abs_out

    # ── Resume: count tiles already on disk ───────────────────────────────────
    # Check all formats — supports mixed folders (e.g. PNG sky + JPEG body tiles)
    def _tile_exists(td, r, c):
        base = os.path.join(td, f"tile_r{r:04d}_c{c:04d}")
        return any(os.path.exists(base + x) for x in (".png", ".jpg", ".jpeg"))

    already_done = sum(
        1 for _r in range(N_rows) for _c in range(N_cols)
        if _tile_exists(tile_dir, _r, _c)
    )
    remaining = total - already_done

    # ── ETA from previous session ──────────────────────────────────────────────
    prev_timing  = load_timing(timing_path)
    prev_avg     = prev_timing.get("avg_seconds_per_tile")
    prev_total_s = prev_timing.get("total_render_seconds", 0.0)
    if already_done > 0 and prev_total_s == 0.0 and not preview_mode:
        print(f"\n  ⚠  TIMING WARNING: resuming ({already_done} tiles done) but"
              f" render_timing.json has no accumulated time.")
        print(f"     Prior sessions' GPU time was lost (script update / file reset?).")
        print(f"     Total render time in render_stats.txt will be incomplete.\n")
    if prev_avg and remaining > 0:
        eta_str = f"~{_fmt(prev_avg * remaining)}  ({prev_avg:.1f} s/tile from last session)"
    elif prev_avg:
        eta_str = f"all tiles already done  ({prev_avg:.1f} s/tile from last session)"
    else:
        eta_str = "no prior timing data — will appear after first tile"

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  render_tiles_unlimited.py  —  camera-shift / no resolution cap")
    print(f"{'─' * 64}")
    print(f"  Camera      : {cam_obj.name}  [{cam.type}]")
    print(f"  Final image : {total_W:,} × {total_H:,} px"
          f"  (long edge = {max(total_W, total_H):,})")
    print(f"  Tile size   : {tile_W} × {tile_H} px")
    print(f"  Grid        : {N_cols} cols × {N_rows} rows  =  {total} tiles")
    print(f"  Output      : {tile_dir if preview_mode else abs_out}")
    if already_done == 0:
        print(f"  Session     : fresh start — {total} tiles to render")
    else:
        print(f"  Session     : resuming — {already_done}/{total} done,"
              f"  {remaining} remaining")
    print(f"  Est. time   : {eta_str}")
    if hasattr(scene, "cycles"):
        _adapt = getattr(scene.cycles, "use_adaptive_sampling", None)
        _thr   = getattr(scene.cycles, "adaptive_threshold",    None)
        _mins  = getattr(scene.cycles, "adaptive_min_samples",  None)
        if _adapt:
            print(f"  Sampling    : adaptive  {_mins}–{scene.cycles.samples}"
                  f"  threshold {_thr}")
        else:
            print(f"  Sampling    : fixed  {scene.cycles.samples} samples"
                  f"  (adaptive OFF)")
    print(f"{'═' * 64}\n")

    # ── Manifest — preserve tile_times from previous sessions ───────────────────
    _old_manifest = {}
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path) as _f:
                _old_manifest = json.load(_f)
        except Exception:
            pass
    tile_times = dict(_old_manifest.get("tile_times", {}))
    if already_done > 0 and not preview_mode:
        _backfilled = _backfill_tile_times(tile_dir, tile_times, N_rows, N_cols)
        if _backfilled > 0:
            print(f"  ⏱  Backfilled timing estimates for {_backfilled} tiles"
                  f" from file timestamps.")

    manifest = {
        "scene"       : name,
        "camera"      : cam_obj.name,
        "cam_type"    : cam.type,
        "method"      : "camera_shift",
        "total_W"     : total_W,
        "total_H"     : total_H,
        "tile_W"      : tile_W,
        "tile_H"      : tile_H,
        "cols"        : N_cols,
        "rows"        : N_rows,
        "format"      : file_format,
        "long_edge"   : max(total_W, total_H),
        "tiles_total" : total,
        "tiles_done"  : already_done,
        "tile_times"  : tile_times,
        "note"        : "row/col in filename = image coordinates (row 0 = top-left)",
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Center-outward tile order (main subject renders first) ─────────────────
    cx = (N_cols - 1) / 2.0
    cy = (N_rows - 1) / 2.0
    tile_order = sorted(
        ((row, col) for row in range(N_rows) for col in range(N_cols)),
        key=lambda t: (t[0] - cy) ** 2 + (t[1] - cx) ** 2,
    )

    # ── Resolve sensor_fit once (AUTO → HORIZONTAL or VERTICAL) ──────────────
    resolved_fit = resolve_sensor_fit(cam, r.resolution_x, r.resolution_y)
    print(f"  sensor_fit  : {cam.sensor_fit}  →  resolved as {resolved_fit}"
          f"  (res {r.resolution_x}×{r.resolution_y})")

    # ── Save original state ────────────────────────────────────────────────────
    state         = save_state(scene, cam_obj)
    errors         = []
    done           = 0
    render_times   = []
    session_start  = time.time()
    session_seconds = 0.0   # computed in finally so it's always saved

    # ── Settings snapshot: save on first session, check on resume ─────────────
    snapshot_path = os.path.join(abs_out, "render_snapshot.json")
    _cur_samples  = scene.cycles.samples if hasattr(scene, "cycles") else None
    if not preview_mode:
        if already_done == 0:
            _save_snapshot(snapshot_path, state, cam_obj,
                           total_W, total_H, tile_W, tile_H, _cur_samples)
            print(f"  Snapshot    : settings saved  →  render_snapshot.json")
        else:
            _mismatches = _check_snapshot(snapshot_path, state, cam_obj,
                                          total_W, total_H, tile_W, tile_H, _cur_samples)
            if _mismatches:
                print(f"\n{'⚠' * 32}")
                print(f"  SETTINGS MISMATCH — resuming with different settings than first session!")
                print(f"{'─' * 64}")
                for _field, _saved, _cur in _mismatches:
                    print(f"  {_field:<22}  was: {_saved!r:<30}  now: {_cur!r}")
                print(f"{'─' * 64}")
                print(f"  Tiles already on disk were rendered with the OLD settings.")
                print(f"  If intentional, ignore this warning and continue.")
                print(f"  Otherwise: stop (Ctrl+C) and delete the tile folder to start fresh.")
                print(f"{'⚠' * 32}\n")

    try:
        # ── GPU ───────────────────────────────────────────────────────────────
        if use_gpu:
            setup_gpu(scene, compute_device)

        # ── Preview: reduce samples for speed ────────────────────────────────
        if preview_mode and hasattr(scene, "cycles"):
            scene.cycles.samples = preview_samples

        # ── Render resolution = tile size, NOT the full image ──────────────────
        # This is the line that breaks the 65 536 ceiling:
        # we never set resolution_x/y to total_W/total_H.
        r.resolution_x          = tile_W
        r.resolution_y          = tile_H
        r.resolution_percentage = 100
        r.use_border            = False   # explicitly off — not used here
        r.use_crop_to_border    = False
        r.use_compositing       = True    # TEST — check if compositor causes black tiles
        r.use_sequencer         = False   # not needed, saves overhead
        r.dither_intensity      = 1.0     # match Blender UI behaviour — prevents gradient banding
        r.image_settings.file_format = file_format
        if file_format == "JPEG":
            r.image_settings.quality = jpeg_quality
        elif file_format == "PNG":
            r.image_settings.color_depth = '16'  # 16-bit PNG — eliminates gradient banding

        # ── Zoom camera so one tile fills the frame ────────────────────────────
        apply_zoom(cam, N_cols, N_rows, resolved_fit)

        # ── ORTHO: precompute tile world dimensions and original location ──────
        # After apply_zoom:
        #   HORIZONTAL → cam.ortho_scale == tile width  in world units
        #   VERTICAL   → cam.ortho_scale == tile height in world units
        # For PERSP these are unused (shift_x/shift_y are dimensionless).
        if cam.type == "ORTHO":
            if resolved_fit == "HORIZONTAL":
                tile_W_world = cam.ortho_scale
                tile_H_world = tile_W_world * (tile_H / tile_W)
            else:  # VERTICAL
                tile_H_world = cam.ortho_scale
                tile_W_world = tile_H_world * (tile_W / tile_H)
        else:
            tile_W_world = 0.0
            tile_H_world = 0.0
        orig_loc     = state["cam_location"]   # mathutils.Vector copy from save_state

        # ── Framing shift: bake original cam shifts into tile positioning ──────
        # Users often set shift_x/shift_y for framing.  For ORTHO this script
        # moves the camera via location offsets, so the shift must be converted
        # to a world-space vector and the shift parameters zeroed out — otherwise
        # the film-plane shift applies uniformly to every tile, offsetting the
        # whole assembled image by shift × total_W pixels.
        # For PERSP the per-tile shift_x/y values are in the same sensor-width
        # units as cam.shift_x/y, so original framing shifts add directly.
        orig_shift_x = state["shift_x"]
        orig_shift_y = state["shift_y"]
        framing_vec  = None

        if orig_shift_x != 0.0 or orig_shift_y != 0.0:
            print(f"  Framing shift : shift_x={orig_shift_x:+.4f}  shift_y={orig_shift_y:+.4f}"
                  f"  → baked into {'world-space location offset' if cam.type == 'ORTHO' else 'per-tile shift values'}")

        if cam.type == "ORTHO":
            # Zero out shifts — replaced by location offset below.
            cam.shift_x = 0.0
            cam.shift_y = 0.0
            if orig_shift_x != 0.0 or orig_shift_y != 0.0:
                import mathutils as _mu
                # original ortho_scale before zoom (= 1 sensor_width in world units)
                full_ortho_scale = (tile_W_world * N_cols if resolved_fit == "HORIZONTAL"
                                    else tile_H_world * N_rows)
                _mat   = cam_obj.matrix_world.to_3x3()
                _right = _mat.col[0].normalized()
                _up    = _mat.col[1].normalized()
                # Sign convention: shift_y = +0.150 shifts view UP → camera must move UP
                # → framing_vec = +shift_y × scale × up  (same sign as shift value)
                framing_vec = (
                    orig_shift_x * full_ortho_scale * _mu.Vector(_right)
                    + orig_shift_y * full_ortho_scale * _mu.Vector(_up)
                )

        # ── Render loop ───────────────────────────────────────────────────────
        for row, col in tile_order:
            # row 0 = TOP of image, col 0 = LEFT.
            filename = f"tile_r{row:04d}_c{col:04d}.{ext}"
            filepath = os.path.join(tile_dir, filename)

            done += 1
            if skip_existing and _tile_exists(tile_dir, row, col):
                print(f"  [{done:>4}/{total}] SKIP  {filename}")
                continue

            apply_shift(cam_obj, cam, N_cols, N_rows, col, row,
                        orig_loc, tile_W_world, tile_H_world,
                        framing_vec=framing_vec,
                        orig_shift_x=orig_shift_x, orig_shift_y=orig_shift_y,
                        tile_aspect_y=tile_H / tile_W,
                        resolved_fit=resolved_fit)
            r.filepath = filepath[:-(len(ext) + 1)]   # Blender appends the ext

            if cam.type == "PERSP":
                pos_info = f"shift_x={cam.shift_x:+.4f}  shift_y={cam.shift_y:+.4f}"
            else:
                pos_info = (f"loc=({cam_obj.location.x:.2f},"
                            f"{cam_obj.location.y:.2f},"
                            f"{cam_obj.location.z:.2f})")
            print(f"  [{done:>4}/{total}]  r{row:04d} c{col:04d}"
                  f"  {pos_info}"
                  f"  →  {filename}")

            try:
                tile_start   = time.time()
                bpy.ops.render.render(write_still=True)
                tile_elapsed = time.time() - tile_start

                if preview_mode and N_cols == 1:
                    # Single-tile preview: done after the first (only) tile.
                    print(f"\n{'═' * 64}")
                    print(f"  PREVIEW saved: {filepath}")
                    print(f"{'─' * 64}")
                    print(f"  Now do a Blender F12 render on camera '{cam_obj.name}'.")
                    print(f"  If the framing matches → set PREVIEW_GRID = 5 and")
                    print(f"  re-run to validate the shift formula, then set")
                    print(f"  PREVIEW_MODE = False for the full render.")
                    print(f"{'═' * 64}\n")
                    break   # only 1 tile needed

                render_times.append(tile_elapsed)
                manifest["tile_times"][filename] = round(tile_elapsed, 2)
                already_done += 1
                manifest["tiles_done"] = already_done
                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                avg       = sum(render_times) / len(render_times)
                remaining = total - already_done        # tiles still to render (skipped don't count)
                eta       = avg * remaining
                elapsed   = time.time() - session_start
                print(f"         ↳  tile {_fmt_hms(tile_elapsed)}"
                      f"  |  avg {_fmt_hms(avg)}"
                      f"  |  elapsed {_fmt_hms(elapsed)}"
                      f"  |  ETA {_fmt_hms(eta)}")

            except Exception as exc:
                print(f"         !! ERROR: {exc}")
                errors.append(filename)

        # ── Grid preview stitch ───────────────────────────────────────────
        if preview_mode and N_cols > 1 and not errors:
            preview_out = os.path.join(tile_dir, "grid_preview.png")
            print(f"\n  Stitching {N_cols}×{N_rows} preview tiles…")
            _stitch_grid_preview(tile_dir, N_cols, N_rows, tile_W, tile_H, ext,
                                 preview_out)
            print(f"\n{'═' * 64}")
            print(f"  GRID PREVIEW saved: {preview_out}")
            print(f"{'─' * 64}")
            print(f"  {N_cols}×{N_rows} tiles — scene should fill the full frame")
            print(f"  with no black bands at top/bottom or left/right.")
            print(f"  If correct → set PREVIEW_MODE = False for the full render.")
            print(f"{'═' * 64}\n")

    finally:
        # Runs on normal exit, Ctrl+C, stop button, or any exception.
        # The .blend file on disk is never written by this script, so even a
        # force-kill leaves the file untouched — this block is an extra layer.
        restore_state(scene, cam_obj, state)
        print("\n  Camera and render settings restored.")
        # Always save timing — even on interruption — so accumulated render time
        # is never lost across sessions (the 45h + 8h bug fix).
        session_seconds = time.time() - session_start
        if render_times and not preview_mode:
            session_avg = sum(render_times) / len(render_times)
            save_timing(timing_path, session_avg,
                        sum(manifest["tile_times"].values()),
                        tile_W, tile_H, compute_device, name)

    # ── Final manifest update ──────────────────────────────────────────────────
    if errors:
        manifest["errors"] = errors
    manifest["tiles_done"] = already_done
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Statistics file ────────────────────────────────────────────────────────
    if not preview_mode:
        total_render_seconds = sum(manifest["tile_times"].values())
        _write_stats(
            stats_path           = os.path.join(abs_out, "render_stats.txt"),
            name                 = name,
            cam_name             = cam_obj.name,
            cam_type             = cam.type,
            compute_device       = compute_device,
            total_W              = total_W,
            total_H              = total_H,
            tile_W               = tile_W,
            tile_H               = tile_H,
            N_cols               = N_cols,
            N_rows               = N_rows,
            render_times         = render_times,
            session_seconds      = session_seconds,
            total_render_seconds = total_render_seconds,
            tiles_done           = already_done,
            tiles_total          = total,
            errors               = errors,
            abs_out              = abs_out,
            ext                  = ext,
            completed_at         = datetime.datetime.now(datetime.timezone.utc).astimezone(),
            scene_stats          = scene_stats,
        )
        print(f"  stats  →  {os.path.join(abs_out, 'render_stats.txt')}")

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'═' * 64}")
    print(f"  {already_done}/{total} tiles rendered  |  {len(errors)} errors")
    if errors:
        print("  Failed tiles:", ", ".join(errors))
    print(f"  manifest →  {manifest_path}")
    if render_times:
        print(f"  avg tile →  {sum(render_times)/len(render_times):.1f} s")
    print(f"{'═' * 64}\n")

    return manifest


if __name__ == "__main__":
    render_tiles()
