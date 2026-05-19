"""
stitch_to_dzi.py
================
Convert a render to a DZI pyramid for the web viewer.

Two input modes — set SOURCE to either:

  A) A render-tiles folder (output of render_tiles_65k.py):
       SOURCE = r"C:\...\tiles\MyScene"
     Must contain manifest.json + tile_r****_c****.jpg files.
     Memory-efficient: max level is cut tile-by-tile, never loads the full image.

  B) A single stitched image file (.jpg, .tif, .png …):
       SOURCE = r"C:\...\MyScene.jpg"
     Use this after editing the stitched image (fog, glare, etc.) and re-exporting
OR  single image file  (see modes A/     as a single file. The full image is loaded into memory — needs ~12 GB for 65k.

Workflow after Blender edits:
  Option 1 — re-tile in Blender (recommended for 65k):
    render_tiles_65k.py  →  stitch_to_dzi.py  (mode A, new folder)

  Option 2 — export as single file from Blender:
    stitch_render_tiles.py  →  edit  →  export  →  stitch_to_dzi.py  (mode B)

After it finishes, refresh the viewer — images.js is updated automatically.
"""

import os, json, math, time, bisect
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Edit config.py (gitignored) — copy config.example.py if it doesn't exist yet.
try:
    from config import (SOURCE, WEBSITE_DIR, SERIES, IMAGE_NAME,
                        DZI_TILE_SIZE, DZI_OVERLAP, DZI_FORMAT, DZI_QUALITY,
                        STITCH_START_LEVEL_OFFSET, CROP_TOP, CROP_BOTTOM)
except ImportError:
    raise SystemExit("config.py not found — copy config.example.py to config.py and fill in your paths.")
# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── Source detection ──────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".exr"}

def detect_source_mode(source):
    """Return 'tiles' or 'image'."""
    if os.path.isfile(source):
        ext = os.path.splitext(source)[1].lower()
        if ext in IMAGE_EXTS:
            return "image"
        raise ValueError(f"Unrecognised file type: {ext}")
    if os.path.isdir(source):
        manifest = os.path.join(source, "manifest.json")
        if os.path.exists(manifest):
            return "tiles"
        raise FileNotFoundError(f"No manifest.json found in {source}")
    raise FileNotFoundError(f"SOURCE not found: {source}")


def load_manifest(tiles_dir):
    path = os.path.join(tiles_dir, "manifest.json")
    with open(path) as f:
        return json.load(f)


def render_tile_path(tiles_dir, row, col, fmt=None):
    """Return the path to a render tile, trying PNG first then JPEG.
    Supports mixed-format folders (e.g. PNG sky tiles + JPEG body tiles)."""
    base = os.path.join(tiles_dir, f"tile_r{row:04d}_c{col:04d}")
    for ext in ("png", "jpg", "jpeg"):
        p = base + "." + ext
        if os.path.exists(p):
            return p
    # Fall back to format-derived name (tile may not exist yet)
    if fmt:
        ext = "jpg" if fmt.upper() == "JPEG" else fmt.lower()
    else:
        ext = "jpg"
    return base + "." + ext


class RenderTileGrid:
    """
    Lazy-loading cache for render tiles.
    Caches the last N tiles to avoid re-reading when adjacent DZI tiles share a render tile.
    crop_top / crop_bottom trim the assembled image vertically (pixels in full-scene space).
    After cropping, self.H is the effective output height; self._full_H is the original.
    """
    def __init__(self, tiles_dir, manifest, cache_size=4, crop_top=0, crop_bottom=0):
        self.tiles_dir = tiles_dir
        self.W        = manifest["total_W"]
        self._full_H  = manifest["total_H"]
        self.tw       = manifest["tile_W"]
        self.th       = manifest["tile_H"]
        self.cols     = manifest["cols"]
        self.rows     = manifest["rows"]
        self.fmt      = manifest["format"]
        self._cache      = {}   # (r,c) → PIL Image
        self._order      = []   # LRU order
        self._cache_size = cache_size
        self._crop_top    = crop_top
        self._crop_bottom = crop_bottom
        self.H = self._full_H - crop_top - crop_bottom  # effective (cropped) height

        self._method = manifest.get("method", "render_border")

        if self._method == "camera_shift":
            # All tiles are the same pixel size (tile_W × tile_H), but each tile
            # only covers  total_W/cols × total_H/rows  scene pixels (slightly
            # less than tile_W × tile_H due to oversampling).  Positioning tiles
            # by their pixel dimensions would accumulate a growing offset across
            # columns/rows, breaking any feature that crosses tile boundaries
            # (diagonal lines, cables, ridgelines).
            # Use the true scene coverage per tile instead.
            # _scene_th uses _full_H (not cropped H) so tile positions are always
            # in full-scene space; cropping is applied in pixel_region/stitch.
            self._scene_tw = self.W / self.cols   # e.g. 4000.0 for 120 000 / 30
            self._scene_th = self._full_H / self.rows
            self._x_starts = [round(c * self._scene_tw) for c in range(self.cols)]
            self._y_starts = [round(r * self._scene_th) for r in range(self.rows)]
        else:
            # render_border (render_tiles_65k.py): Blender clips edge tiles to
            # their exact pixel dimensions, so tile pixel size == scene coverage.
            # Pre-compute y-starts from actual tile headers so the partial first
            # row (top of image, shorter when total_H % tile_H != 0) is placed
            # correctly.
            y = 0
            self._y_starts = []
            for rt_r in range(self.rows):
                self._y_starts.append(y)
                probe = Image.open(render_tile_path(self.tiles_dir, rt_r, 0))
                y += probe.height
                probe.close()
            self._x_starts = [c * self.tw for c in range(self.cols)]
            self._scene_tw = self.tw
            self._scene_th = self.th

    def get(self, rt_row, rt_col):
        key = (rt_row, rt_col)
        if key in self._cache:
            self._order.remove(key)
            self._order.append(key)
            return self._cache[key]
        path = render_tile_path(self.tiles_dir, rt_row, rt_col)
        if not os.path.exists(path):
            # Tile not rendered yet — return solid black placeholder
            img = Image.new("RGB", (self.tw, self.th), (0, 0, 0))
        else:
            img = Image.open(path)
            img.load()
        self._cache[key] = img
        self._order.append(key)
        if len(self._order) > self._cache_size:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return img

    def pixel_region(self, x0, y0, x1, y1):
        """
        Return a PIL Image containing pixels [x0:x1, y0:y1] from the assembled image.
        y0/y1 are in CROPPED image space (0 = top of crop region).
        Internally offsets y by _crop_top to look up tiles in full-scene coordinates.
        """
        # Convert cropped image coords → full-scene coords for tile lookups
        y0 = y0 + self._crop_top
        y1 = y1 + self._crop_top
        x0 = max(0, x0); y0 = max(0, y0)
        x1 = min(self.W, x1); y1 = min(self._full_H, y1)
        out = Image.new("RGB", (x1 - x0, y1 - y0))
        # Which render tiles overlap this region? Use full scene-space starts.
        rt_c0 = max(0, bisect.bisect_right(self._x_starts, x0) - 1)
        rt_c1 = min(bisect.bisect_right(self._x_starts, x1 - 1) - 1, self.cols - 1)
        rt_r0 = max(0, bisect.bisect_right(self._y_starts, y0) - 1)
        rt_r1 = min(bisect.bisect_right(self._y_starts, y1 - 1) - 1, self.rows - 1)
        for rt_r in range(rt_r0, rt_r1 + 1):
            for rt_c in range(rt_c0, rt_c1 + 1):
                tile = self.get(rt_r, rt_c)
                # Scene-space extent of this render tile (full-scene coords)
                tx0s = self._x_starts[rt_c]
                ty0s = self._y_starts[rt_r]
                tx1s = self._x_starts[rt_c + 1] if rt_c + 1 < self.cols else self.W
                ty1s = self._y_starts[rt_r + 1] if rt_r + 1 < self.rows else self._full_H
                # Overlap with the requested region (full-scene space)
                ox0 = max(x0, tx0s); oy0 = max(y0, ty0s)
                ox1 = min(x1, tx1s); oy1 = min(y1, ty1s)
                if ox1 <= ox0 or oy1 <= oy0:
                    continue
                if self._method == "camera_shift":
                    # Tile pixels oversample the scene: map scene coords → tile pixels
                    sx = tile.width  / (tx1s - tx0s)
                    sy = tile.height / (ty1s - ty0s)
                    lx0 = round((ox0 - tx0s) * sx); ly0 = round((oy0 - ty0s) * sy)
                    lx1 = round((ox1 - tx0s) * sx); ly1 = round((oy1 - ty0s) * sy)
                    crop = tile.crop((lx0, ly0, lx1, ly1))
                    dw, dh = ox1 - ox0, oy1 - oy0
                    if crop.size != (dw, dh):
                        crop = crop.resize((dw, dh), Image.LANCZOS)
                else:
                    # render_border: tile pixels == scene pixels
                    crop = tile.crop((ox0 - tx0s, oy0 - ty0s, ox1 - tx0s, oy1 - ty0s))
                out.paste(crop, (ox0 - x0, oy0 - y0))
        return out

    def close(self):
        for img in self._cache.values():
            img.close()
        self._cache.clear()
        self._order.clear()


def ensure_rgb(img):
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (0, 0, 0))
        bg.paste(img, mask=img.split()[3])
        return bg
    return img.convert("RGB") if img.mode != "RGB" else img


def generate_dzi_level_from_grid(grid, level, max_level, level_dir):
    """
    Generate DZI tiles for the given level by reading from the render tile grid.
    Used for max_level only (full resolution).
    """
    scale = 2 ** (level - max_level)          # = 1.0 at max_level
    lw = max(1, math.ceil(grid.W * scale))
    lh = max(1, math.ceil(grid.H * scale))
    # (scale == 1 here, but kept general for clarity)

    os.makedirs(level_dir, exist_ok=True)
    cols = math.ceil(lw / DZI_TILE_SIZE)
    rows = math.ceil(lh / DZI_TILE_SIZE)

    t0 = time.time()
    total = cols * rows
    done = 0

    for row in range(rows):
        for col in range(cols):
            x0 = col * DZI_TILE_SIZE - (DZI_OVERLAP if col > 0 else 0)
            y0 = row * DZI_TILE_SIZE - (DZI_OVERLAP if row > 0 else 0)
            x1 = min(x0 + DZI_TILE_SIZE + DZI_OVERLAP * 2, lw)
            y1 = min(y0 + DZI_TILE_SIZE + DZI_OVERLAP * 2, lh)
            tile = grid.pixel_region(x0, y0, x1, y1)
            tile = ensure_rgb(tile)
            out_path = os.path.join(level_dir, f"{col}_{row}.{DZI_FORMAT}")
            os.makedirs(level_dir, exist_ok=True)
            tile.save(out_path, quality=DZI_QUALITY, optimize=True)
            done += 1
            if done % 500 == 0 or done == total:
                elapsed = time.time() - t0
                pct = done / total * 100
                eta = elapsed / done * (total - done)
                print(f"      {done}/{total} tiles  {pct:.0f}%  elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m")

    print(f"    level {level:2d}: {lw:6d}×{lh:5d}  ({cols}×{rows}) — done in {(time.time()-t0)/60:.1f} min")


def generate_dzi_level_from_image(img, level, max_level, level_dir):
    """
    Generate DZI tiles for the given level from a PIL Image (already at this level's size).
    """
    scale = 2 ** (level - max_level)
    lw = max(1, math.ceil(img.width))
    lh = max(1, math.ceil(img.height))
    os.makedirs(level_dir, exist_ok=True)
    cols = math.ceil(lw / DZI_TILE_SIZE)
    rows = math.ceil(lh / DZI_TILE_SIZE)

    for row in range(rows):
        for col in range(cols):
            x0 = col * DZI_TILE_SIZE - (DZI_OVERLAP if col > 0 else 0)
            y0 = row * DZI_TILE_SIZE - (DZI_OVERLAP if row > 0 else 0)
            x1 = min(x0 + DZI_TILE_SIZE + DZI_OVERLAP * 2, lw)
            y1 = min(y0 + DZI_TILE_SIZE + DZI_OVERLAP * 2, lh)
            tile = img.crop((x0, y0, x1, y1))
            tile = ensure_rgb(tile)
            out_path = os.path.join(level_dir, f"{col}_{row}.{DZI_FORMAT}")
            os.makedirs(level_dir, exist_ok=True)
            tile.save(out_path, quality=DZI_QUALITY, optimize=True)

    print(f"    level {level:2d}: {lw:6d}×{lh:5d}  ({cols}×{rows})")


def stitch_at_scale(grid, scale):
    """
    Stitch all render tiles into a PIL Image at the given scale factor (< 1).
    Each render tile is individually downsampled before stitching → low peak RAM.
    Respects grid._crop_top / _crop_bottom: tiles outside the crop are skipped,
    tiles that straddle a crop boundary are sliced before resizing.
    """
    lw = max(1, math.ceil(grid.W * scale))
    lh = max(1, math.ceil(grid.H * scale))  # grid.H is already the cropped height
    out = Image.new("RGB", (lw, lh))

    crop_end = grid._full_H - grid._crop_bottom  # bottom of crop in full-scene coords

    for rt_r in range(grid.rows):
        for rt_c in range(grid.cols):
            # Full-scene extent of this render tile
            tx0s = grid._x_starts[rt_c]
            ty0s = grid._y_starts[rt_r]
            tx1s = grid._x_starts[rt_c + 1] if rt_c + 1 < grid.cols else grid.W
            ty1s = grid._y_starts[rt_r + 1] if rt_r + 1 < grid.rows else grid._full_H
            # Clip to crop region; skip tiles fully outside
            ety0s = max(ty0s, grid._crop_top)
            ety1s = min(ty1s, crop_end)
            if ety1s <= ety0s:
                continue
            tile = grid.get(rt_r, rt_c)
            tile = ensure_rgb(tile)
            # Map full-scene clip extent → tile pixel rows
            tile_sy = tile.height / (ty1s - ty0s) if grid._method == "camera_shift" else 1.0
            py0 = round((ety0s - ty0s) * tile_sy)
            py1 = round((ety1s - ty0s) * tile_sy)
            strip = tile.crop((0, py0, tile.width, py1)) if (py0 > 0 or py1 < tile.height) else tile
            # Destination in the cropped output (y relative to crop_top, then scaled)
            sx0 = round(tx0s * scale)
            sy0 = round((ety0s - grid._crop_top) * scale)
            sw  = max(1, min(round(tx1s  * scale) - sx0, lw - sx0))
            sh  = max(1, min(round((ety1s - grid._crop_top) * scale) - sy0, lh - sy0))
            small = strip.resize((sw, sh), Image.LANCZOS)
            out.paste(small, (sx0, sy0))
            small.close()
            if strip is not tile:
                strip.close()
        print(f"      stitched row {rt_r+1}/{grid.rows} at scale {scale:.4f}")

    return out


def write_dzi_file(dzi_path, W, H):
    with open(dzi_path, "w") as f:
        f.write(f"""<?xml version="1.0" encoding="UTF-8"?>
<Image xmlns="http://schemas.microsoft.com/deepzoom/2008"
       Format="{DZI_FORMAT}" Overlap="{DZI_OVERLAP}" TileSize="{DZI_TILE_SIZE}">
    <Size Width="{W}" Height="{H}"/>
</Image>""")


def update_images_js(website_dir, series, image_name, W, H, dzi_rel_path):
    """
    Add/update the entry for image_name in images.js.
    If the image already exists, updates its dimensions.
    If not, appends it to the correct series block.
    """
    images_js_path = os.path.join(website_dir, "images.js")
    with open(images_js_path, encoding="utf-8") as f:
        content = f.read()

    new_entry = (
        f'      {{ name: "{image_name}", '
        f'width: {W}, height: {H}, '
        f'caption: "", '
        f'dzi: "{dzi_rel_path}" }},'
    )

    # Check if image already present — update width/height if so
    if f'name: "{image_name}"' in content:
        import re
        updated = re.sub(
            rf'(name:\s*"{re.escape(image_name)}",\s*width:\s*)\d+(\s*,\s*height:\s*)\d+',
            rf'\g<1>{W}\2{H}',
            content,
        )
        if updated == content:
            print(f"  images.js: entry for '{image_name}' unchanged.")
        else:
            with open(images_js_path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"  images.js: updated '{image_name}' → {W}×{H}")
        return

    # Find insertion point: last image in the correct series block
    # Look for the series id line and the closing ']' of its images array
    import re
    # Find the series block by its id
    series_pattern = re.compile(
        rf'(id:\s*"{re.escape(series)}".*?images:\s*\[)(.*?)(\s*\])',
        re.DOTALL
    )
    match = series_pattern.search(content)
    if not match:
        print(f"  ⚠ Series '{series}' not found in images.js — entry NOT added.")
        print(f"    Add manually:\n      {new_entry}")
        return

    # Insert before the closing ] of the images array
    before = content[:match.start(3)]
    after  = content[match.start(3):]
    content = before + "\n" + new_entry + "\n    " + after

    with open(images_js_path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"  images.js: added '{image_name}' to series '{series}'")


def main():
    t_start = time.time()

    # ── Detect source mode ────────────────────────────────────────────────────
    mode = detect_source_mode(SOURCE)
    print(f"Source mode: {'render-tiles folder' if mode == 'tiles' else 'single image file'}")
    print(f"  {SOURCE}\n")

    # ── Output paths ──────────────────────────────────────────────────────────
    out_base  = os.path.join(WEBSITE_DIR, "tiles", SERIES)
    dzi_dir   = os.path.join(out_base, IMAGE_NAME)
    fils_dir  = os.path.join(dzi_dir, IMAGE_NAME + "_files")
    dzi_path  = os.path.join(dzi_dir, IMAGE_NAME + ".dzi")
    dzi_rel   = f"tiles/{SERIES}/{IMAGE_NAME}/{IMAGE_NAME}.dzi"

    os.makedirs(fils_dir, exist_ok=True)

    # ── MODE A: render-tiles folder ───────────────────────────────────────────
    if mode == "tiles":
        manifest   = load_manifest(SOURCE)
        W          = manifest["total_W"]
        H          = manifest["total_H"]
        tiles_done = manifest["tiles_done"]
        tiles_total= manifest["tiles_total"]
        print(f"Render tile manifest: {W}×{H} px, {tiles_done}/{tiles_total} tiles done")
        if tiles_done < tiles_total:
            print(f"  ⚠ Only {tiles_done}/{tiles_total} tiles rendered — output will have black patches!")
            input("  Press Enter to continue anyway, or Ctrl+C to abort.")

        grid = RenderTileGrid(SOURCE, manifest, crop_top=CROP_TOP, crop_bottom=CROP_BOTTOM)
        H = grid.H   # effective height after crop (= manifest H when crop is 0)
        if CROP_TOP > 0 or CROP_BOTTOM > 0:
            print(f"  Crop applied: top {CROP_TOP} px, bottom {CROP_BOTTOM} px  →  {W} × {H} px")

        write_dzi_file(dzi_path, W, H)
        max_level = math.ceil(math.log2(max(W, H)))
        print(f"\nDZI levels: {max_level + 1}  (max level {max_level} = {2**max_level} px grid)\n")

        print(f"  Level {max_level} (full res — reading render tiles directly)…")
        level_dir = os.path.join(fils_dir, str(max_level))
        generate_dzi_level_from_grid(grid, max_level, max_level, level_dir)

        stitch_level = max_level - STITCH_START_LEVEL_OFFSET
        stitch_scale = 2 ** (stitch_level - max_level)
        stitch_lw    = max(1, math.ceil(W * stitch_scale))
        stitch_lh    = max(1, math.ceil(H * stitch_scale))
        print(f"\n  Level {stitch_level} (stitching at 1/{2**STITCH_START_LEVEL_OFFSET} scale → {stitch_lw}×{stitch_lh})…")
        current_img = stitch_at_scale(grid, stitch_scale)
        grid.close()
        level_dir = os.path.join(fils_dir, str(stitch_level))
        generate_dzi_level_from_image(current_img, stitch_level, max_level, level_dir)

        for level in range(stitch_level - 1, -1, -1):
            scale = 2 ** (level - max_level)
            lw    = max(1, math.ceil(W * scale))
            lh    = max(1, math.ceil(H * scale))
            print(f"\n  Level {level}…")
            next_img = current_img.resize((lw, lh), Image.LANCZOS)
            current_img.close()
            current_img = next_img
            level_dir = os.path.join(fils_dir, str(level))
            generate_dzi_level_from_image(current_img, level, max_level, level_dir)

        current_img.close()

    # ── MODE B: single image file ─────────────────────────────────────────────
    else:
        print(f"Loading image… (may take a moment for large files)")
        img = Image.open(SOURCE)
        img.load()
        if img.mode == "RGBA":
            bg = Image.new("RGB", img.size, (0, 0, 0))
            bg.paste(img, mask=img.split()[3])
            img.close()
            img = bg
        elif img.mode != "RGB":
            img = img.convert("RGB")

        W, H = img.size
        print(f"  {W}×{H} px")

        write_dzi_file(dzi_path, W, H)
        max_level = math.ceil(math.log2(max(W, H)))
        print(f"\nDZI levels: {max_level + 1}  (max level {max_level} = {2**max_level} px grid)\n")

        # Max level from the loaded image
        level_dir = os.path.join(fils_dir, str(max_level))
        print(f"  Level {max_level} (full res)…")
        generate_dzi_level_from_image(img, max_level, max_level, level_dir)

        # All lower levels by halving
        current_img = img
        for level in range(max_level - 1, -1, -1):
            scale = 2 ** (level - max_level)
            lw    = max(1, math.ceil(W * scale))
            lh    = max(1, math.ceil(H * scale))
            print(f"\n  Level {level}…")
            next_img = current_img.resize((lw, lh), Image.LANCZOS)
            current_img.close()
            current_img = next_img
            level_dir = os.path.join(fils_dir, str(level))
            generate_dzi_level_from_image(current_img, level, max_level, level_dir)

        current_img.close()

    # ── Update images.js ──────────────────────────────────────────────────────
    print(f"\nUpdating images.js…")
    update_images_js(WEBSITE_DIR, SERIES.lower(), IMAGE_NAME, W, H, dzi_rel)

    elapsed = time.time() - t_start
    print(f"\n✓ Done in {elapsed/60:.1f} min")
    print(f"  DZI at:  {dzi_dir}")
    print(f"  Refresh the viewer to see the new image.")


if __name__ == "__main__":
    main()
