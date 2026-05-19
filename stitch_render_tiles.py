"""
stitch_render_tiles.py
======================
Assembles render tiles (from render_tiles_65k.py) into one large image for editing.

Typical workflow:
  render_tiles_65k.py  →  stitch_render_tiles.py  →  edit in Blender / PS
                       →  re-render tiles (render_tiles_65k.py again)
                       →  stitch_to_dzi.py  →  web viewer

⚠ Memory: a 65 000×63 000 RGB image needs ~12 GB RAM.
   The canvas is allocated all at once. Make sure you have enough free RAM.

Supported output formats:
  JPEG  — lossy, ~0.5–2 GB. Fine for editing if the source tiles are already JPEG.
  TIFF  — lossless. Huge (~12 GB uncompressed, or ~3–6 GB with compression).
  PNG   — lossless, slow encoder. Practical up to ~20k px; very slow at 65k.

Usage: edit the CONFIG block below, then:
  python stitch_render_tiles.py
"""

import os, json, time
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


# ── CONFIG ────────────────────────────────────────────────────────────────────
# Edit config.py (gitignored) — copy config.example.py if it doesn't exist yet.
try:
    from config import RENDER_TILES_DIR, OUTPUT_DIR, OUTPUT_NAME, FORMAT, JPEG_QUALITY, TIFF_COMPRESSION
except ImportError:
    raise SystemExit("config.py not found — copy config.example.py to config.py and fill in your paths.")

# Overwrite existing output file?
OVERWRITE = False

# ── END CONFIG ────────────────────────────────────────────────────────────────


EXT_MAP = {"JPEG": ".jpg", "TIFF": ".tif", "PNG": ".png"}


def load_manifest(tiles_dir):
    path = os.path.join(tiles_dir, "manifest.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"manifest.json not found in {tiles_dir}")
    with open(path) as f:
        return json.load(f)


def render_tile_path(tiles_dir, row, col, fmt):
    ext = "jpg" if fmt.upper() == "JPEG" else fmt.lower()
    return os.path.join(tiles_dir, f"tile_r{row:04d}_c{col:04d}.{ext}")


def main():
    t0 = time.time()

    # ── Load manifest ─────────────────────────────────────────────────────────
    manifest = load_manifest(RENDER_TILES_DIR)
    W         = manifest["total_W"]
    H         = manifest["total_H"]
    tile_w    = manifest["tile_W"]
    tile_h    = manifest["tile_H"]
    cols      = manifest["cols"]
    rows      = manifest["rows"]
    fmt       = manifest["format"]
    done      = manifest["tiles_done"]
    total_t   = manifest["tiles_total"]

    print(f"Manifest: {W}×{H} px  |  {cols}×{rows} tiles  |  {done}/{total_t} rendered")

    if done < total_t:
        print(f"  ⚠ Only {done}/{total_t} tiles rendered — output will have blank patches!")
        input("  Press Enter to continue anyway, or Ctrl+C to abort.")

    # ── Output path ───────────────────────────────────────────────────────────
    ext = EXT_MAP.get(FORMAT.upper(), ".jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, OUTPUT_NAME + ext)

    if os.path.exists(out_path) and not OVERWRITE:
        print(f"\n✓ Output already exists: {out_path}")
        print("  Set OVERWRITE = True to replace it.")
        return

    # ── Memory estimate ───────────────────────────────────────────────────────
    ram_gb = W * H * 3 / 1024**3
    print(f"\nCanvas size: {W}×{H} px  →  ~{ram_gb:.1f} GB RAM needed")
    if ram_gb > 8:
        print("  ⚠ Large allocation — ensure you have enough free RAM.")
    print(f"Output: {out_path}\n")

    # ── Allocate canvas ───────────────────────────────────────────────────────
    print("Allocating canvas…")
    canvas = Image.new("RGB", (W, H), color=(0, 0, 0))

    # ── Paste tiles ───────────────────────────────────────────────────────────
    print("Stitching tiles…")
    pasted = 0
    for r in range(rows):
        for c in range(cols):
            path = render_tile_path(RENDER_TILES_DIR, r, c, fmt)
            if not os.path.exists(path):
                print(f"  ⚠ Missing: tile_r{r:04d}_c{c:04d} — skipped")
                pasted += 1
                continue
            tile = Image.open(path)
            tile.load()
            x = c * tile_w
            y = r * tile_h
            if tile.mode != "RGB":
                tile = tile.convert("RGB")
            canvas.paste(tile, (x, y))
            tile.close()
            pasted += 1

        # Progress per row
        elapsed = time.time() - t0
        pct     = pasted / (rows * cols) * 100
        eta     = elapsed / pasted * (rows * cols - pasted) if pasted else 0
        print(f"  row {r+1:2d}/{rows}  —  {pct:.0f}%  elapsed {elapsed:.0f}s  ETA {eta:.0f}s")

    # ── Save ──────────────────────────────────────────────────────────────────
    print(f"\nSaving {FORMAT} …  (this may take a minute)")
    t_save = time.time()

    save_kwargs = {}
    if FORMAT.upper() == "JPEG":
        save_kwargs = {"quality": JPEG_QUALITY, "optimize": True, "subsampling": 0}
    elif FORMAT.upper() == "TIFF":
        save_kwargs = {"compression": TIFF_COMPRESSION}

    canvas.save(out_path, **save_kwargs)
    canvas.close()

    size_mb = os.path.getsize(out_path) / 1024**2
    print(f"  saved in {time.time()-t_save:.0f}s  —  {size_mb:.0f} MB")
    print(f"\n✓ Done in {(time.time()-t0)/60:.1f} min")
    print(f"  {out_path}")


if __name__ == "__main__":
    main()
