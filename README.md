# Blender Huge Render

A tile-based rendering pipeline for Blender, built to produce images far beyond what a GPU can hold in memory.

---

## Why this exists

I'm Tiphaine Buccino, a French artist and photographer. Since 2023 I've been working with open LiDAR data published by the IGN (French national geographic institute) — elevation scans of the French Alps, captured by aircraft, accurate to a few centimetres. I process these point clouds in Blender, turning millions of raw survey points into large-format renders: mountain faces, glaciers, alpine valleys, rendered as fields of coloured spheres suspended in black space.

The images I want to make are large. Not large as in "high resolution" — large as in **96 000 × 120 000 pixels**, 11.5 gigapixels, to be explored as deep zoom images on the web. Blender cannot render an image that size in one pass. No GPU has enough VRAM to hold it. So I wrote this pipeline.

The idea is simple: split the camera frustum into a grid of tiles, render each tile at normal resolution, and reassemble. Each tile is a valid render on its own. The full image only exists once they're all stitched together.

---

## Pipeline overview

```
render_tiles_unlimited.py   →   stitch_to_dzi.py   →   web viewer
        (Blender)                  (Python)           (OpenSeadragon)
```

1. **`render_tiles_unlimited.py`** — Blender script. Divides the camera into an N×N grid, renders each tile using camera shift (no resolution cap, no quality loss), saves tiles to disk. Supports resume across sessions: already-rendered tiles are skipped. Accumulates render time per tile in the manifest so the total is never lost if the session is interrupted.

2. **`stitch_render_tiles.py`** — Assembles all tiles into a single large image for editing (JPEG, TIFF, or PNG). Useful for reviewing the full image before publishing.

3. **`stitch_to_dzi.py`** — Converts the tile grid directly into a [Deep Zoom Image](https://en.wikipedia.org/wiki/Deep_Zoom) pyramid (DZI), without assembling the full image in memory first. The resulting tiles feed the web viewer.

---

## Typical render numbers

| Scene | Resolution | Grid | Tiles | GPU time |
|---|---|---|---|---|
| Aiguille du Midi | 96 000 × 120 000 px | 30 × 30 | 900 | ~80 h |
| Barre des Écrins | 120 000 × 120 000 px | 30 × 30 | 900 | — |
| Avoriaz | 120 000 × 60 000 px | 30 × 30 | 900 | — |

All renders use [OPTIX](https://developer.nvidia.com/optix) (NVIDIA RTX), 1500 samples, LiDAR point clouds rendered as geometry nodes spheres.

---

## Setup

Copy `config.example.py` to `config.py` and fill in your paths. `config.py` is gitignored and will never be committed.

```bash
cp config.example.py config.py
# then edit config.py
```

`render_tiles_unlimited.py` and `render_tiles_65k.py` run inside Blender's scripting editor — open the `.blend` file, paste or load the script, set the CONFIG block, and run.

The other scripts run from a normal terminal with Python 3 and [Pillow](https://python-pillow.org/):

```bash
pip install Pillow
python stitch_to_dzi.py
```

---

## Files

| File | Role |
|---|---|
| `render_tiles_unlimited.py` | Main render script — no resolution cap, camera-shift method |
| `render_tiles_65k.py` | Older script — fixed 65 536 px long edge |
| `stitch_render_tiles.py` | Assemble tiles into one image for editing |
| `stitch_to_dzi.py` | Convert tile grid to DZI pyramid for the web viewer |
| `sample_test.py` | Preview a small region before committing to a full render |
| `noise_test.py` | Compare sample counts on a single tile |
| `seam_test.py` | Check tile borders for visible seams |
| `config.example.py` | Configuration template — copy to `config.py` |

---

## The web viewer

The rendered DZI tiles feed a custom deep zoom viewer built on [OpenSeadragon](https://openseadragon.github.io/), available at [lidar.tiphainebuccino.com](https://lidar.tiphainebuccino.com). It displays all my LiDAR series as nested island layouts — you can zoom from a full overview down to individual pixels of a 120-megapixel render.

---

## License

Scripts are MIT licensed. The LiDAR renders and photography are © Tiphaine Buccino — all rights reserved.
