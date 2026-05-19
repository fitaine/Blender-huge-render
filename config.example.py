# Copy this file to config.py and fill in your own paths.
# config.py is gitignored and will never be committed.

# ── stitch_render_tiles.py ────────────────────────────────────────────────────

# Folder produced by render_tiles_65k.py (must contain manifest.json)
RENDER_TILES_DIR  = r"C:\path\to\project\tiles\SceneName"

# Where to save the stitched image
OUTPUT_DIR        = r"C:\path\to\project"

# Output filename (without extension)
OUTPUT_NAME       = "SceneName"

# Output format: "JPEG", "TIFF", or "PNG"
FORMAT            = "JPEG"

# JPEG quality (1–95)
JPEG_QUALITY      = 92

# TIFF compression: "none", "lzw", "zip"
TIFF_COMPRESSION  = "lzw"


# ── stitch_to_dzi.py ──────────────────────────────────────────────────────────

# Render-tiles folder (must contain manifest.json)
SOURCE            = r"C:\path\to\project\tiles\SceneName"

# Website base directory (where index.html and images.js live)
WEBSITE_DIR       = r"C:\path\to\website"

# Series folder name inside tiles/ and images.js
SERIES            = "LIDAR"

# Name of the image in the viewer
IMAGE_NAME        = "SceneName"

# DZI parameters
DZI_TILE_SIZE     = 256
DZI_OVERLAP       = 1
DZI_FORMAT        = "jpeg"
DZI_QUALITY       = 100

# Levels below max to switch from render-tile cutting to full-image stitching
STITCH_START_LEVEL_OFFSET = 1

# Vertical crop in pixels (0 = no crop)
CROP_TOP          = 0
CROP_BOTTOM       = 0
