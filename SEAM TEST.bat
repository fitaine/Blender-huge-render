@echo off
:: -----------------------------------------------------------------------------
:: SEAM TEST - runs seam_test.py on a .blend file
::
:: For each noise threshold, renders two adjacent sub-panels independently
:: (exactly like two neighbouring tiles) and stitches them cleanly.
:: Seamless stitch = threshold is safe. Visible step = too loose.
::
:: Set NOISE_THRESHOLDS, SEAM_FX, SEAM_FY in the CONFIG block of
:: seam_test.py before use.
::
:: HOW TO USE
::   Drag & drop your .blend file onto this .bat file
::   - OR -
::   Edit BLEND_FILE below to hardcode a path
:: -----------------------------------------------------------------------------

set BLENDER="C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"
set SCRIPT="%~dp0seam_test.py"

if "%~1"=="" (
    set /p BLEND_FILE="Drag your .blend file here or type its path: "
) else (
    set BLEND_FILE="%~1"
)

echo.
echo Starting seam banding test...
echo Blender : %BLENDER%
echo Scene   : %BLEND_FILE%
echo Script  : %SCRIPT%
echo.

%BLENDER% --background %BLEND_FILE% --python %SCRIPT%

echo.
echo Done. Press any key to close.
pause >nul
