@echo off
:: -----------------------------------------------------------------------------
:: NOISE TEST - runs noise_test.py on a .blend file
::
:: Renders regions at multiple Cycles noise thresholds (adaptive sampling)
:: and produces a single comparison image. Includes a seam-test row that
:: simulates adjacent tile boundaries to check for banding.
::
:: Set NOISE_THRESHOLDS, MIN_SAMPLES, MAX_SAMPLES, REGIONS, SEAM_TEST
:: in the CONFIG block of noise_test.py before use.
::
:: HOW TO USE
::   Drag & drop your .blend file onto this .bat file
::   - OR -
::   Edit BLEND_FILE below to hardcode a path
:: -----------------------------------------------------------------------------

:: Path to your Blender executable
set BLENDER="C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

:: Path to the render script
set SCRIPT="%~dp0noise_test.py"

:: Use drag-and-drop argument if provided, otherwise ask
if "%~1"=="" (
    set /p BLEND_FILE="Drag your .blend file here or type its path: "
) else (
    set BLEND_FILE="%~1"
)

echo.
echo Starting noise threshold comparison...
echo Blender : %BLENDER%
echo Scene   : %BLEND_FILE%
echo Script  : %SCRIPT%
echo.

%BLENDER% --background %BLEND_FILE% --python %SCRIPT%

echo.
echo Done. Press any key to close.
pause >nul
