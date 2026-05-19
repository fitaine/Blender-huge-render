@echo off
:: -----------------------------------------------------------------------------
:: SAMPLE TEST - runs sample_test.py on a .blend file
::
:: Renders multiple regions at multiple sample counts and produces a single
:: comparison image to help choose the right sample count before a full render.
:: Set SAMPLE_COUNTS, REGIONS and MAX_TIME_SECS in the CONFIG block of
:: sample_test.py before use.
::
:: HOW TO USE
::   Drag & drop your .blend file onto this .bat file
::   - OR -
::   Edit BLEND_FILE below to hardcode a path
:: -----------------------------------------------------------------------------

:: Path to your Blender executable - adjust if needed
set BLENDER="C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

:: Path to the render script
set SCRIPT="%~dp0sample_test.py"

:: Use drag-and-drop argument if provided, otherwise ask
if "%~1"=="" (
    set /p BLEND_FILE="Drag your .blend file here or type its path: "
) else (
    set BLEND_FILE="%~1"
)

echo.
echo Starting sample count comparison...
echo Blender : %BLENDER%
echo Scene   : %BLEND_FILE%
echo Script  : %SCRIPT%
echo.

%BLENDER% --background %BLEND_FILE% --python %SCRIPT%

echo.
echo Done. Press any key to close.
pause >nul
