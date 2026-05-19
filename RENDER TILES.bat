@echo off
:: ─────────────────────────────────────────────────────────────────────────────
:: RENDER TILES — runs render_tiles_65k.py on a .blend file via CLI
:: Blender renders in background (no UI freeze), progress shown in this window.
::
:: HOW TO USE
::   Drag & drop your .blend file onto this .bat file
::   — OR —
::   Edit BLEND_FILE below to hardcode a path
:: ─────────────────────────────────────────────────────────────────────────────

:: Path to your Blender executable — adjust if needed
set BLENDER="C:\Program Files\Blender Foundation\Blender 5.0\blender.exe"

:: Path to the render script
set SCRIPT="%~dp0render_tiles_65k.py"

:: Use drag-and-drop argument if provided, otherwise ask
if "%~1"=="" (
    set /p BLEND_FILE="Drag your .blend file here or type its path: "
) else (
    set BLEND_FILE="%~1"
)

echo.
echo Starting tiled render...
echo Blender : %BLENDER%
echo Scene   : %BLEND_FILE%
echo Script  : %SCRIPT%
echo.

%BLENDER% --background %BLEND_FILE% --python %SCRIPT%

echo.
echo Done. Press any key to close.
pause >nul
