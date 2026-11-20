@echo off
title Push AGRIAGENT to GitHub (Replace Old with Clean Code)
echo ======================================================================
echo   AGRIAGENT - ONE-CLICK GITHUB SYNC (CLEAN UPLOAD)
echo   Target: https://github.com/SaiiieSJaswanth/AGRIAGENT
echo ======================================================================
echo.

cd /d "%~dp0"
if not exist "src" (
    cd /d "D:\AIML_CASESTUDY\AGRIAGENT_FINAL_COMPLETE\AGRIAGENT_CROP_ML_HYBRID"
)

echo Current Directory: %CD%
echo.
echo Adding all updated files...
git add -A
git commit -m "Upload complete clean AGRIAGENT project: 3D Web UI, 750+ line ML pipelines, and working official portals" 2>nul

echo.
echo Pushing and overwriting remote with fresh clean code...
git push -u origin main --force
echo.

if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo [SUCCESS] Your project is now 100%% updated and clean on GitHub!
    echo.
    echo Click to view: https://github.com/SaiiieSJaswanth/AGRIAGENT
    echo ======================================================================
) else (
    echo.
    echo [NOTE] If a browser window opened, click "Authorize git-credential-manager" or enter GitHub credentials.
    echo Then press any key to push again.
    pause
    git push -u origin main --force
)
echo.
pause
