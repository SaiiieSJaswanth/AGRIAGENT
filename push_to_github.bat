@echo off
title Push AGRIAGENT to GitHub
echo ======================================================================
echo   AGRIAGENT - ONE-CLICK GITHUB PUSH
echo   Target: https://github.com/SaiiieSJaswanth/AGRIAGENT
echo ======================================================================
echo.

cd /d "D:\AIML_CASESTUDY\AGRIAGENT_FINAL_COMPLETE\AGRIAGENT_CROP_ML_HYBRID"

echo Current Directory: %CD%
echo.
echo Pushing project to GitHub...
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo ======================================================================
    echo [SUCCESS] Your project is now 100%% uploaded to GitHub!
    echo.
    echo Click to view: https://github.com/SaiiieSJaswanth/AGRIAGENT
    echo ======================================================================
) else (
    echo [NOTE] If a browser window opened, click "Authorize git-credential-manager".
    echo Then press any key to try pushing again.
    pause
    git push -u origin main
)
echo.
pause
