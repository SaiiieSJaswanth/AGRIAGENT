@echo off
title Push AGRIAGENT to GitHub
echo ======================================================================
echo   AGRIAGENT - ONE-CLICK GITHUB PUSH
echo   Target Repository: https://github.com/SaiiieSJaswanth/AGRIAGENT
echo ======================================================================
echo.

cd /d "D:\AIML_CASESTUDY\AGRIAGENT_FINAL_COMPLETE\AGRIAGENT_CROP_ML_HYBRID"

echo Current Directory: %CD%
echo.
echo Pushing commits to GitHub branch 'main'...
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] AGRIAGENT successfully pushed to GitHub!
    echo.
    echo View your code at: https://github.com/SaiiieSJaswanth/AGRIAGENT
) else (
    echo.
    echo [NOTE] If a browser window opened, click "Authorize git-credential-manager".
    echo If GitHub rejected due to files already on GitHub, run:
    echo   git pull origin main --rebase
    echo   git push -u origin main
)
echo.
pause
