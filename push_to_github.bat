@echo off
title Push AGRIAGENT to GitHub
echo ======================================================================
echo   AGRIAGENT - ONE-CLICK GITHUB PUSH
echo   Target Repository: https://github.com/SaiiieSJaswanth/AGRIAGENT
echo ======================================================================
echo.
echo Make sure you have created the empty repository "AGRIAGENT" at:
echo   https://github.com/new
echo.
echo Pushing commits to GitHub branch 'main'...
echo.
git push -u origin main
echo.
if %ERRORLEVEL% EQU 0 (
    echo [SUCCESS] AGRIAGENT successfully pushed to GitHub!
    echo View your code at: https://github.com/SaiiieSJaswanth/AGRIAGENT
) else (
    echo [NOTE] If repository was not found, please create it at https://github.com/new
    echo with repository name: AGRIAGENT (Public or Private)
    echo Then run this script again.
)
echo.
pause
