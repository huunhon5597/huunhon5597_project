@echo off
REM Script to refresh sstock and valueinvesting.io cookies and auto-push to GitHub
REM This script should be run periodically (e.g., daily) to keep cookies fresh

echo ========================================
echo   Refreshing cookies (sstock & valueinvesting.io)...
echo ========================================

REM Navigate to the project directory
cd /d D:\Nhon\Stocks\Project\streamlit

REM Run Python script to fetch new cookies
echo Running cookie fetcher...
python auth/fetch_cookie.py

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ========================================
    echo   SUCCESS: Cookies refreshed!
    echo ========================================
    
    echo.
    echo Pushing to GitHub...
    echo.
    
    REM Add changes for both cookie files
    git add sstock_cookie.txt valueinvesting.txt
    
    REM Commit
    git commit -m "Update cookies (sstock & valueinvesting.io)"
    
    REM Push to remote
    git push origin main
    
    if %ERRORLEVEL% EQU 0 (
        echo.
        echo ========================================
        echo   SUCCESS: Pushed to GitHub!
        echo ========================================
    ) else (
        echo.
        echo ========================================
        echo   WARNING: Failed to push to GitHub
        echo ========================================
    )
) else (
    echo.
    echo ========================================
    echo   ERROR: Failed to refresh cookies
    echo ========================================
    exit /b 1
)

REM Exit
exit /b 0
