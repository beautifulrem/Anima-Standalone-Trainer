@echo off
cd /d %~dp0

if not exist venv (
    echo Creating venv...
    python -m venv venv
) else (
    echo Venv already exists.
)

call venv\Scripts\activate.bat

echo ----------------------------------------------------------------------
echo Installing requirements from requirements.txt...
echo ----------------------------------------------------------------------
pip install -r requirements.txt

echo.


echo.
echo ----------------------------------------------------------------------
echo Installing UI dependencies (npm install)...
echo ----------------------------------------------------------------------
cd training-ui
call npm install
cd ..

echo.
echo ----------------------------------------------------------------------
echo Installation Complete!
echo ----------------------------------------------------------------------
pause

