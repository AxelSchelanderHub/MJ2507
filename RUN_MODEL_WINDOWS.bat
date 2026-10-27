@echo off
cd /d "%~dp0"
set OPENBLAS_NUM_THREADS=1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
python -m pip install -r requirements.txt
if errorlevel 1 (echo Failed to install dependencies. & pause & exit /b 1)
python occupancy_shallow_nn.py
if errorlevel 1 (echo Model failed to run. & pause & exit /b 1)
echo.
echo Open occupancy_nn_outputs\prediction_quality_dashboard.png
pause
