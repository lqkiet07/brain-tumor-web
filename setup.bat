@echo off
setlocal
cd /d "%~dp0"

set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not exist "%PY%" set "PY=python"

echo ============================================
echo  Brain Tumor MRI - Cai dat moi truong ao
echo ============================================

if not exist ".venv\Scripts\python.exe" (
    echo [1/3] Tao moi truong ao .venv ...
    "%PY%" -m venv .venv
    if errorlevel 1 goto :err
) else (
    echo [1/3] Truong ao .venv da ton tai, giu nguyen.
)

echo [2/3] Cai dat thu vien (co the mat vai phut) ...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :err
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :err

echo [3/3] Chuan bi file model ...
if not exist "models" mkdir models
if not exist "..\project 1\report_asset\results_vgg16_b4b5\vgg16_block5_finetuned.h5" (
    echo LOI: Thieu file model vgg16_b4b5 goc trong project 1\report_asset.
    goto :err
)
copy /Y "..\project 1\report_asset\results_vgg16_b4b5\vgg16_block5_finetuned.h5" "models\vgg16_b4b5.h5" >nul
if errorlevel 1 goto :err

if not exist "..\project 1\report_asset\resnet\resnet50_finetuned.h5" (
    echo LOI: Thieu file model resnet50 goc trong project 1\report_asset.
    goto :err
)
copy /Y "..\project 1\report_asset\resnet\resnet50_finetuned.h5" "models\resnet50.h5" >nul
if errorlevel 1 goto :err

if not exist "..\project 1\report_asset\vgg16_block5\vgg16_block5_finetuned.h5" (
    echo LOI: Thieu file model vgg16_block5 goc trong project 1\report_asset.
    goto :err
)
copy /Y "..\project 1\report_asset\vgg16_block5\vgg16_block5_finetuned.h5" "models\vgg16_block5.h5" >nul
if errorlevel 1 goto :err

echo.
echo ============================================
echo  SAN THAI.
echo  Chay run_demo.bat de mo demo tai
echo  http://127.0.0.1:8010
echo ============================================
pause
exit /b 0

:err
echo.
echo LOI: buoc cai dat khong thanh cong.
echo Xem thong bao loi ben tren roi chay lai setup.bat.
pause
exit /b 1
