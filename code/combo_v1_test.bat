@echo off
rem COMBO_V1 pipeline test (order V4_COMBO_V1_CR1 s2.2): 2 tiles per year, "DEV" = last 4 TRAIN tiles,
rem *_TEST outputs, no OKX. DEV and OOS are not opened. CPU only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set COMBO_TEST=1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\COMBO_V1_TEST.log
echo === COMBO1_TEST %DATE% %TIME% > %LOG%
%PY% -I combo_v1.py >> %LOG% 2>&1
echo === COMBO1_TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
