@echo off
rem COMBO_V0 pipeline test (ADR-0010): stages 0-1 on TRAIN, stage 2 on the last 12 TRAIN tiles
rem (*_TEST outputs, no OKX). DEV and OOS are not opened. CPU only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set COMBO_TEST=1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\COMBO_V0_TEST.log
echo === COMBO_TEST %DATE% %TIME% > %LOG%
%PY% -I combo_v0.py >> %LOG% 2>&1
echo === COMBO_TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
