@echo off
rem XS_SEL1 10-16: OKX items plumbing on TRAIN (plant entry), CPU 4 threads; no DEV, no OOS.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_OKX2_TRAIN.log
echo === OKX2 TRAIN %DATE% %TIME% > %LOG%
%PY% -I sel1_okx_pool2.py --world train --plant >> %LOG% 2>&1
echo === OKX2 TRAIN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
