@echo off
rem XS_SEL1 supplement A: segment-sampled scoring vs full A half, rank correlation + seconds (GPU VM, TRAIN A months only; no B, DEV, OOS). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SEGTEST.log
echo === SEL1_SEGTEST %DATE% %TIME% > %LOG%
%PY% -I sel1_segtest.py >> %LOG% 2>&1
echo === SEL1_SEGTEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
