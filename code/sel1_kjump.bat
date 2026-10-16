@echo off
rem XS_SEL1 supplement B: K-jump check, plant formula 60 combos x K, pc + real worlds (GPU VM, TRAIN only; no DEV, OOS). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_KJUMP.log
echo === SEL1_KJUMP %DATE% %TIME% > %LOG%
%PY% -I sel1_kjump.py >> %LOG% 2>&1
echo === SEL1_KJUMP rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
