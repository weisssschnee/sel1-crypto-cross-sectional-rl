@echo off
rem XS_SEL1 supplement B power table (GPU VM + CPU book). Check only, not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_POWER.log
echo === SEL1_POWER %DATE% %TIME% > %LOG%
%PY% -I sel1_power.py >> %LOG% 2>&1
echo === SEL1_POWER rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
