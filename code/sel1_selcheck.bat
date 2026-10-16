@echo off
rem XS_SEL1 supplement 10-15 step 2: GPU mask + selection, zero-diff check + timing (GPU VM, features only). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SELCHECK.log
echo === SEL1_SELCHECK %DATE% %TIME% > %LOG%
%PY% -I sel1_selcheck.py >> %LOG% 2>&1
echo === SEL1_SELCHECK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
