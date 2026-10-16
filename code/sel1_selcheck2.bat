@echo off
rem XS_SEL1 supplement 10-15 step 2: GPU mask + selection, second check on the trainer's own candidates (GPU VM, features only). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SELCHECK2.log
echo === SEL1_SELCHECK2 %DATE% %TIME% > %LOG%
%PY% -I sel1_selcheck2.py >> %LOG% 2>&1
echo === SEL1_SELCHECK2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
