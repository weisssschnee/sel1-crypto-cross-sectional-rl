@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OMP_NUM_THREADS=4
%PY% -I sel1_sweepcheck.py --old-pc SEL1_PC_V1_COMBOTOKEN > D:\ChengboRemote\runtime\basin3_h1\sweepcheck.log 2>&1
echo rc %errorlevel% >> D:\ChengboRemote\runtime\basin3_h1\sweepcheck.log
