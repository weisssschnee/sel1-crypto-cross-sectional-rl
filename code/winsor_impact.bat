@echo off
rem Ruler winsor impact (report only, CPU, V4 TRAIN). winsor_impact.py refuses if its output exists.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\WINSOR_IMPACT.log
echo === WIMP %DATE% %TIME% > %LOG%
%PY% -I winsor_impact.py >> %LOG% 2>&1
echo === WIMP rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
