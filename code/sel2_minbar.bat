@echo off
rem XS_SEL2 step 1 (peer ruling 10-16): m1 shard columns -> V4 hourly grid, TRAIN months only, CPU 4 threads.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL2_MINBAR.log
echo === SEL2 MINBAR %DATE% %TIME% > %LOG%
%PY% -I sel2_minbar.py >> %LOG% 2>&1
echo === SEL2 MINBAR rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
