@echo off
rem ENR1 TEST (order ENR1 step 2 timing): real world replaced by a null world, 4 null worlds, *_TEST outputs. CPU only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set ENR1B_TEST=1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\ENR1B_TEST.log
echo === ENR1B_TEST %DATE% %TIME% > %LOG%
%PY% -I enr1b.py >> %LOG% 2>&1
echo === ENR1B_TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
