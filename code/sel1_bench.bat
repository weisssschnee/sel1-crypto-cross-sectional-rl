@echo off
rem XS_SEL1 s3.1 step timing and book selftests (GPU VM + CPU book). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_BENCH.log
echo === SEL1_BENCH %DATE% %TIME% > %LOG%
%PY% -I sel1_bench.py >> %LOG% 2>&1
echo === SEL1_BENCH rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
