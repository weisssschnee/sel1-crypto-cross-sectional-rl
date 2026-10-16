@echo off
rem XS_SEL1 supplement A: fused TS operators, bit-for-bit check + timing (GPU VM, features only). Not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\sel1_fastbench.log
echo === sel1_fastbench %DATE% %TIME% > %LOG%
%PY% -I sel1_fastbench.py >> %LOG% 2>&1
echo === sel1_fastbench rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
