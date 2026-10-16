@echo off
rem XS_SEL1 10-16: T_null rerun (sel1_null_tnull2.py = progress-line copy) on NULL ckpt 0527, CPU, 4 threads. NULL world only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_TNULL_R2.log
echo === TNULL R2 %DATE% %TIME% > %LOG%
%PY% -I sel1_null_tnull2.py --device cpu --tag _r2 >> %LOG% 2>&1
echo === TNULL R2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
