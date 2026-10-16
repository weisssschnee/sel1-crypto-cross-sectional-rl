@echo off
rem XS_SEL1 10-16 morning: GFN self-check (CPU) + mid-run T_null on NULL ckpt 0527 (CPU, 4 threads). NULL world only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_TNULL_MID.log
echo === GFN SELFTEST %DATE% %TIME% > %LOG%
%PY% -I sel1_gfn.py selftest >> %LOG% 2>&1
echo === GFN rc=%ERRORLEVEL% >> %LOG%
echo === TNULL MID %DATE% %TIME% >> %LOG%
%PY% -I sel1_null_tnull.py --device cpu >> %LOG% 2>&1
echo === TNULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
