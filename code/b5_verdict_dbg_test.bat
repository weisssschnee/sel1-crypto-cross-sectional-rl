@echo off
rem Freeze-only pipeline test of b5_verdict_dbg.py on the debug smoke dir (CPU; DEV not opened).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set B5V_DBG_TEST=1
set CUDA_VISIBLE_DEVICES=
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_VDBG_TEST.log
echo === VDBG TEST %DATE% %TIME% > %LOG%
%PY% -I b5_verdict_dbg.py B5_DBG_SMOKE_P32 >> %LOG% 2>&1
echo === VDBG TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
