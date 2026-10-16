@echo off
rem Pipeline test of b5_round_dbg.py on the debug smoke dir (hof-n 5, 6 workers, ROUND_REPORT_TEST.json).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_RDBG_TEST.log
echo === RDBG TEST %DATE% %TIME% > %LOG%
%PY% -I b5_round_dbg.py B5_DBG_SMOKE_P32 --mode p32 --hof-rank conc_k10_net --hof-n 3 --final-n 4 --workers 8 --out ROUND_REPORT_TEST.json >> %LOG% 2>&1
echo === RDBG TEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
