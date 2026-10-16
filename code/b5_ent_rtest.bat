@echo off
rem BASIN5_PC_ENT report smoke: b5_round_ent.py on the smoke dir, CPU, 8 per family, no ref history,
rem writes ROUND_REPORT_SMOKE.json in B5_SMOKE_P8_ENT only. Runs while the PC trains (4 threads).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_ENT_RTEST.log
echo === RTEST %DATE% %TIME% > %LOG%
%PY% -I b5_round_ent.py B5_SMOKE_P8_ENT --mode p8 --device cpu --per-family 8 --no-ref-history --hist-n 3 --config %ROOT%\artifacts\m2h_v1\B5_PC_P8_ENT\RUN_CONFIG.json --out ROUND_REPORT_SMOKE.json >> %LOG% 2>&1
echo === RTEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
