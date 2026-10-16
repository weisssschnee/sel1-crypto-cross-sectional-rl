@echo off
rem BASIN5_PC_ENT after the positive control (called by b5_pc_ent.bat / b5_pc_ent_resume.bat, or run
rem alone): b5_round_ent.py B5_PC_P8_ENT --mode p8 -> ROUND_REPORT.json (0 = found, 10 = not found).
rem Then stops whatever the result (order s7): no real-data round, no null, no DEV.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_PC_ENT.log
set PC=B5_PC_P8_ENT
if exist "%A%\%PC%\ROUND_REPORT.json" (
  echo === REPORT EXISTS, not rewritten %DATE% %TIME% >> %LOG%
  exit /b 0
)
echo === REPORT %DATE% %TIME% >> %LOG%
%PY% -I b5_round_ent.py %PC% --mode p8 >> %LOG% 2>&1
echo === REPORT rc=%ERRORLEVEL% %DATE% %TIME% (0 found, 10 not found); STOP, waiting for the user >> %LOG%
