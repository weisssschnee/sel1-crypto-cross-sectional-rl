@echo off
rem BASIN5 debug round, after the PC: (1) wait up to 8 h for the full step-1 gate
rem (B5_GATE1\DECISION.json with limit 0); stop unless it says proceed; (2) b5_round_dbg.py
rem B5_DBG_PC_P32 --mode p32 (hall-of-fame found test, ADR-0009); not found -> stop; (3) found ->
rem real-data formal round B5_DBG_REAL (3 families, 960 updates, seed 20261010, <= 24 h, same
rem trainer and arguments) -> b5_round_dbg.py --mode real -> b5_verdict_dbg.py.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set PC=B5_DBG_PC_P32
set REAL=B5_DBG_REAL
echo === AFTER wait gate %DATE% %TIME% >> %LOG%
%PY% -I b5_wait_gate.py --plant p32 --hours 8 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
if not "%RC%"=="0" (
  echo === GATE says stop or missing rc=%RC%: STOP, waiting for the user >> %LOG%
  exit /b %RC%
)
if not exist D:\ChengboRemote\runtime\basin3_h1\b5_round_dbg.py (
  echo === b5_round_dbg.py missing: STOP >> %LOG%
  exit /b 7
)
echo === PC REPORT %DATE% %TIME% >> %LOG%
%PY% -I b5_round_dbg.py %PC% --mode p32 >> %LOG% 2>&1
set RC=%ERRORLEVEL%
echo === PC REPORT rc=%RC% %DATE% %TIME% (0 found, 10 not found) >> %LOG%
if "%RC%"=="10" (
  echo === NOT FOUND: STOP, no second debug round, waiting for the user >> %LOG%
  exit /b 10
)
if not "%RC%"=="0" (
  echo === REPORT FAILED: STOP >> %LOG%
  exit /b 7
)
if exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  echo === REAL REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\%REAL%\checkpoint_*.pt" (
  echo === REAL REFUSED %DATE% %TIME%: checkpoints exist, use b5_dbg_real_resume.bat >> %LOG%
  exit /b 3
)
echo === REAL START %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_trainer_dbg.py %REAL% --updates 960 --batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward gross_t --select full >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
call D:\ChengboRemote\runtime\basin3_h1\b5_dbg_real_post.bat
