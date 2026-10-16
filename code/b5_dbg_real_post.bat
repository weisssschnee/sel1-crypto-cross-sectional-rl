@echo off
rem BASIN5 formal round post: needs the receipt; b5_round_dbg.py B5_DBG_REAL --mode real
rem (hall of fame, converged-to), then b5_verdict_dbg.py B5_DBG_REAL (TRAIN HoF freeze ->
rem FROZEN_B5_TOP5.json, then DEV K10 paired verdict + K5/K20, full pool, OKX column). No OOS.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set REAL=B5_DBG_REAL
if not exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  echo === REAL has no receipt, stops ^(resume: b5_dbg_real_resume.bat^) >> %LOG%
  exit /b 6
)
echo === REAL REPORT %DATE% %TIME% >> %LOG%
%PY% -I b5_round_dbg.py %REAL% --mode real >> %LOG% 2>&1
echo === REAL REPORT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if not exist D:\ChengboRemote\runtime\basin3_h1\b5_verdict_dbg.py (
  echo === b5_verdict_dbg.py missing: STOP before DEV >> %LOG%
  exit /b 7
)
echo === VERDICT %DATE% %TIME% >> %LOG%
%PY% -I b5_verdict_dbg.py %REAL% >> %LOG% 2>&1
echo === VERDICT rc=%ERRORLEVEL% %DATE% %TIME%; STOP, waiting for the user >> %LOG%
