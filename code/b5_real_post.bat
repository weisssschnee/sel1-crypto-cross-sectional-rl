@echo off
rem BASIN5 after the real run ends (receipt or STOPPED): round report (converged-to column, conc TRAIN net)
rem and the verdict (FROZEN_B5_TOP5.json, then DEV conc K10 paired verdict + OKX column).
rem Each step is skipped if its output exists; the verdict refuses to overwrite VERDICT_B5.json.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_CHAIN.log
set REAL=B5_REAL_H1
if not exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  if not exist "%A%\%REAL%\STOPPED.json" (
    echo === REAL has no receipt and no STOPPED.json, stops ^(resume: b5_real_resume.bat^) >> %LOG%
    exit /b 6
  )
)
if not exist "%A%\%REAL%\ROUND_REPORT.json" (
  echo === REAL REPORT %DATE% %TIME% >> %LOG%
  %PY% -I b5_round.py %REAL% --mode real >> %LOG% 2>&1
  echo === REAL REPORT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
)
if not exist "%A%\%REAL%\VERDICT_B5.json" (
  echo === VERDICT %DATE% %TIME% >> %LOG%
  %PY% -I b5_verdict.py %REAL% >> %LOG% 2>&1
  echo === VERDICT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
)
echo === B5 CHAIN END %DATE% %TIME% >> %LOG%
