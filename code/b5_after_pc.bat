@echo off
rem BASIN5 s5 after the positive control (called by b5_pc.bat / b5_pc_resume.bat, or run alone).
rem 1 b5_round.py B5_PC_P8 --mode p8 -> ROUND_REPORT.json; exit 0 = found (ADR-0008), 10 = not found.
rem 2 not found: stop (order s5: deliver the s6 logs, no real-data round).
rem 3 found: real TRAIN data, three families, 960 updates (B5_REAL_H1, seed 20261010, max 24 h),
rem   then b5_round.py --mode real (converged-to column) and b5_verdict.py (FROZEN_B5_TOP5 + DEV conc verdict).
rem The real run refuses if it already has a receipt or a checkpoint; resume = b5_real_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_CHAIN.log
set REAL=B5_REAL_H1
if exist "%A%\B5_PC_P8\ROUND_REPORT.json" (
  %PY% -I -c "import json,sys; sys.exit(0 if json.load(open(r'%A%\B5_PC_P8\ROUND_REPORT.json',encoding='utf-8'))['found'] else 10)"
) else (
  echo === PC REPORT %DATE% %TIME% >> %LOG%
  %PY% -I b5_round.py B5_PC_P8 --mode p8 >> %LOG% 2>&1
)
set RC=%ERRORLEVEL%
echo === PC REPORT rc=%RC% %DATE% %TIME% >> %LOG%
if "%RC%"=="10" (
  echo === PC NOT FOUND: stop, no real-data round ^(order s5^) >> %LOG%
  exit /b 10
)
if not "%RC%"=="0" (
  echo === PC REPORT FAILED, chain stops >> %LOG%
  exit /b 7
)
if exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  echo === REAL REFUSED %DATE% %TIME%: receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\%REAL%\checkpoint_*.pt" (
  echo === REAL REFUSED %DATE% %TIME%: checkpoints exist, use b5_real_resume.bat >> %LOG%
  exit /b 3
)
echo === REAL START %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_trainer.py %REAL% --updates 960 --batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward gross_t --select full >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
call D:\ChengboRemote\runtime\basin3_h1\b5_real_post.bat
