@echo off
rem BASIN5 formal round, memo resume (user 10-13: stop at a checkpoint, speed up). Run only after the
rem B5_DBG_CHAIN task and its trainer are stopped. (1) drop a half-written last JSON line;
rem (2) check run B5_DBG_REAL_MEMOCHK: m2h_b5_trainer_dbg_memo resumed from the second-latest
rem B5_DBG_REAL checkpoint redoes the 16 logged updates after it; b5_memochk compares the rows;
rem (3) pass -> resume B5_DBG_REAL with the memo trainer, fail -> resume with the plain trainer;
rem (4) b5_dbg_real_post.bat (HoF report -> TRAIN freeze -> DEV verdict -> stop). No OOS.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set A=%ROOT%\artifacts\m2h_v1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_DBG.log
set REAL=B5_DBG_REAL
set CHK=B5_DBG_REAL_MEMOCHK
set BASE=--batch 128 --tiles 16 --seed 20261010 --max-hours 24 --reward gross_t --select full
if exist "%A%\%REAL%\TRAIN_RECEIPT_V1.json" (
  echo === MEMO REFUSED %DATE% %TIME%: %REAL% receipt exists >> %LOG%
  exit /b 3
)
if exist "%A%\%CHK%\FORMULAS.jsonl" (
  echo === MEMO REFUSED %DATE% %TIME%: %CHK% already ran >> %LOG%
  exit /b 3
)
echo === MEMO %DATE% %TIME% >> %LOG%
%PY% -I b5_tailfix.py %REAL% >> %LOG% 2>&1
set CK=
for /f "tokens=1-3" %%a in ('%PY% -I b5_memo_plan.py') do (
  set CK=%%a
  set U0=%%b
  set U1=%%c
)
if "%CK%"=="" (
  echo === MEMO REFUSED %DATE% %TIME%: fewer than two checkpoints >> %LOG%
  exit /b 3
)
set /a UPD=%U1%+1
echo === MEMOCHK from %CK% updates %U0%..%U1% %DATE% %TIME% >> %LOG%
%PY% -I m2h_b5_trainer_dbg_memo.py %CHK% --updates %UPD% %BASE% --resume %CK% >> %LOG% 2>&1
echo === MEMOCHK run rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I b5_memochk.py %U0% %U1% >> %LOG% 2>&1
if errorlevel 1 (
  echo === MEMOCHK FAIL: resuming with the plain trainer >> %LOG%
  set TR=m2h_b5_trainer_dbg.py
) else (
  set TR=m2h_b5_trainer_dbg_memo.py
)
echo === REAL RESUME %TR% %DATE% %TIME% >> %LOG%
%PY% -I %TR% %REAL% --updates 960 %BASE% --resume auto >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
call D:\ChengboRemote\runtime\basin3_h1\b5_dbg_real_post.bat
