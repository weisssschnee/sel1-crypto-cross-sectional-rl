@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_STAGE1_20261008_VERDICT
set LOG=D:\ChengboRemote\runtime\m1_stage1\S1_VERDICT_EVAL.log
if exist "D:\ChengboRemote\runtime\m1_stage1\s1_paired_eval_S1_STAGE1_20261008_VERDICT.json" (
  echo === REFUSED %DATE% %TIME%: output json exists >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -u s1_paired_eval.py >> %LOG% 2>&1
echo EXIT=%ERRORLEVEL% >> %LOG%
