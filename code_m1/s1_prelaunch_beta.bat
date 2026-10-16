@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set PRE_PART=beta
set PRE_OUT=D:/ChengboRemote/runtime/m1_stage1/prelaunch_beta_tiefix.json
%PY% -u s1_prelaunch.py > D:\ChengboRemote\runtime\m1_stage1\prelaunch_beta_tiefix.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\prelaunch_beta_tiefix.log
