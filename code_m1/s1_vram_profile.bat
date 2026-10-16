@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% -u s1_vram_profile.py > D:\ChengboRemote\runtime\m1_stage1\vram_profile.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\vram_profile.log
