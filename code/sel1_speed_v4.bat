@echo off
rem XS_SEL1 supplement A fused TS operators + GPU mask/selection, no CSE, resumed from the baseline's checkpoint_0002 (copied into SEL1_SPEED_V4): updates 3..5.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SPEED_V4.log
echo === SEL1_SPEED_V4 %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer.py SEL1_SPEED_V4 --mode null --updates 6 --checkpoint-every 3 --workers 8 --gpu-workers 0 --memo global --fastts --gpu-select --speed-resume --resume auto >> %LOG% 2>&1
echo === SEL1_SPEED_V4 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
