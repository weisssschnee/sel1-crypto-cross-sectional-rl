@echo off
rem XS_SEL1 supplement 10-15 night prep: --reward mean zero-diff (speed-resume vs V0), t / t_dd hand check, NULL plant pipeline at alpha 0. Checks only, one GPU job at a time.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_PREP_N.log
echo === SEL1_PREP_N %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer.py SEL1_SPEED_V5 --mode null --updates 6 --checkpoint-every 3 --workers 8 --gpu-workers 0 --memo global --fastts --gpu-select --speed-resume --resume auto >> %LOG% 2>&1
echo === V5 rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_logdiff.py SEL1_SPEED_V0 SEL1_SPEED_V5 3 >> %LOG% 2>&1
echo === LOGDIFF rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_rewardcheck.py >> %LOG% 2>&1
echo === REWARDCHECK rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_nullplant.py ddmax >> %LOG% 2>&1
echo === DDMAX rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_nullplant.py pipe >> %LOG% 2>&1
echo === PIPE rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
