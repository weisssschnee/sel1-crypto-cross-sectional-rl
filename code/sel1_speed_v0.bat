@echo off
rem XS_SEL1 s3.1 baseline speed run (null world, 6 updates, checkpoint every 3): everything in-process, per-update memo.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SPEED_V0.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1\SEL1_SPEED_V0\checkpoint_0002.pt (
  echo === REFUSED: checkpoint exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === SEL1_SPEED_V0 %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer.py SEL1_SPEED_V0 --mode null --updates 6 --checkpoint-every 3 --workers 0 --gpu-workers 0 --memo update >> %LOG% 2>&1
echo === SEL1_SPEED_V0 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
