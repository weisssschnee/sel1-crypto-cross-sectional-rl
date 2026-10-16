@echo off
rem BASIN5 step 1 (BASIN5_ORDER s1): reward pick, re-scoring only, no training.
rem b5_reward_pick.py itself waits until RGP_P8_V2 and RGP_NULL_V2 both wrote FROZEN_RGP_TOP5.json and
rem the RGP chain logged "END null" (s0: one GPU, one job), then scores both T1 pools on GPU.
rem Refuses if REWARD_PICK.json already exists; there is no resume (the step is a pure re-score).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_PICK.log
if exist "%ROOT%\artifacts\m2h_v1\BASIN5_20261012\REWARD_PICK.json" (
  echo === REFUSED %DATE% %TIME%: REWARD_PICK.json exists >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I b5_reward_pick.py >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
