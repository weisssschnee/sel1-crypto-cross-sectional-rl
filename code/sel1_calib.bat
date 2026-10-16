@echo off
rem XS_SEL1 s3.2 plant calibration + launch checks (visible / rankable). V4 TRAIN only, no DEV, no OOS, no training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_CALIB.log
echo === SEL1_CALIB %DATE% %TIME% > %LOG%
%PY% -I sel1_calib.py >> %LOG% 2>&1
echo === SEL1_CALIB rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
