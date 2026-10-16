@echo off
rem XS_SEL1 supplement 10-15 late night: NULL-world plant check grid (strength {20,40} x K {5,10,20}, t_dd combo rule). Check only, not a training.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_NULLPLANT.log
echo === SEL1_NULLPLANT %DATE% %TIME% > %LOG%
%PY% -I sel1_nullplant.py grid >> %LOG% 2>&1
echo === SEL1_NULLPLANT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
