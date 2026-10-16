@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B3_DEVPANEL.log
echo === DEVPANEL %DATE% %TIME% >> %LOG%
%PY% -I m2h_devpanel.py --build >> %LOG% 2>&1
echo === DEVPANEL_END rc=%ERRORLEVEL% >> %LOG%
