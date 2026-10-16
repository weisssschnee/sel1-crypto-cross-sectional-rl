@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B3_VERDICT.log
echo === VERDICT %DATE% %TIME% >> %LOG%
%PY% -I m2h_verdict.py >> %LOG% 2>&1
echo === VERDICT_END rc=%ERRORLEVEL% >> %LOG%
