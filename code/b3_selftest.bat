@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\B3_SELFTEST.log
echo === SELFTEST %DATE% %TIME% >> %LOG%
%PY% -I m2h_selftest.py >> %LOG% 2>&1
echo === SELFTEST_END rc=%ERRORLEVEL% >> %LOG%
echo === SMOKE %DATE% %TIME% >> %LOG%
%PY% -I m2h_trainer.py SMOKE_B3_V2 --updates 2 --batch 32 --seed 20261009 --checkpoint-every 1 >> %LOG% 2>&1
echo === SMOKE_END rc=%ERRORLEVEL% >> %LOG%
