@echo off
rem TV1 auto-chain after a crash: ended rounds are not retrained, a partial round resumes with
rem --resume auto (the trainer refuses if any argument or module sha differs).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set TV1=%ROOT%\artifacts\m2h_v1\TV1_20261010
set LOG=%TV1%\tv1_chain.log
echo === RESUME %DATE% %TIME% >> %LOG%
%PY% -I tv1_chain.py --resume-partial >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
