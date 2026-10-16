@echo off
rem TV1 auto-chain R1 -> R2 -> R4 -> R3 -> BASIN4. Refuses if the chain already ran once
rem (TV1_CHAIN_STATE.json exists); after a crash use tv1_chain_resume.bat.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set TV1=%ROOT%\artifacts\m2h_v1\TV1_20261010
set LOG=%TV1%\tv1_chain.log
if exist "%TV1%\TV1_CHAIN_STATE.json" (
  echo === REFUSED %DATE% %TIME%: TV1_CHAIN_STATE.json exists, use tv1_chain_resume.bat >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I tv1_chain.py >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
