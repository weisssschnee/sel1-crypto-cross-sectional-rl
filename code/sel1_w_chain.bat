@echo off
rem XS_SEL1 winner chain (when ARMS winner != V2 RL): rename V2 artifacts, winner PC -> round null (P1 only) -> NULL/REAL.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_WINNER.log
set A=%O%\SEL1_ARMS_RULE.json
if not exist "%O%\SEL1_LAUNCH_MANIFEST_W.json" (
  echo === REFUSED %DATE% %TIME%: no winner manifest >> %LOG%
  exit /b 3
)
echo === SEL1 WINNER CHAIN %DATE% %TIME% > %LOG%
set /p WRUN=<%O%\_pc_W.txt
set /p WNAME=<%O%\_cmd_W.txt
echo === Winner: %WNAME% (run %WRUN%) >> %LOG%
echo === Renaming V2 artifacts >> %LOG%
for %%d in (SEL1_PC SEL1_NULL SEL1_REAL) do (
  if exist "%O%\%%d" (
    move "%O%\%%d" "%O%\%%d_V2_SUPERSEDED" >> %LOG% 2>&1
  )
)
for %%f in (_flags_NULL_V2.txt _flags_PC_V2.txt _flags_REAL_V2.txt COMBO_PC.json ROUND_PC.json ROUND_NULL.json ROUND_REAL.json) do (
  if exist "%O%\%%f" (
    move "%O%\%%f" "%O%\%%f.v2bak" >> %LOG% 2>&1
  )
)
echo === Winner PC (ARMS run as PC) >> %LOG%
if exist "%O%\SEL1_PC" (
  echo === ERROR: SEL1_PC already exists >> %LOG%
  exit /b 32
)
xcopy /E /I /Q "%O%\%WRUN%" "%O%\SEL1_PC" >> %LOG% 2>&1
%PY% -I sel1_combo.py pc >> %LOG% 2>&1
echo === COMBO PC rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if errorlevel 1 (
  echo === COMBO PC failed, stopped >> %LOG%
  exit /b 30
)
%PY% -I sel1_round.py pc >> %LOG% 2>&1
echo === ROUND PC (P1 only) rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if errorlevel 1 (
  echo === ROUND PC failed, stopped >> %LOG%
  exit /b 31
)
echo === NULL and REAL with equal budgets >> %LOG%
for /f "delims=" %%f in ('type %O%\_flags_null_W.txt') do set FNULL=%%f
%PY% -I %WNAME% SEL1_NULL %FNULL% >> %LOG% 2>&1
echo === NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_round.py null >> %LOG% 2>&1
echo === ROUND NULL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
if errorlevel 1 (
  echo === ROUND NULL failed, stopped >> %LOG%
  exit /b 40
)
for /f "delims=" %%f in ('type %O%\_flags_real_W.txt') do set FREAL=%%f
%PY% -I %WNAME% SEL1_REAL %FREAL% >> %LOG% 2>&1
echo === REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_round.py real >> %LOG% 2>&1
echo === ROUND REAL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === SEL1 WINNER CHAIN END %DATE% %TIME% >> %LOG%
