@echo off
rem XS_SEL1: mechanics test of sel1_round (pc / null / real, first 10 hall entries of SEL1_SPEED_V5, _TEST outputs) and a
rem dry run of sel1_launch_manifest into a _test file. No training; V4 TRAIN only, no DEV, no OOS.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_ROUNDTEST.log
echo === SEL1_ROUNDTEST %DATE% %TIME% > %LOG%
%PY% -I sel1_round.py SEL1_SPEED_V5 --mode pc --top 10 >> %LOG% 2>&1
echo === pc rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_round.py SEL1_SPEED_V5 --mode null --pc SEL1_SPEED_V5 --top 10 >> %LOG% 2>&1
echo === null rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_round.py SEL1_SPEED_V5 --mode real --null SEL1_SPEED_V5 --top 10 >> %LOG% 2>&1
echo === real rc=%ERRORLEVEL% >> %LOG%
%PY% -I -c "import sys,os; sys.path.insert(0,os.getcwd()); import sel1_launch_manifest as L, sel1_common as S; L.OUT=os.path.join(S.OUT_ROOT,'_test_LAUNCH_MANIFEST.json'); L.main()" >> %LOG% 2>&1
echo === manifest rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
