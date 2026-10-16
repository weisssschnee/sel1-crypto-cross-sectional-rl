@echo off
rem XS_SEL1 supplement 10-15 CPU part 2 while V2 holds the GPU: sel1_combo self-check on the V2 PC hall, then the P2 pick (SEL1_ARMS_RULE.json). TRAIN only; no DEV, OOS.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set CUDA_VISIBLE_DEVICES=-1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_XCPU2.log
echo === SEL1_XCPU2 %DATE% %TIME% > %LOG%
%PY% -I sel1_combo.py SEL1_PC --mode pc --device cpu >> %LOG% 2>&1
echo === COMBO pc rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_p2pick.py --device cpu >> %LOG% 2>&1
echo === P2PICK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
