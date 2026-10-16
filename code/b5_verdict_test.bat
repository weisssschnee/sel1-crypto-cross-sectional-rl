@echo off
rem Pipeline test of b5_verdict.py on the smoke checkpoint, CPU only (the GPU is training).
rem B5V_TEST=1: 8 per family, 4 TRAIN / 3 DEV tiles, outputs *_TEST.json; not used for selection.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set CUDA_VISIBLE_DEVICES=
set B5V_TEST=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\B5_VERDICT_TEST.log
echo === START %DATE% %TIME% >> %LOG%
%PY% -I b5_verdict.py B5_SMOKE_P8 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
