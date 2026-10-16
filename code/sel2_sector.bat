@echo off
rem XS_SEL1: SEL2 step 2 sector clustering (8 clusters, 720h lookback, TRAIN 2024-2025).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=8
set MKL_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL2_SECTOR.log
echo === SEL2 SECTOR %DATE% %TIME% > %LOG%
%PY% -I sel2_sector.py --k 8 >> %LOG% 2>&1
echo === SEL2 SECTOR rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
