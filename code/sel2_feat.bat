@echo off
rem XS_SEL1 10-16 afternoon: SEL2 step 1b intra-hour features (sel2_feat.py), single thread, TRAIN grid only.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=1
set MKL_NUM_THREADS=1
set OPENBLAS_NUM_THREADS=1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL2_FEAT.log
echo === SEL2_FEAT %DATE% %TIME% > %LOG%
%PY% -I sel2_feat.py >> %LOG% 2>&1
echo === SEL2_FEAT rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
