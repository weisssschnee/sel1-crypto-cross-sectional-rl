@echo off
rem XS_SEL1 10-16 ARMS v2: sel1_ga2 self-check on CPU (operators 100 % legal incl. hoist / shrink / window +-1; plant t_A 5.88), pcnull world as the v1 GA self-check (no plant2). No GPU.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_GA2_SELFTEST.log
echo === GA2 SELFTEST %DATE% %TIME% > %LOG%
%PY% -I sel1_ga2.py SEL1_GA2_SELFTEST --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --device cpu --cpu-select --selftest --inject-plant >> %LOG% 2>&1
echo === GA2 SELFTEST rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
