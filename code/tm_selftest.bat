@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set CUDA_VISIBLE_DEVICES=
%PY% sel1_timing_selftest.py %* > D:\ChengboRemote\runtime\basin3_h1\tm_selftest.log 2>&1
echo rc %errorlevel% >> D:\ChengboRemote\runtime\basin3_h1\tm_selftest.log
