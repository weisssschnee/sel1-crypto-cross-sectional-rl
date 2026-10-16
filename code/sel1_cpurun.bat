@echo off
rem XS_SEL1 helper: m1cu128 env, CPU only, 4 threads; runs %PY% -I with the given args (short self-checks).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
%PY% -I %*
