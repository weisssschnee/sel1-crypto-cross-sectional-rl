@echo off
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set OPENBLAS_NUM_THREADS=4
set CUDA_VISIBLE_DEVICES=
%PY% sel1_timing_trainer.py SMOKE_NOSTEP --events D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1\TIMING\SELFTEST_EVENTS_REAL.csv --device cpu --threads 4 --batch 32 --no-step > D:\ChengboRemote\runtime\basin3_h1\tm_smoke.log 2>&1
echo rc %errorlevel% >> D:\ChengboRemote\runtime\basin3_h1\tm_smoke.log
