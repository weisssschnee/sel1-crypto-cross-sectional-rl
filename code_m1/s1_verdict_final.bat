@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_RUN=S1_STAGE1_20261008_VERDICT
set S1_CKPT=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2_v1\S1_STAGE1_20261008\checkpoint_0127.pt
set S1_OUT_DIR=D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2_v1\S1_STAGE1_20261008_VERDICT
%PY% -u s1_final_batch.py
echo EXIT=%ERRORLEVEL%
