@echo off
rem XS_SEL1 10-16 ARMS v2: CPU crash smoke of sel1_trainer_x2 (risk-seek + pq + gamma 1, then gfn2 + explore + pq), ARMS world, small batch. No GPU.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=4
set MKL_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_X2SMOKE.log
set FS=--mode pcnull --alpha 6.908190339580098e-05 --plant2 %O%\SEL1_ARMS_RULE.json --k 5 --reward t_dd --dd-max 2.283547812033886 --batch 48 --workers 0 --gpu-workers 0 --memo global --fastts --combo-sweep --checkpoint-every 1 --receipt-on-stop --device cpu --updates 3 --max-hours 2
echo === X2 SMOKE %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer_x2.py SEL1_X2SMOKE_RISK %FS% --risk-seek 0.05 --pq 8 --ent-gamma 1.0 >> %LOG% 2>&1
echo === RISK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x2.py SEL1_X2SMOKE_GFN2 %FS% --gfn2 1 --explore 0.05 --pq 8 >> %LOG% 2>&1
echo === GFN2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
