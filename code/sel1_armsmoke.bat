@echo off
rem XS_SEL1 SEL1_ARMS pipeline smoke on GPU (V2 keeps running): ARMS world, 6 arms x 1 update (batch 4),
rem GA pop 20 x 1 generation, ranking --top 20, winner plan --dry. Pipeline only, no conclusion.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=2
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_ARMSMOKE.log
set P=SEL1_ARMSMOKE_
set A=%O%\SEL1_ARMS_RULE.json
set FA=--mode pcnull --alpha 6.908190339580098e-05 --plant2 %A% --updates 1 --k 5 --reward t_dd --dd-max 2.283547812033886 --batch 4 --workers 2 --gpu-workers 0 --memo global --fastts --gpu-select --combo-sweep --checkpoint-every 16 --device cuda --receipt-on-stop
echo === SEL1_ARMSMOKE %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer_x.py %P%RL %FA% >> %LOG% 2>&1
echo === RL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %P%DEDUP %FA% --dedup >> %LOG% 2>&1
echo === DEDUP rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %P%GFN %FA% --gfn 1 >> %LOG% 2>&1
echo === GFN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_ga.py %P%GA --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --plant2 %A% --workers 2 --device cuda --pop 20 --gens 1 >> %LOG% 2>&1
echo === GA rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %P%FROZEN %FA% --frozen >> %LOG% 2>&1
echo === FROZEN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %P%POOL %FA% --pool 20 >> %LOG% 2>&1
echo === POOL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_arms_rank.py --prefix %P% --top 20 --device cuda >> %LOG% 2>&1
echo === RANK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_winner_plan.py --result %O%\%P%RESULT_TEST.json --winner %P%POOL --dry >> %LOG% 2>&1
echo === PLAN POOL rc=%ERRORLEVEL% >> %LOG%
%PY% -I sel1_winner_plan.py --result %O%\%P%RESULT_TEST.json --winner %P%GA --dry >> %LOG% 2>&1
echo === PLAN GA rc=%ERRORLEVEL% >> %LOG%
echo === SEL1_ARMSMOKE END %DATE% %TIME% >> %LOG%
