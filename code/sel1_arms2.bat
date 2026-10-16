@echo off
rem XS_SEL1 ARMS v2 relay (supplement 10-16 noon). Started by the tasks SEL1_V2_RESUME / SEL1_W_CHAIN, which ARMS v1's
rem sel1_arms.bat runs at its end; they point here until ARMS v2 is done (V2 NULL stays paused). The original actions are
rem kept as SEL1_V2_RESUME_REAL / SEL1_W_CHAIN_REAL and are run from here once the merged ranking is out.
rem   0-diff: sel1_trainer_x2 with every switch off, fork of V2 PC ckpt 0015, logs 16..18 vs SEL1_PC (as ARMS v1's checks)
rem   arms:   PG-risk, GFN-fix, GA-fix, 3 h each, ARMS v1's world / seed / flags
rem   rank:   sel1_arms_rank2 (v1 RL / DEDUP / FROZEN / POOL rows + the three v2 arms) -> winner rule as v1
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_ARMS2.log
set A=%O%\SEL1_ARMS_RULE.json
if exist "%O%\SEL1_ARMS2_RESULT.json" (
  echo === REFUSED %DATE% %TIME%: SEL1_ARMS2_RESULT.json exists >> %LOG%
  exit /b 3
)
if exist "%O%\SEL1_ARMS2_PGRISK" (
  echo === REFUSED %DATE% %TIME%: SEL1_ARMS2_PGRISK exists, relay already ran >> %LOG%
  exit /b 3
)
echo === SEL1 ARMS v2 RELAY %DATE% %TIME% > %LOG%
set /p FPC=<"%O%\_flags_PC_V2.txt"
set PFX=SEL1_ARMS2_
set FA=--mode pcnull --alpha 6.908190339580098e-05 --plant2 %A% --updates 3122 --k 5 --reward t_dd --dd-max 2.283547812033886 --batch 128 --workers 16 --gpu-workers 0 --memo global --fastts --gpu-select --combo-sweep --checkpoint-every 16 --receipt-on-stop --max-hours 3
%PY% -I sel1_trainer_x2.py SEL1_XCHK3_X2OFF %FPC% --resume %O%\SEL1_PC\checkpoint_0015.pt --speed-resume --fork --updates 19 --receipt-on-stop >> %LOG% 2>&1
if errorlevel 1 goto x2fail
%PY% -I sel1_logdiff2.py SEL1_PC SEL1_XCHK3_X2OFF 16 18 --near pg_loss --tag x2off >> %LOG% 2>&1
if errorlevel 1 goto x2fail
echo === X2 OFF 0-diff PASSED %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x2.py %PFX%PGRISK %FA% --risk-seek 0.05 --pq 64 --ent-gamma 1.0 >> %LOG% 2>&1
echo === PGRISK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x2.py %PFX%GFN %FA% --gfn2 1 --explore 0.05 --pq 64 >> %LOG% 2>&1
echo === GFN2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
goto ga
:x2fail
echo === X2 OFF 0-diff FAILED %DATE% %TIME%: PG-risk and GFN-fix not run >> %LOG%
:ga
%PY% -I sel1_ga2.py %PFX%GA --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --plant2 %A% --workers 16 --max-hours 3 >> %LOG% 2>&1
echo === GA2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === MERGED RANKING >> %LOG%
%PY% -I sel1_arms_rank2.py >> %LOG% 2>&1
set RC_RANK=%ERRORLEVEL%
echo === RANK2 rc=%RC_RANK% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_ARMS2_RESULT.json" (
  echo === RANKING incomplete, V2 resumed >> %LOG%
  schtasks /run /tn SEL1_V2_RESUME_REAL >nul
  exit /b 7
)
if %RC_RANK%==0 (
  echo === WINNER = V2 RL, resume NULL with module sha check >> %LOG%
  schtasks /run /tn SEL1_V2_RESUME_REAL >nul
  exit /b 0
)
if %RC_RANK%==10 (
  echo === WINNER != V2 RL, launching winner chain >> %LOG%
  for %%f in (SEL1_LAUNCH_MANIFEST_W.json _flags_NULL_W.txt _flags_REAL_W.txt _cmd_W.txt _pc_W.txt) do (
    if exist "%O%\%%f" move /y "%O%\%%f" "%O%\%%f.arms1" >> %LOG% 2>&1
  )
  %PY% -I sel1_winner_plan2.py --result %O%\SEL1_ARMS2_RESULT.json >> %LOG% 2>&1
  echo === WINNER PLAN2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
  schtasks /run /tn SEL1_W_CHAIN_REAL >nul
  exit /b 10
)
echo === RANK2 rc unexpected, V2 resumed >> %LOG%
schtasks /run /tn SEL1_V2_RESUME_REAL >nul
exit /b 7
