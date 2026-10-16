@echo off
rem XS_SEL1 ARMS (supplement 10-15 evening): pause V2 NULL -> GPU 0-diff checks -> 6 arms x 3 h -> rank -> act.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set O=%ROOT%\artifacts\m2h_v1\XS_SEL1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_ARMS.log
set A=%O%\SEL1_ARMS_RULE.json
for %%R in (SEL1_ARMS_RESULT.json SEL1_ARMS_RESULT_TEST.json) do (
  if exist "%O%\%%R" (
    echo === REFUSED %DATE% %TIME%: %%R exists >> %LOG%
    exit /b 3
  )
)
echo === SEL1 ARMS PAUSE CHECK RANK (try 2) %DATE% %TIME% > %LOG%
powershell -NoProfile -ExecutionPolicy Bypass -File sel1_pause_v2.ps1 -Run SEL1_NULL -MaxWaitMin 60 >> %LOG% 2>&1
if errorlevel 1 (
  echo === pause failed rc=%ERRORLEVEL%, ARMS not started >> %LOG%
  exit /b 4
)
echo === paused %DATE% %TIME% >> %LOG%
echo === THREE GPU 0-diff CHECKS (fork from PC ckpt 0015, PC world = V2 PC flags, no plant2) >> %LOG%
set /p FPC=<"%O%\_flags_PC_V2.txt"
%PY% -I sel1_trainer_x.py SEL1_XCHK2_DEDUP0 %FPC% --resume %O%\SEL1_PC\checkpoint_0015.pt --speed-resume --fork --updates 19 --receipt-on-stop >> %LOG% 2>&1
if not errorlevel 1 (
  %PY% -I sel1_logdiff2.py SEL1_PC SEL1_XCHK2_DEDUP0 16 18 --near pg_loss --tag dedup_off >> %LOG% 2>&1
  if errorlevel 1 (
    echo === DEDUP_OFF GPU diff, ARMS cancelled, V2 resumed >> %LOG%
    schtasks /run /tn SEL1_V2_RESUME >nul
    exit /b 8
  )
)
%PY% -I sel1_trainer_x.py SEL1_XCHK2_POOL0 %FPC% --resume %O%\SEL1_PC\checkpoint_0015.pt --speed-resume --fork --pool 0 --updates 19 --receipt-on-stop >> %LOG% 2>&1
if not errorlevel 1 (
  %PY% -I sel1_logdiff2.py SEL1_PC SEL1_XCHK2_POOL0 16 18 --near pg_loss --tag pool0 >> %LOG% 2>&1
  if errorlevel 1 (
    echo === POOL0 GPU diff, ARMS cancelled, V2 resumed >> %LOG%
    schtasks /run /tn SEL1_V2_RESUME >nul
    exit /b 8
  )
)
%PY% -I sel1_trainer_x.py SEL1_XCHK2_POOL20 %FPC% --resume %O%\SEL1_PC\checkpoint_0015.pt --speed-resume --fork --pool 20 --updates 17 --receipt-on-stop >> %LOG% 2>&1
if not errorlevel 1 (
  %PY% -I sel1_logdiff2.py SEL1_PC SEL1_XCHK2_POOL20 16 16 --drop pg_loss,entropy_loss,gradient_norm,mean_entropy,gain --gain-eq-score --tag pool20 >> %LOG% 2>&1
  if errorlevel 1 (
    echo === POOL20 GPU diff, ARMS cancelled, V2 resumed >> %LOG%
    schtasks /run /tn SEL1_V2_RESUME >nul
    exit /b 8
  )
)
echo === GPU 0-diff checks PASSED %DATE% %TIME% >> %LOG%
echo === SIX ARMS x 3 h >> %LOG%
set PFX=SEL1_ARMS_
set FA=--mode pcnull --alpha 6.908190339580098e-05 --plant2 %A% --updates 3122 --k 5 --reward t_dd --dd-max 2.283547812033886 --batch 128 --workers 16 --gpu-workers 0 --memo global --fastts --gpu-select --combo-sweep --checkpoint-every 16 --receipt-on-stop --max-hours 3
%PY% -I sel1_trainer_x.py %PFX%RL %FA% >> %LOG% 2>&1
echo === RL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %PFX%DEDUP %FA% --dedup >> %LOG% 2>&1
echo === DEDUP rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %PFX%GFN %FA% --gfn 1 >> %LOG% 2>&1
echo === GFN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_ga.py %PFX%GA --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --plant2 %A% --workers 16 --max-hours 3 >> %LOG% 2>&1
echo === GA rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %PFX%FROZEN %FA% --frozen >> %LOG% 2>&1
echo === FROZEN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_trainer_x.py %PFX%POOL %FA% --pool 20 >> %LOG% 2>&1
echo === POOL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
echo === RANKING >> %LOG%
%PY% -I sel1_arms_rank.py >> %LOG% 2>&1
set RC_RANK=%ERRORLEVEL%
echo === RANK rc=%RC_RANK% %DATE% %TIME% >> %LOG%
if not exist "%O%\SEL1_ARMS_RESULT.json" (
  echo === RANKING incomplete, V2 resumed >> %LOG%
  schtasks /run /tn SEL1_V2_RESUME >nul
  exit /b 7
)
if %RC_RANK%==0 (
  echo === WINNER = V2 RL, resume NULL with module sha check >> %LOG%
  schtasks /run /tn SEL1_V2_RESUME >nul
  exit /b 0
)
if %RC_RANK%==10 (
  echo === WINNER != V2 RL, launching winner chain >> %LOG%
  %PY% -I sel1_winner_plan.py --result %O%\SEL1_ARMS_RESULT.json >> %LOG% 2>&1
  echo === WINNER PLAN rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
  schtasks /run /tn SEL1_W_CHAIN >nul
  exit /b 10
)
echo === RANK rc unexpected, V2 resumed >> %LOG%
schtasks /run /tn SEL1_V2_RESUME >nul
exit /b 7
