@echo off
rem TV1 RGP control, v2 (user 10-11): p8 then the shuffled null, same budget and seed as RGP_P8_V1
rem (10 rounds, pop 512, keep 96, seed 20261011). No seeding from planted RL rounds (--seed-run none).
rem The null runs whether or not p8 finds the plant; it is skipped only if p8 stops on a gate (rc 4/5).
rem Refuses if either run dir already has output; there is no resume.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\RGP_V2_CHAIN.log
for %%R in (RGP_P8_V2 RGP_NULL_V2) do (
  if exist "%ROOT%\artifacts\m2h_v1\%%R\PLANT_GATE.json" (
    echo === REFUSED %DATE% %TIME%: %%R has output >> %LOG%
    exit /b 3
  )
)
echo === START p8 %DATE% %TIME% >> %LOG%
%PY% -I m2h_rgp.py --out RGP_P8_V2 --mode p8 --seed-run none --rounds 10 --pop 512 --keep 96 --seed 20261011 --max-hours 12 > D:\ChengboRemote\runtime\basin3_h1\RGP_P8_V2.log 2>&1
set RC=%ERRORLEVEL%
echo === END p8 %DATE% %TIME% rc=%RC% >> %LOG%
if "%RC%"=="4" exit /b 4
if "%RC%"=="5" exit /b 5
echo === START null %DATE% %TIME% >> %LOG%
%PY% -I m2h_rgp.py --out RGP_NULL_V2 --mode null --seed-run none --rounds 10 --pop 512 --keep 96 --seed 20261011 --max-hours 12 > D:\ChengboRemote\runtime\basin3_h1\RGP_NULL_V2.log 2>&1
echo === END null %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
