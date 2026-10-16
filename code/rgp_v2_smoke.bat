@echo off
rem TV1 RGP control, v2 smoke (user 10-11): gate 1 = windowed share of round-0 candidates > 0.30,
rem gate 2 = plant formula scored before the search, gross ~ base + lift and corr with s > 0.95.
rem m2h_rgp.py exits 4 (gate 2) or 5 (gate 1) on a failure. Refuses if the run dir has output.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set RUN_DIR=%ROOT%\artifacts\m2h_v1\RGP_SMOKE_V2
set LOG=D:\ChengboRemote\runtime\basin3_h1\RGP_SMOKE_V2.log
if exist "%RUN_DIR%\PLANT_GATE.json" (
  echo === REFUSED %DATE% %TIME%: PLANT_GATE.json exists >> %LOG%
  exit /b 3
)
echo === START %DATE% %TIME% >> %LOG%
%PY% -I m2h_rgp.py --out RGP_SMOKE_V2 --mode p8 --seed-run none --rounds 2 --pop 64 --keep 16 --seed 20261011 --max-hours 1 >> %LOG% 2>&1
echo === END %DATE% %TIME% rc=%ERRORLEVEL% >> %LOG%
