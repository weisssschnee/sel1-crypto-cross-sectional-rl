@echo off
rem V4 panel (order V4_COMBO_V1_CR1 s2.1). CPU only, no OOS. v4_panel.py refuses to overwrite its outputs.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set CUDA_VISIBLE_DEVICES=-1
set OMP_NUM_THREADS=8
set LOG=D:\ChengboRemote\runtime\basin3_h1\V4_PANEL.log
if exist D:\riskpg_stateful_v5s_governor_nextgen_beem\runtime\xs_v4_panel_v1\V4_PANEL_MANIFEST.json (
  echo === V4_PANEL REFUSED: manifest exists %DATE% %TIME% >> %LOG%
  exit /b 3
)
echo === V4_PANEL %DATE% %TIME% > %LOG%
%PY% -I v4_panel.py >> %LOG% 2>&1
echo === V4_PANEL rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
