@echo off
rem CR1 import check only (no data read).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
%PY% -I -c "import sys;sys.path.insert(0,r'D:\ChengboRemote\runtime\basin3_h1');import cr1_common,cr1_calib,cr1_round,cr1_trainer;print('IMPORT OK')" > D:\ChengboRemote\runtime\basin3_h1\CR1_IMPORT.log 2>&1
