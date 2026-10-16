@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
set S1_TABLE_MONTHS=2026-01,2026-02,2026-03,2026-04,2026-05,2026-06
set S1_TABLE_OUT=D:/ChengboRemote/runtime/m1_stage1/s1_table_result_dev6.json
%PY% -u s1_table.py > D:\ChengboRemote\runtime\m1_stage1\s1_table_dev6.log 2>&1
echo EXIT=%ERRORLEVEL% >> D:\ChengboRemote\runtime\m1_stage1\s1_table_dev6.log
