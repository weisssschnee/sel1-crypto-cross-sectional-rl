@echo off
call D:\ChengboRemote\runtime\m1_stage1\s1_env.bat
%PY% -I s1_sha.py m2_trainer.py m1_ruler.py m1_common.py m1_sample.py s1_final_batch.py s1_paired_eval.py s1_paired_report.py
