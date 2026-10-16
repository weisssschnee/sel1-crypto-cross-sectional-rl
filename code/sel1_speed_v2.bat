@echo off
rem XS_SEL1 s3.1 speed test V2: resume V0 checkpoint_0002 with the VM blocked path (BLOCK_ELEMS 64M), 8 book workers, global memo.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_SPEED_V2.log
echo === SEL1_SPEED_V2 %DATE% %TIME% > %LOG%
%PY% -I sel1_trainer.py SEL1_SPEED_V2 --mode null --updates 6 --checkpoint-every 3 --workers 8 --gpu-workers 0 --memo global --block-elems 67108864 --speed-resume --resume auto >> %LOG% 2>&1
echo === SEL1_SPEED_V2 rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
