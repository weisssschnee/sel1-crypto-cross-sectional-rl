@echo off
rem XS_SEL1 supplement 10-15 CPU self-checks while V2 holds the GPU (CUDA hidden, 4 threads). TRAIN only; no DEV, OOS.
rem 1 GA operators + plant through the fitness (pcnull s20 / K5) 2 GFN TB hand examples 3 pool / zfam / books.
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
cd /d D:\ChengboRemote\runtime\basin3_h1
set OMP_NUM_THREADS=4
set CUDA_VISIBLE_DEVICES=-1
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_XCPU.log
echo === SEL1_XCPU %DATE% %TIME% > %LOG%
%PY% -I sel1_ga.py SEL1_GA_SELFTEST --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --device cpu --selftest --inject-plant >> %LOG% 2>&1
echo === GA selftest rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_ga.py SEL1_GA_SELFTEST_RGP --mode pcnull --alpha 6.908190339580098e-05 --reward t_dd --dd-max 2.283547812033886 --k 5 --device cpu --selftest --rgp 0.5 >> %LOG% 2>&1
echo === GA rgp selftest rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_gfn.py selftest >> %LOG% 2>&1
echo === GFN selftest rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
%PY% -I sel1_xcheck.py >> %LOG% 2>&1
echo === XCHECK rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
