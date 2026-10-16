@echo off
rem XS_SEL1 s3.1 VM cost profile (diagnostic, read-only, TRAIN only), BLOCK_ELEMS given as %1 (0 = whole panel).
call D:\ChengboRemote\runtime\basin3_h1\b3_env.bat
set LOG=D:\ChengboRemote\runtime\basin3_h1\SEL1_VMPROF.log
echo === SEL1_VMPROF %DATE% %TIME% > %LOG%
%PY% -I sel1_vmprof.py 0 fast >> %LOG% 2>&1
echo === SEL1_VMPROF rc=%ERRORLEVEL% %DATE% %TIME% >> %LOG%
