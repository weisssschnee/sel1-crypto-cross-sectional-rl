# XS_SEL1 SEL1_ARMS (supplement 10-15 evening): pause V2 right after a checkpoint. Wait for a checkpoint newer than
# the one present at start, stop the chain / resume bat first and confirm it is gone (it cannot reach round null or
# REAL), then stop the trainer's process tree. Only our own processes; 11308 / 25400 are never touched.
#   powershell -NoProfile -ExecutionPolicy Bypass -File sel1_pause_v2.ps1 [-Run SEL1_NULL] [-MaxWaitMin 60]
# rc 0 paused, 4 no new checkpoint in time, 5 a bat survived, 6 the trainer survived, 7 no trainer running.
param([string]$Run = "SEL1_NULL", [int]$MaxWaitMin = 60)
$O = "D:\riskpg_stateful_v5s_governor_nextgen_beem\artifacts\m2h_v1\XS_SEL1"
$keep = @(11308, 25400)
$batRe = 'sel1_chain_v2\.bat|sel1_v2_resume\.bat'
$trRe = "sel1_trainer\.py $Run "

function Trainers { Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ? { $_.CommandLine -match $trRe -and $keep -notcontains $_.ProcessId } }
function Bats { Get-CimInstance Win32_Process -Filter "Name='cmd.exe'" | ? { $_.CommandLine -match $batRe -and $keep -notcontains $_.ProcessId } }
function Latest { (Get-ChildItem "$O\$Run" -Filter "checkpoint_*.pt" -ErrorAction SilentlyContinue | ? { $_.Name -match '^checkpoint_\d{4}\.pt$' } | Sort-Object Name | Select-Object -Last 1).Name }

if (-not (Trainers)) { "pause: no $Run trainer running"; exit 7 }
$ck0 = Latest
"pause: waiting for a checkpoint after $ck0"
$t0 = Get-Date
while ((Latest) -eq $ck0) {
  if (((Get-Date) - $t0).TotalMinutes -gt $MaxWaitMin) { "pause: no new checkpoint in $MaxWaitMin min"; exit 4 }
  if (-not (Trainers)) { "pause: $Run trainer gone while waiting"; exit 7 }
  Start-Sleep 5
}
$ck = Latest
foreach ($p in Bats) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue }
Start-Sleep 2
if (Bats) { "pause: bat still alive"; exit 5 }
"pause: bat stopped after $ck"
foreach ($p in Trainers) { taskkill /F /T /PID $p.ProcessId 2>&1 | Out-Null }
Start-Sleep 5
if (Trainers) { "pause: trainer still alive"; exit 6 }
"pause: $Run stopped, resume point $ck"
exit 0
