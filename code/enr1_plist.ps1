Get-CimInstance Win32_Process -Filter "Name='python.exe'" | ForEach-Object {
  $c = [string]$_.CommandLine
  if ($c -match '(sel1_\w+\.py(\s+\S+)?)') { $s = $matches[1] } elseif ($c -match 'multiprocessing') { $s = "mp-worker of $($_.ParentProcessId)" } else { $s = $c.Substring(0, [Math]::Min(70, $c.Length)) }
  "{0} {1} {2}MB {3}" -f $_.ProcessId, $_.CreationDate.ToString('HH:mm'), [int]($_.WorkingSetSize/1MB), $s
}
