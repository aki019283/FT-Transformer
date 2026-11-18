param(
 [ValidateSet('Smoke','Full')][string]$Mode='Smoke',
 [string]$Project='F:\aki\mlp_factor',
 [string]$Python='F:\aki\mlp_factor\.venv312\Scripts\python.exe'
)
$ErrorActionPreference='Stop'
$env:PYTHONUTF8='1'
$env:OMP_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$base=Join-Path $Project 'data\experiments\ft_paper_v2_rank64'
New-Item -ItemType Directory -Path $base -Force | Out-Null
$local=Join-Path $base 'source_snapshot'
if (!(Test-Path -LiteralPath $local)) {
 New-Item -ItemType Directory -Path $local | Out-Null
}
Get-ChildItem -LiteralPath $PSScriptRoot -Recurse -File | Where-Object { $_.FullName -notmatch '__pycache__' } | ForEach-Object {
 $rel=$_.FullName.Substring($PSScriptRoot.Length).TrimStart('\')
 $target=Join-Path $local $rel
 if (Test-Path -LiteralPath $target) {
  if ((Get-FileHash -LiteralPath $_.FullName).Hash -ne (Get-FileHash -LiteralPath $target).Hash) { throw "Frozen source differs: $rel. Use a new experiment directory." }
 } else {
  New-Item -ItemType Directory -Path (Split-Path $target) -Force | Out-Null
  Copy-Item -LiteralPath $_.FullName -Destination $target
 }
}
if (!(Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Python missing: $Python" }
$output=Join-Path $base $Mode.ToLower()
New-Item -ItemType Directory -Path $output -Force | Out-Null
$lock=$null
try {
 $lock=[IO.File]::Open((Join-Path $output 'run.lock'),[IO.FileMode]::OpenOrCreate,[IO.FileAccess]::ReadWrite,[IO.FileShare]::None)
 # Capture native stderr as a file, not a PowerShell ErrorRecord (Windows PS 5.1).
 function Invoke-PythonLogged([string[]]$Argv,[string]$Stdout,[string]$Stderr) {
  $quoted=@($Argv | ForEach-Object {
   if ($_ -match '"' -or $_.EndsWith('\')) { throw 'Unsupported quote/trailing slash in argument' }
   '"'+$_+'"'
  })
  $process=Start-Process -FilePath $Python -ArgumentList $quoted -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput $Stdout -RedirectStandardError $Stderr
  return $process.ExitCode
 }
 $stamp=Get-Date -Format 'yyyyMMdd_HHmmss_fff'
 $testout=Join-Path $output ("tests_$stamp.stdout.log")
 $testerr=Join-Path $output ("tests_$stamp.stderr.log")
 $exitCode=Invoke-PythonLogged @('-B',(Join-Path $local 'test_paper.py')) $testout $testerr
 if ($exitCode -ne 0) { Get-Content -LiteralPath $testerr; throw 'Architecture tests failed' }
 $arguments=@('-B','-u',(Join-Path $local 'study.py'),'--project-root',$Project,'--output',$output)
 if ($Mode -eq 'Smoke') { $arguments+='--smoke' }
 $log=Join-Path $output ('console_'+(Get-Date -Format 'yyyyMMdd_HHmmss')+'.log')
 Write-Host "Cloud output: $output ; log: $log"
 $errlog=$log+'.stderr.log'
 Write-Host "Python warnings/errors: $errlog"
 $exitCode=Invoke-PythonLogged $arguments $log $errlog
 if ($exitCode -ne 0) {
  Get-Content -LiteralPath $errlog -Tail 45
  $audit=Join-Path $output 'report\raw_numeric_audit.json'
  if (Test-Path -LiteralPath $audit) {
   (Get-Content -LiteralPath $audit -Raw | ConvertFrom-Json).columns | Where-Object { $_.infinite -gt 0 -or $_.finite_float32_overflow -gt 0 } | ConvertTo-Json -Depth 6
  }
  throw "Python exit code $exitCode. Inspect $errlog. No successful performance claimed."
 }
 Get-Content -LiteralPath (Join-Path $output 'report\status.json')
} finally { if ($lock) { $lock.Dispose() } }
