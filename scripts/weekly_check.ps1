param(
    [string]$Python = 'python',
    [string[]]$Expected = @(),
    [switch]$SkipTests
)
$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $RepoRoot
try {
    if (-not $SkipTests) {
        & $Python -m unittest discover -s tests -v
        if ($LASTEXITCODE -ne 0) { throw 'Regression tests failed.' }
    }
    & $Python scripts/verify_runs.py --baseline
    if ($LASTEXITCODE -ne 0) { throw 'W2 baseline evidence is incomplete or inconsistent.' }
    git diff --check
    if ($LASTEXITCODE -ne 0) { throw 'Working-tree whitespace check failed.' }
    git diff --cached --check
    if ($LASTEXITCODE -ne 0) { throw 'Staged whitespace check failed.' }
    $Forbidden = @(git ls-files | Where-Object {
        $_ -match '(^|/)(tensorboard|checkpoints|datasets|\.venv|wandb)/|\.tfevents\.|\.(pt|pth|mp4)$'
    })
    if ($Forbidden.Count -gt 0) { throw ('Forbidden generated files are tracked: ' + ($Forbidden -join ', ')) }
    if ($Expected.Count -gt 0) {
        $Today = (Get-Date).Date
        $DaysSinceMonday = (([int]$Today.DayOfWeek + 6) % 7)
        $Monday = $Today.AddDays(-$DaysSinceMonday).ToString('yyyy-MM-ddT00:00:00')
        $Subjects = @(git log origin/main "--since=$Monday" '--format=%s')
        if ($LASTEXITCODE -ne 0) { throw 'Cannot read origin/main commit history.' }
        foreach ($StudentId in $Expected) {
            if ($StudentId -notmatch '^\d{9,12}$') { throw "Invalid student ID: $StudentId" }
            if (-not ($Subjects | Where-Object { $_ -match [regex]::Escape($StudentId) })) {
                throw "No pushed commit with student ID $StudentId in this weekly window."
            }
        }
    } else {
        Write-Host 'Student-ID participation audit not requested (pass -Expected to check it).'
    }
    Write-Host 'PASS: W2 evidence, repository hygiene and requested checks.'
} finally {
    Pop-Location
}
