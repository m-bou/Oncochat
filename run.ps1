<#
.SYNOPSIS
  OncoChat task runner for Windows. Same commands as run.sh (macOS / Linux / devcontainer); keep both in sync.

.DESCRIPTION
  Works with Windows PowerShell 5.1 (built into Windows 10/11) and PowerShell 7+.
  Keep this file ASCII-only: Windows PowerShell 5.1 reads BOM-less files as ANSI.

  If script execution is disabled on your machine, either run once:
      Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
  or call the script through:
      powershell -NoProfile -ExecutionPolicy Bypass -File .\run.ps1 <command>

.EXAMPLE
  .\run.ps1 up
  .\run.ps1 test -k guard
  .\run.ps1 eval --tier fast --repeat 3
#>

# Output goes through Write-Host on purpose: this is an interactive console runner, not a pipeline cmdlet.
# No param() block on purpose: every argument lands in $args untouched, so options meant for pytest or
# eval/run_eval.py (-k, --tier ...) are not interpreted by PowerShell itself.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot

$AppUrl = 'http://localhost:8000'
if ($args.Count -gt 0) { $Cmd = [string]$args[0] } else { $Cmd = 'help' }
if ($args.Count -gt 1) { $Rest = @($args[1..($args.Count - 1)]) } else { $Rest = @() }

function Fail([string]$Message) {
    Write-Host "error: $Message" -ForegroundColor Red
    exit 1
}

function Test-Command([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Assert-Uv {
    if (-not (Test-Command 'uv')) {
        Fail "'uv' not found. Open the project in the VS Code devcontainer, or install uv: https://docs.astral.sh/uv/"
    }
}

# Run a native program and stop with its exit code if it fails.
function Invoke-Native([string]$Exe, [object[]]$Arguments) {
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
}

function Invoke-Compose([object[]]$Arguments) {
    if (-not (Test-Command 'docker')) {
        Fail 'docker not found. Install Docker Desktop: https://docs.docker.com/get-docker/'
    }
    Invoke-Native 'docker' (@('compose') + $Arguments)
}

# Create .env from the template, never overwriting an existing one.
function Initialize-Env {
    if (Test-Path -LiteralPath '.env') { return }
    if (-not (Test-Path -LiteralPath '.env.example')) { Fail '.env.example not found' }
    Copy-Item -LiteralPath '.env.example' -Destination '.env'
    Write-Host 'created .env from .env.example (edit it to change models, threads, Ollama host...)'
}

function Show-Usage {
    @'
Usage: .\run.ps1 <command> [args...]

  init          Create .env from .env.example if it does not exist (never overwrites)
  up            Build and start everything in the background (first run downloads ~7 GB of models)
  down          Stop the stack (models and history are kept in Docker volumes)
  logs          Follow app and Ollama logs (Ctrl+C to stop)
  status        Show container status and the app health check

  dev           Run the API with auto-reload on :8000 (needs uv)
  test [args]   Offline test suite with a fake LLM (args go to pytest, e.g. -k guard)
  lint          ruff check
  fmt           ruff format + ruff check --fix
  eval [args]   Live evaluation against Ollama (args go to eval/run_eval.py, e.g. --tier fast --repeat 3)

  help          Show this message
'@ | Write-Host
}

switch ($Cmd) {
    'init' {
        Initialize-Env
    }
    'up' {
        Initialize-Env
        Invoke-Compose (@('up', '--build', '-d') + $Rest)
        Write-Host "Starting: $AppUrl  (first run: the model download can take several minutes; .\run.ps1 logs)"
    }
    'down' {
        Invoke-Compose (@('down') + $Rest)
    }
    'logs' {
        Invoke-Compose (@('logs', '-f', 'app', 'ollama') + $Rest)
    }
    'status' {
        Invoke-Compose @('ps')
        Write-Host ''
        try {
            Invoke-RestMethod -Uri "$AppUrl/api/health" -TimeoutSec 5 | ConvertTo-Json -Depth 5 | Write-Host
        } catch {
            Write-Host "app not reachable at $AppUrl"
        }
    }
    'dev' {
        Assert-Uv
        Invoke-Native 'uv' (@('run', 'uvicorn', 'app.main:app', '--host', '0.0.0.0', '--port', '8000', '--reload') + $Rest)
    }
    'test' {
        Assert-Uv
        Invoke-Native 'uv' (@('run', 'pytest', '-q') + $Rest)
    }
    'lint' {
        Assert-Uv
        Invoke-Native 'uv' (@('run', 'ruff', 'check', '.') + $Rest)
    }
    'fmt' {
        Assert-Uv
        Invoke-Native 'uv' @('run', 'ruff', 'format', '.')
        Invoke-Native 'uv' @('run', 'ruff', 'check', '.', '--fix')
    }
    'eval' {
        if (Test-Command 'uv') {
            Invoke-Native 'uv' (@('run', 'python', '-m', 'eval.run_eval') + $Rest)
        } else {
            Write-Host 'uv not found: running the eval inside the app container'
            Invoke-Compose (@('exec', 'app', 'python', '-m', 'eval.run_eval') + $Rest)
            # bring the JSON reports back to the host (never overwrite existing files)
            $tmp = Join-Path ([System.IO.Path]::GetTempPath()) ('oncochat-eval-' + [guid]::NewGuid())
            New-Item -ItemType Directory -Path $tmp | Out-Null
            try {
                Invoke-Compose @('cp', 'app:/app/eval/results/.', $tmp) | Out-Null
                Get-ChildItem -LiteralPath $tmp -Filter '*.json' | ForEach-Object {
                    $dest = Join-Path 'eval/results' $_.Name
                    if (-not (Test-Path -LiteralPath $dest)) { Copy-Item -LiteralPath $_.FullName -Destination $dest }
                }
                Write-Host 'reports copied to eval/results/'
            } finally {
                Remove-Item -LiteralPath $tmp -Recurse -Force
            }
        }
    }
    { $_ -in @('help', '-h', '--help', '/?') } {
        Show-Usage
    }
    default {
        Show-Usage
        Fail "unknown command '$Cmd'"
    }
}
