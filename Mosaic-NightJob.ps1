<#
.SYNOPSIS
    Build and update the Mosaic theologian library on the Windows machine.

.DESCRIPTION
    Crawls the public Mosaic site, downloads sermon audio, transcribes it on
    the GPU, and maintains a LanceDB index. Designed to run unattended at
    night and to survive being killed at any moment.

    Resume safety is deliberately NOT built on trapping Ctrl+C. Windows
    PowerShell can terminate without running finally blocks, and it takes the
    child python.exe with it. Instead every worker advances state only after
    its artifact is on disk, and artifacts are written to .part files and
    renamed atomically. A kill costs at most the single in-flight item.

.EXAMPLE
    .\Mosaic-NightJob.ps1 -Action EnsureDeps
    .\Mosaic-NightJob.ps1 -Action Run -Until 06:00
    .\Mosaic-NightJob.ps1 -Action Status
    .\Mosaic-NightJob.ps1 -Action Reindex
    .\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable -Full
#>

[CmdletBinding()]
param(
    [ValidateSet('EnsureDeps', 'Run', 'Status', 'Stop', 'Export', 'Reindex')]
    [string] $Action = 'Run',

    # Wall-clock stop, checked between items so work halts on a boundary.
    [string] $Until,
    [int]    $MaxMinutes,

    # Items per stage per cycle. The run keeps cycling until the work is done
    # or the deadline hits; this only sets how much each cycle bites off.
    [int]    $BatchSize = 20,

    # Safety stop for the cycle loop. 0 means "until done or out of time".
    [int]    $MaxCycles = 0,

    # After on-disk audio is drained, refresh listings before download/crawl.
    [switch] $DiscoverFirst,

    # Export only.
    [string] $Destination,
    [switch] $Full,

    # Re-do completed steps for matching items.
    [switch] $Force,

    # Add a single URL to the queue.
    [string] $Url,

    # Crawl and index page text but skip GPU transcription this run.
    [switch] $SkipTranscribe,

    # Limit how many messages the crawler discovers (useful for first run).
    [int]    $MaxPages = 0
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$Root      = $PSScriptRoot
$StateDir  = Join-Path $Root 'state'
$ConfigDir = Join-Path $Root 'config'
$LogPath   = Join-Path $StateDir 'run.log'
$DepsPath  = Join-Path $StateDir 'deps.json'
$LockPath  = Join-Path $StateDir 'night.lock'
$VenvDir   = Join-Path $Root '.venv'
$VenvPython = Join-Path $VenvDir 'Scripts\python.exe'

New-Item -ItemType Directory -Force -Path $StateDir | Out-Null

# ---------------------------------------------------------------- logging --

function Write-Log {
    param([string] $Message, [string] $Level = 'INFO')
    $line = '{0} [{1}] {2}' -f (Get-Date -Format 'yyyy-MM-dd HH:mm:ss'), $Level, $Message
    Write-Host $line
    Add-Content -Path $LogPath -Value $line -Encoding UTF8
}

function Write-Section {
    param([string] $Title)
    Write-Host ''
    Write-Host ('=' * 70) -ForegroundColor DarkGray
    Write-Host "  $Title" -ForegroundColor Cyan
    Write-Host ('=' * 70) -ForegroundColor DarkGray
}

# -------------------------------------------------------------- CUDA pin --

function Import-DotEnv {
    param([string] $Path = (Join-Path $Root '.env'))
    if (-not (Test-Path $Path)) { return }
    Get-Content -Path $Path -Encoding UTF8 | ForEach-Object {
        $line = $_.Trim()
        if (-not $line -or $line.StartsWith('#')) { return }
        $eq = $line.IndexOf('=')
        if ($eq -lt 1) { return }
        $name = $line.Substring(0, $eq).Trim()
        $value = $line.Substring($eq + 1).Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Find-CudaToolkit {
    param([string] $Version)
    $base = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA'
    if ($Version) {
        $candidate = Join-Path $base ("v{0}" -f $Version)
        if (Test-Path (Join-Path $candidate 'bin')) { return $candidate }
    }
    if (-not (Test-Path $base)) { return $null }
    $dirs = Get-ChildItem $base -Directory -ErrorAction SilentlyContinue |
        Sort-Object Name -Descending
    foreach ($dir in $dirs) {
        if ($dir.Name -like 'v12*' -and (Test-Path (Join-Path $dir.FullName 'bin'))) {
            return $dir.FullName
        }
    }
    if ($dirs) { return $dirs[0].FullName }
    return $null
}

function Initialize-CudaEnv {
    <#
        Pin one CUDA toolkit on PATH. Two toolkits (or toolkit + leftover
        CUDA 11 bins) is what makes faster-whisper fail to find cublas.
        .env is the source of truth: CUDA_VERSION and CUDA_PATH.
    #>
    $envFile = Join-Path $Root '.env'
    if (-not (Test-Path $envFile)) {
        $example = Join-Path $Root '.env.example'
        if (Test-Path $example) {
            Copy-Item $example $envFile
            Write-Log "Created .env from .env.example"
        }
    }

    Import-DotEnv

    $version = if ($env:CUDA_VERSION) { $env:CUDA_VERSION } else { '12.4' }
    $env:CUDA_VERSION = $version

    if (-not $env:CUDA_PATH -or -not (Test-Path $env:CUDA_PATH)) {
        $found = Find-CudaToolkit $version
        if ($found) { $env:CUDA_PATH = $found }
    }

    $kept = [System.Collections.Generic.List[string]]::new()
    foreach ($part in ($env:PATH -split ';')) {
        if (-not $part) { continue }
        if ($part -match 'NVIDIA GPU Computing Toolkit\\CUDA') { continue }
        $kept.Add($part)
    }

    $prepend = [System.Collections.Generic.List[string]]::new()
    if ($env:CUDA_PATH) {
        $bin = Join-Path $env:CUDA_PATH 'bin'
        if (Test-Path $bin) { $prepend.Add($bin) }
    }
    $nvidiaRoot = Join-Path $VenvDir 'Lib\site-packages\nvidia'
    foreach ($pkg in @('cublas', 'cudnn', 'cuda_runtime', 'cuda_nvrtc')) {
        $pkgBin = Join-Path $nvidiaRoot "$pkg\bin"
        if (Test-Path $pkgBin) { $prepend.Add($pkgBin) }
    }

    $env:PATH = (@($prepend) + @($kept)) -join ';'

    if ($env:CUDA_PATH -and (Test-Path $env:CUDA_PATH)) {
        Write-Log "CUDA pinned: version=$version  path=$($env:CUDA_PATH)"
    } else {
        Write-Log "CUDA_VERSION=$version (toolkit folder not found; using venv NVIDIA DLLs if present)" 'WARN'
    }
}

# ------------------------------------------------------------- dep record --

function Read-DepState {
    if (Test-Path $DepsPath) {
        try {
            $text = [System.IO.File]::ReadAllText($DepsPath, [System.Text.Encoding]::UTF8)
            return $text | ConvertFrom-Json
        } catch {
            Write-Log "deps.json unreadable, starting a fresh record: $($_.Exception.Message)" 'WARN'
        }
    }
    return [pscustomobject]@{ items = @() }
}

function Write-DepRecord {
    <#
        Records whether THIS script installed something or merely found it.
        Nothing is ever uninstalled; deps.json is an audit trail, not a
        package manager.
    #>
    param(
        [Parameter(Mandatory)] [string] $Name,
        [string] $Version = '',
        [string] $Source = '',
        [bool]   $Installed = $false,
        [bool]   $AlreadyPresent = $false,
        [hashtable] $Extra
    )

    $state = Read-DepState
    $items = @($state.items | Where-Object { $_.name -ne $Name })

    $record = [ordered]@{
        name            = $Name
        version         = $Version
        source          = $Source
        installedByUs   = $Installed
        alreadyPresent  = $AlreadyPresent
        recorded        = (Get-Date).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ')
    }
    if ($Extra) { foreach ($k in $Extra.Keys) { $record[$k] = $Extra[$k] } }

    $items += [pscustomobject]$record
    $payload = [pscustomobject]@{ items = $items } | ConvertTo-Json -Depth 6

    $tmp = "$DepsPath.part"
    [System.IO.File]::WriteAllText($tmp, $payload, (New-Object System.Text.UTF8Encoding($false)))
    Move-Item $tmp $DepsPath -Force
}

function Get-Settings {
    $path = Join-Path $ConfigDir 'settings.json'
    $text = [System.IO.File]::ReadAllText($path, [System.Text.Encoding]::UTF8)
    return $text | ConvertFrom-Json
}

function Test-CommandExists {
    param([string] $Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function ConvertTo-NativeArgument {
    <#
        Quote one argument for a Windows command line.

        Start-Process -ArgumentList joins an array with spaces and does not
        quote the elements, so any path containing a space is silently split
        into two arguments. Install under "C:\Mosaic LLM\..." and every probe
        breaks with a confusing "can't open file 'C:\Mosaic'".

        Follows the CommandLineToArgvW rules: double the backslashes that
        precede a quote, double a trailing backslash run, then wrap.
    #>
    param([string] $Value)

    if ($null -eq $Value -or $Value -eq '') { return '""' }
    if ($Value -notmatch '[ \t"]')          { return $Value }

    $escaped = [regex]::Replace($Value, '(\\*)"', '$1$1\"')
    $escaped = [regex]::Replace($escaped, '(\\+)$', '$1$1')
    return '"' + $escaped + '"'
}

function Invoke-Native {
    <#
        Run an external command without letting its stderr kill the script.

        $ErrorActionPreference = 'Stop' applies to native commands too: anything
        a child process writes to stderr becomes a terminating error. Plenty of
        well-behaved tools use stderr for warnings - pip announces upgrades,
        torch warns about NumPy, winget writes progress - so without this every
        one of those becomes a crash. Redirecting with 2>$null is not enough,
        because the error record is raised before the redirect applies.

        Returns a hashtable: ExitCode, StdOut, StdErr.
    #>
    param(
        [Parameter(Mandatory)] [string]   $FilePath,
        [string[]] $Arguments = @(),
        [switch]   $PassThruOutput   # also echo stdout to the console
    )

    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'

    $outFile = Join-Path $StateDir ("native-{0}.out" -f [guid]::NewGuid().ToString('N'))
    $errFile = "$outFile.err"

    try {
        $startArgs = @{
            FilePath               = $FilePath
            NoNewWindow            = $true
            Wait                   = $true
            PassThru               = $true
            RedirectStandardOutput = $outFile
            RedirectStandardError  = $errFile
        }
        if ($Arguments.Count -gt 0) {
            # One pre-quoted string, not the raw array: see ConvertTo-NativeArgument.
            $startArgs.ArgumentList =
                ($Arguments | ForEach-Object { ConvertTo-NativeArgument $_ }) -join ' '
        }

        $proc = Start-Process @startArgs

        $stdout = if (Test-Path $outFile) {
            [System.IO.File]::ReadAllText($outFile, [System.Text.Encoding]::UTF8) } else { '' }
        $stderr = if (Test-Path $errFile) {
            [System.IO.File]::ReadAllText($errFile, [System.Text.Encoding]::UTF8) } else { '' }

        if ($PassThruOutput -and $stdout.Trim()) { Write-Host $stdout.TrimEnd() }

        return @{
            ExitCode = $proc.ExitCode
            StdOut   = $stdout.Trim()
            StdErr   = $stderr.Trim()
        }
    }
    finally {
        Remove-Item $outFile, $errFile -Force -ErrorAction SilentlyContinue
        $ErrorActionPreference = $previous
    }
}

function Invoke-PipInstall {
    param([string[]] $PipArgs, [string] $Label)

    $result = Invoke-Native -FilePath $VenvPython -Arguments (@('-m', 'pip', 'install') + $PipArgs)
    if ($result.ExitCode -ne 0) {
        Write-Log "pip install failed for $Label (exit $($result.ExitCode))" 'WARN'
        $detail = ($result.StdErr -split "`n" | Select-Object -Last 4) -join ' '
        if ($detail) { Write-Log "  $detail" 'WARN' }
        return $false
    }
    return $true
}

# ------------------------------------------------------------ dependencies --

function Install-WithWinget {
    param([string] $Id, [string] $Command, [string] $Friendly)

    if (Test-CommandExists $Command) {
        # Some tools (ffmpeg among them) print their banner on stderr, so read
        # both streams before deciding the version string is empty.
        $probe = Invoke-Native -FilePath $Command -Arguments @('--version')
        $text = if ($probe.StdOut) { $probe.StdOut } else { $probe.StdErr }
        $version = ($text -split "`n" | Select-Object -First 1).Trim()
        Write-Log "$Friendly already present ($version)"
        Write-DepRecord -Name $Friendly -Version "$version" -Source 'pre-existing' -AlreadyPresent $true
        return $true
    }

    if (-not (Test-CommandExists 'winget')) {
        Write-Log "$Friendly missing and winget is unavailable. Install $Friendly manually." 'ERROR'
        return $false
    }

    Write-Log "Installing $Friendly via winget ($Id)"
    Invoke-Native -FilePath 'winget' -Arguments @(
        'install', '--id', $Id,
        '--accept-source-agreements', '--accept-package-agreements', '--silent'
    ) | Out-Null

    # winget updates PATH for new processes, not this one.
    $env:Path = [System.Environment]::GetEnvironmentVariable('Path', 'Machine') + ';' +
                [System.Environment]::GetEnvironmentVariable('Path', 'User')

    if (Test-CommandExists $Command) {
        Write-Log "$Friendly installed"
        Write-DepRecord -Name $Friendly -Source "winget:$Id" -Installed $true
        return $true
    }

    Write-Log "$Friendly still not on PATH. Open a new shell and re-run EnsureDeps." 'WARN'
    return $false
}

function Test-Torch {
    <#
        Report torch version and whether it can see the GPU.

        NumPy has to be importable before this runs. Without it torch still
        imports but warns on stderr, and that warning is enough to abort the
        script when native stderr is treated as an error.
    #>
    $code = 'import sys' + "`n" +
            'try:' + "`n" +
            '    import torch' + "`n" +
            '    sys.stdout.write(torch.__version__ + "|" + str(torch.cuda.is_available()))' + "`n" +
            'except Exception as exc:' + "`n" +
            '    sys.stdout.write("ABSENT|" + type(exc).__name__)' + "`n"

    $result = Invoke-Native -FilePath $VenvPython -Arguments @('-W', 'ignore', '-c', $code)
    return $result.StdOut
}

function Initialize-PythonEnv {
    if (-not (Test-Path $VenvPython)) {
        Write-Log 'Creating virtual environment (.venv)'
        $venv = Invoke-Native -FilePath 'python' -Arguments @('-m', 'venv', $VenvDir)
        if ($venv.ExitCode -ne 0) {
            throw "Could not create the virtual environment: $($venv.StdErr)"
        }
        Write-DepRecord -Name 'venv' -Source '.venv' -Installed $true
    } else {
        Write-Log 'Virtual environment already present'
    }

    Invoke-PipInstall -PipArgs @('--upgrade', 'pip', '--quiet') -Label 'pip' | Out-Null

    # NumPy first, before anything imports torch. torch degrades to a stderr
    # warning when NumPy is missing, and that warning aborts the run.
    Write-Log 'Installing numpy'
    if (-not (Invoke-PipInstall -PipArgs @('numpy>=1.26', '--quiet') -Label 'numpy')) {
        throw 'numpy is required before torch can be installed.'
    }

    # torch must come from a CUDA index. A plain "pip install torch" pulls a
    # CPU-only wheel and the embedding pass then runs an order slower.
    $probe = Test-Torch

    if ($probe -match '\|True$') {
        Write-Log "torch with CUDA already present ($probe)"
        Write-DepRecord -Name 'torch' -Version "$probe" -Source 'pre-existing' -AlreadyPresent $true
    }
    else {
        $indexes = @(
            @{ Name = 'cu124'; Url = 'https://download.pytorch.org/whl/cu124' },
            @{ Name = 'cu121'; Url = 'https://download.pytorch.org/whl/cu121' }
        )

        $installed = $false
        foreach ($index in $indexes) {
            Write-Log "Installing torch ($($index.Name) build); this download is large"
            if (Invoke-PipInstall -Label "torch $($index.Name)" `
                    -PipArgs @('torch', '--index-url', $index.Url, '--quiet')) {
                $probe = Test-Torch
                Write-Log "torch reports: $probe"
                if ($probe -match '\|True$') {
                    Write-DepRecord -Name 'torch' -Version "$probe" `
                        -Source "pytorch $($index.Name) index" -Installed $true `
                        -Extra @{ cuda = $true }
                    $installed = $true
                    break
                }
                Write-Log "torch installed from $($index.Name) but cannot see the GPU." 'WARN'
            }
        }

        if (-not $installed) {
            Write-Log 'Falling back to the default (CPU) torch wheel.' 'WARN'
            Invoke-PipInstall -PipArgs @('torch', '--quiet') -Label 'torch cpu' | Out-Null
            $probe = Test-Torch
            Write-Log "torch reports: $probe"
            Write-DepRecord -Name 'torch' -Version "$probe" -Source 'pypi (cpu)' `
                -Installed $true -Extra @{ cuda = $false }
            Write-Log 'Embedding will run on CPU. Indexing still works, just slower.' 'WARN'
        }
    }

    # sentence-transformers depends on torch. Installing it after torch means
    # pip sees the requirement satisfied and leaves the CUDA build in place.
    Write-Log 'Installing Python requirements'
    if (Invoke-PipInstall -Label 'requirements' `
            -PipArgs @('-r', (Join-Path $Root 'requirements-windows.txt'), '--quiet')) {
        Write-DepRecord -Name 'python-requirements' -Source 'requirements-windows.txt' -Installed $true
    } else {
        throw 'Could not install the Python requirements. See the messages above.'
    }

    # Confirm torch survived the requirements install.
    $final = Test-Torch
    if ($final -notmatch '\|True$') {
        Write-Log "torch after requirements: $final (GPU not available)" 'WARN'
    }
}

function Test-WhisperGpu {
    <#
        faster-whisper runs on CTranslate2, not torch, and needs cuBLAS and
        cuDNN on PATH. When they are missing it does not raise - it silently
        falls back to CPU, which turns a week of nights into two months. So
        actually construct the model on cuda rather than trusting the import.
    #>
    Write-Log 'Verifying faster-whisper GPU mode'

    $probe = @'
import sys
try:
    from faster_whisper import WhisperModel
    WhisperModel("tiny", device="cuda", compute_type="float16")
    sys.stdout.write("GPU_OK")
except Exception as exc:
    sys.stdout.write("GPU_FAIL:" + type(exc).__name__ + ":" + str(exc)[:200])
'@

    $probeFile = Join-Path $StateDir 'gpu_probe.py'
    [System.IO.File]::WriteAllText($probeFile, $probe, (New-Object System.Text.UTF8Encoding($false)))
    $run = Invoke-Native -FilePath $VenvPython -Arguments @('-W', 'ignore', $probeFile)
    Remove-Item $probeFile -ErrorAction SilentlyContinue
    $result = "$($run.StdOut) $($run.StdErr)"

    if ($result -match 'GPU_OK') {
        Write-Log 'faster-whisper GPU mode confirmed'
        Write-DepRecord -Name 'whisper-gpu' -Source 'ctranslate2/cuda' -Installed $false `
            -Extra @{ gpuMode = $true }
        return $true
    }

    Write-Log "faster-whisper is NOT using the GPU: $($result.Trim())" 'WARN'
    Write-Log 'Usual cause is missing cuDNN/cuBLAS DLLs on PATH. Transcription will be very slow.' 'WARN'
    Write-DepRecord -Name 'whisper-gpu' -Source 'ctranslate2/cuda' -Installed $false `
        -Extra @{ gpuMode = $false; detail = $result.Trim() }
    return $false
}

function Resolve-BibleSource {
    <#
        Download the Bible JSON once, and validate it by BOOK COUNT.

        A "does Genesis exist" check is not enough: a truncated copy of this
        exact file starts at Genesis and stops in Leviticus, which would give
        an index that looks fine and answers nothing for the New Testament.
    #>
    $settings = Get-Settings
    $localRel = $settings.Bible.LocalFile
    $local    = Join-Path $Root ($localRel -replace '/', '\')
    $expected = [int]$settings.Bible.ExpectedBookCount

    if (Test-Path $local) {
        Write-Log "Bible source already present: $localRel"
        return $local
    }

    $url = $settings.Bible.SourceUrl
    if ([string]::IsNullOrWhiteSpace($url)) {
        Write-Log 'No Bible.SourceUrl configured; scripture collection will be empty.' 'WARN'
        return $null
    }

    New-Item -ItemType Directory -Force -Path (Split-Path $local) | Out-Null
    $tmp = "$local.part"

    try {
        Write-Log "Downloading Bible JSON from $url"
        $previous = $ProgressPreference
        $ProgressPreference = 'SilentlyContinue'   # keeps large downloads fast
        Invoke-WebRequest -Uri $url -OutFile $tmp -UseBasicParsing -ErrorAction Stop
        $ProgressPreference = $previous

        # ReadAllText with explicit UTF8. Get-Content -Raw on PS 5.1 defaults to
        # the ANSI code page and mangles the curly quotes throughout this text.
        $text  = [System.IO.File]::ReadAllText($tmp, [System.Text.Encoding]::UTF8)
        $probe = $text | ConvertFrom-Json
        $books = @($probe.PSObject.Properties.Name)

        if ($books.Count -lt $expected) {
            throw "Expected $expected books, found $($books.Count). Source file is partial."
        }

        Move-Item $tmp $local -Force
        $sizeMb = [math]::Round((Get-Item $local).Length / 1MB, 1)
        Write-Log "Bible JSON validated: $($books.Count) books, $sizeMb MB"
        Write-DepRecord -Name 'bible-json' -Source $url -Installed $true `
            -Extra @{ bookCount = $books.Count; sizeMb = $sizeMb }
        return $local
    }
    catch {
        Remove-Item $tmp -Force -ErrorAction SilentlyContinue
        Write-Log "Bible download failed: $($_.Exception.Message)" 'WARN'
        Write-DepRecord -Name 'bible-json' -Source $url -Installed $false `
            -Extra @{ error = $_.Exception.Message }

        $fallback = $settings.Bible.FallbackUrl
        if (-not [string]::IsNullOrWhiteSpace($fallback)) {
            Write-Log "Trying public-domain fallback: $fallback"
            try {
                Invoke-WebRequest -Uri $fallback -OutFile $tmp -UseBasicParsing -ErrorAction Stop
                Move-Item $tmp $local -Force
                Write-Log 'Fallback translation downloaded'
                Write-DepRecord -Name 'bible-json' -Source $fallback -Installed $true `
                    -Extra @{ fallback = $true }
                return $local
            } catch {
                Remove-Item $tmp -Force -ErrorAction SilentlyContinue
                Write-Log "Fallback also failed: $($_.Exception.Message)" 'ERROR'
            }
        }
        return $null
    }
}

function Get-ChatModelHint {
    $settings = Get-Settings
    $modelPath = Join-Path $Root ($settings.Chat.ModelFile -replace '/', '\')
    if (Test-Path $modelPath) {
        $sizeGb = [math]::Round((Get-Item $modelPath).Length / 1GB, 2)
        Write-Log "Chat model present: $($settings.Chat.ModelFile) ($sizeGb GB)"
        Write-DepRecord -Name 'chat-model' -Source $settings.Chat.ModelFile -AlreadyPresent $true `
            -Extra @{ sizeGb = $sizeGb }
        return
    }

    Write-Log 'Chat model not found. Download a GGUF and place it at the configured path.' 'WARN'
    Write-Log "  expected: $($settings.Chat.ModelFile)" 'WARN'
    Write-Log '  suggested: Llama 3.1 8B Instruct Q5_K_M (portable to the Mac)' 'WARN'
    Write-Log '  the crawl/transcribe/index pipeline runs fine without it; only chat needs it.' 'WARN'
    Write-DepRecord -Name 'chat-model' -Source $settings.Chat.ModelFile -Installed $false `
        -Extra @{ present = $false }
}

# ------------------------------------------------------------------ python --

function Invoke-Worker {
    <#
        Run a worker module in the venv. Workers own their own state writes,
        so a non-zero exit only means this batch stopped, never that state is
        inconsistent.
    #>
    param(
        [Parameter(Mandatory)] [string] $Module,
        [string[]] $Arguments = @()
    )

    if (-not (Test-Path $VenvPython)) {
        throw "Virtual environment missing. Run: .\Mosaic-NightJob.ps1 -Action EnsureDeps"
    }

    # -X utf8 because sermon titles and verse text are full of curly quotes and
    # dashes. On a cp1252 console, printing one of those raises
    # UnicodeEncodeError and takes the worker down mid-batch.
    $argv = @('-X', 'utf8', '-m', $Module) + $Arguments
    Write-Verbose ("python " + ($argv -join ' '))

    # Workers stream progress for hours. 'Continue' stops an ordinary library
    # warning on stderr from aborting the whole night job.
    #
    # Stdout has to be shown with Out-Host, not left as function output.
    # Callers do `$code = Invoke-Worker`. In PowerShell that assignment
    # collects every printed line plus the exit code into one array, and
    # `$code -ne 0` then filters the array instead of testing the exit code.
    # A successful transcribe batch (exit 0, with progress lines) looks like
    # failure, and the night stops. Out-Host keeps the lines on screen and
    # leaves the return value as the integer exit code.
    #
    # python -m looks up workers from the process working directory. A
    # scheduled task or a wrapper started from another folder would otherwise
    # report "No module named workers".
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    Push-Location $Root
    try {
        & $VenvPython @argv | Out-Host
        $code = $LASTEXITCODE
    }
    finally {
        Pop-Location
        $ErrorActionPreference = $previous
    }

    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) {
        Write-Log "$Module exited with code $code" 'WARN'
    }
    return $code
}

function Get-DeadlineArgs {
    $deadline = @()
    if ($Until)      { $deadline += @('--until', $Until) }
    if ($MaxMinutes) { $deadline += @('--max-minutes', "$MaxMinutes") }
    return $deadline
}

function Get-StopTime {
    <#
        The workers each honour the deadline internally, but the loop needs
        its own copy of it. Otherwise it would happily start another cycle of
        workers that all immediately exit, and spin until the batch counter
        ran out.
    #>
    $stop = $null

    if ($MaxMinutes) { $stop = (Get-Date).AddMinutes($MaxMinutes) }

    if ($Until) {
        $parts = $Until -split ':'
        $hours = [int]$parts[0]
        $minutes = if ($parts.Count -gt 1) { [int]$parts[1] } else { 0 }
        $candidate = (Get-Date).Date.AddHours($hours).AddMinutes($minutes)
        if ($candidate -le (Get-Date)) { $candidate = $candidate.AddDays(1) }
        if ($null -eq $stop -or $candidate -lt $stop) { $stop = $candidate }
    }

    return $stop
}

function Test-OutOfTime {
    param($StopTime)
    if ($null -eq $StopTime) { return $false }
    return (Get-Date) -ge $StopTime
}

function Get-WorkLeft {
    <#
        Ask the Python side what is still pending. Returns $null if the
        question could not be answered, which the caller treats as "stop"
        rather than guessing.
    #>
    $result = Invoke-Native -FilePath $VenvPython `
        -Arguments @('-X', 'utf8', '-m', 'workers.worklist')

    if ($result.ExitCode -ne 0 -or -not $result.StdOut) {
        Write-Log "Could not read the work list: $($result.StdErr)" 'WARN'
        return $null
    }

    try {
        return $result.StdOut | ConvertFrom-Json
    } catch {
        Write-Log "Work list was not valid JSON: $($result.StdOut)" 'WARN'
        return $null
    }
}

function Format-WorkLeft {
    param($Work)
    $next = if ($Work.PSObject.Properties['next_action']) { $Work.next_action } else { '?' }
    $audioDisk = if ($Work.PSObject.Properties['audio_on_disk']) { $Work.audio_on_disk } else { '?' }
    $txDisk = if ($Work.PSObject.Properties['transcripts_on_disk']) { $Work.transcripts_on_disk } else { '?' }
    return ('next={0}  crawl={1}  download={2}  transcribe={3}  index={4}  | on disk: audio={5} transcripts={6}' -f
        $next, $Work.queued_messages, $Work.need_audio, $Work.need_transcribe,
        $Work.need_index, $audioDisk, $txDisk)
}

function Write-WorkBoard {
    param($Work, [string] $Title = 'Queue')
    if ($null -eq $Work) { return }
    Write-Section $Title
    Write-Log (Format-WorkLeft $Work)
    switch ("$($Work.next_action)") {
        'transcribe'     { Write-Log "Next: transcribe $($Work.need_transcribe) file(s) already on disk. No crawl, no new downloads." }
        'download'       { Write-Log "Next: download $($Work.need_audio) queued audio file(s). Transcribe queue is empty." }
        'crawl_messages' { Write-Log "Next: fetch $($Work.queued_messages) message page(s)." }
        'discover'       { Write-Log "Next: refresh listing pages for new messages." }
        'index'          { Write-Log "Next: index $($Work.need_index) finished transcript(s)." }
        'bible'          { Write-Log 'Next: index Scripture.' }
        'idle'           { Write-Log 'Next: nothing left.' }
        default          { }
    }
}

function Invoke-TranscribeDrain {
    <#
        Finish every audio file already on disk before touching crawl or
        download. batch-size 0 means the worker keeps going until the
        queue is empty or the deadline hits.
    #>
    param([string[]] $Deadline, [string[]] $Common)
    Write-Section 'NOW: transcribe audio already on disk'
    Write-Log 'Crawl and new downloads wait until this queue is empty.'
    $code = Invoke-Worker -Module 'workers.transcribe' `
        -Arguments (@('--batch-size', '0') + $Deadline + $Common)
    if ($code -ne 0) {
        Write-Log "Transcription exited with code $code. Crawl and download stay paused until the next run." 'ERROR'
        Write-Log 'The worker lines above are the actual failure.' 'ERROR'
        return $false
    }
    return $true
}

function Invoke-IndexBestEffort {
    <#
        Index is best-effort. A failure must not block crawl or download;
        transcripts stay queued and are retried on a later cycle or run.
    #>
    param([string[]] $Deadline, [string[]] $Common, [string] $Label = 'index')
    Write-Section "NOW: $Label"
    $code = Invoke-Worker -Module 'workers.index_build' `
        -Arguments ($Deadline + $Common)
    if ($code -ne 0) {
        Write-Log 'Indexing failed this pass; continuing with crawl/download. Transcripts stay queued for index.' 'WARN'
        return $false
    }
    return $true
}

function Invoke-DiscoveryPass {
    <#
        Refresh listing pages. Returns $true only when crawl exited cleanly
        or the queued frontier grew, so a failed seed is not treated as done.
    #>
    param(
        [string[]] $Deadline,
        [string[]] $Common,
        [string] $Reason = 'listings only'
    )
    $before = Get-WorkLeft
    $beforeMsg  = if ($before) { [int]$before.queued_messages } else { 0 }
    $beforeList = if ($before) { [int]$before.queued_listings } else { 0 }

    Write-Section "NOW: discovery ($Reason)"
    $crawlArgs = @('--refresh-listings', '--listings-only',
                   '--batch-size', '0') + $Deadline + $Common
    $code = Invoke-Worker -Module 'workers.crawl' -Arguments $crawlArgs

    $after = Get-WorkLeft
    Write-WorkBoard $after 'After discovery'
    if ($null -eq $after) { return $false }

    $grew = ([int]$after.queued_messages -gt $beforeMsg) -or
            ([int]$after.queued_listings -gt $beforeList)
    if ($code -eq 0 -or $grew) {
        return $true
    }
    Write-Log 'Discovery did not succeed; will retry on a later cycle.' 'WARN'
    return $false
}

function Get-WorkSignature {
    <#
        Frontier signature for stall detection. need_index is included so a
        successful index pass resets the counter, but callers must not count
        an index-only no-op as a stall while crawl/download work remains.
    #>
    param($Work)
    return ('{0}|{1}|{2}|{3}|{4}' -f
        $Work.queued_listings, $Work.queued_messages,
        $Work.need_audio, $Work.need_transcribe, $Work.need_index)
}

function Test-FrontierRemaining {
    param($Work, [bool] $DidDiscover)
    if ($null -eq $Work) { return $false }
    return ([int]$Work.need_audio -gt 0) -or
           ([int]$Work.queued_messages -gt 0) -or
           (-not $DidDiscover)
}

# ------------------------------------------------------------------- lock --

function Test-LockHeld {
    if (-not (Test-Path $LockPath)) { return $false }
    try {
        $info = [System.IO.File]::ReadAllText($LockPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        $pidValue = [int]$info.pid
    } catch {
        return $false
    }

    $alive = $null -ne (Get-Process -Id $pidValue -ErrorAction SilentlyContinue)
    if (-not $alive) {
        # Stale lock from a hard kill. Reclaim it rather than blocking forever.
        Write-Log "Clearing stale lock from PID $pidValue"
        Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
        return $false
    }
    return $true
}

# ---------------------------------------------------------------- actions --

function Invoke-EnsureDeps {
    Write-Section 'Dependency bootstrap'
    Initialize-CudaEnv

    Install-WithWinget -Id 'Python.Python.3.12' -Command 'python' -Friendly 'Python'   | Out-Null
    Install-WithWinget -Id 'Gyan.FFmpeg'        -Command 'ffmpeg' -Friendly 'ffmpeg'   | Out-Null
    Install-WithWinget -Id 'Git.Git'            -Command 'git'    -Friendly 'git'      | Out-Null

    Initialize-PythonEnv
    Initialize-CudaEnv
    Test-WhisperGpu | Out-Null
    Resolve-BibleSource | Out-Null
    Get-ChatModelHint

    # Snapshot the embedding model locally so Export can ship it to the Mac.
    Write-Log 'Caching embedding model locally for export'
    Invoke-Worker -Module 'workers.prepare_embedding' | Out-Null

    Write-Section 'Dependency summary'
    $state = Read-DepState
    $state.items | Format-Table name, version, installedByUs, alreadyPresent -AutoSize | Out-String | Write-Host
    Write-Log "Audit trail written to $DepsPath"
}

function Invoke-Run {
    if (Test-LockHeld) {
        Write-Log 'Another run is active. Exiting.' 'WARN'
        return 1
    }

    Write-Section 'Nightly run'
    Initialize-CudaEnv
    if ($Until)      { Write-Log "Will stop at $Until" }
    if ($MaxMinutes) { Write-Log "Will stop after $MaxMinutes minutes" }

    $deadline = Get-DeadlineArgs
    $stopTime = Get-StopTime
    $common = @()
    if ($Force) { $common += '--force' }

    # Stray .part files, jobs that ran ahead of artifacts, and MP3s on
    # disk that never got attached to a job.
    Invoke-Worker -Module 'workers.reconcile' | Out-Null

    Invoke-Worker -Module 'workers.ingest_bible' -Arguments $common | Out-Null

    $work = Get-WorkLeft
    Write-WorkBoard $work 'What is queued right now'

    if ($Url) {
        Write-Log "Queueing single URL: $Url"
        Invoke-Worker -Module 'workers.crawl' `
            -Arguments (@('--single', $Url) + $common) | Out-Null
    }

    # ------------------------------------------------------------------
    # Existing work first. If audio is already on disk, transcribe it
    # before any crawl or any new download. That is what makes a restart
    # pick up the ~30 files sitting in data\audio instead of walking the
    # archive again for twenty minutes.
    #
    # Priority after the on-disk drain:
    #   download -> crawl -> discover -> index
    # Index is last and best-effort so a sticky index failure cannot starve
    # crawl/download for the rest of the night.
    # ------------------------------------------------------------------
    $didDiscover = $false
    $cycle = 0
    $lastSignature = ''
    $stalled = 0
    $lastAction = ''

    while ($true) {
        if (Test-OutOfTime $stopTime) {
            Write-Log "Reached the $Until stop time. Remaining work resumes next run."
            break
        }
        if ($MaxCycles -gt 0 -and $cycle -ge $MaxCycles) {
            Write-Log "Reached -MaxCycles $MaxCycles."
            break
        }

        $work = Get-WorkLeft
        if ($null -eq $work) {
            Write-Log 'Stopping: could not determine remaining work.' 'WARN'
            break
        }

        $needTx    = [int]$work.need_transcribe
        $needIdx   = [int]$work.need_index
        $needDl    = [int]$work.need_audio
        $needCrawl = [int]$work.queued_messages
        $needList  = [int]$work.queued_listings
        # Under -SkipTranscribe, on-disk audio is intentionally left alone
        # and must not keep the idle check from finishing other work.
        $txBlocksIdle = (-not $SkipTranscribe -and $needTx -gt 0)

        if (-not $txBlocksIdle -and $needIdx -le 0 -and $needDl -le 0 -and
            $needCrawl -le 0 -and ($didDiscover -or $needList -le 0) -and
            [int]$work.bible_pending -le 0 -and -not $Url) {
            if (-not $didDiscover -and -not $Url) {
                # No backlog. Look for newly published messages once.
                if (Invoke-DiscoveryPass -Deadline $deadline -Common $common `
                        -Reason 'listings only; backlog was empty') {
                    $didDiscover = $true
                    $stalled = 0
                } else {
                    $stalled++
                    if ($stalled -ge 2) {
                        Write-Log 'Discovery failed twice with an empty backlog; stopping.' 'WARN'
                        break
                    }
                }
                continue
            }
            if ($SkipTranscribe -and $needTx -gt 0) {
                Write-Log 'Non-GPU work is done. Audio remains on disk (-SkipTranscribe).'
            } else {
                Write-Log 'Everything discovered is downloaded, transcribed, and indexed.'
            }
            break
        }

        $cycle++
        Write-Log ''
        Write-Log "--- cycle $cycle | $(Format-WorkLeft $work) ---"
        $lastAction = ''

        # 1. Transcribe everything already on disk. Do not crawl or download.
        if (-not $SkipTranscribe -and $needTx -gt 0) {
            if (-not (Invoke-TranscribeDrain -Deadline $deadline -Common $common)) {
                break
            }
            Invoke-IndexBestEffort -Deadline $deadline -Common $common `
                -Label 'index after transcription' | Out-Null
            Write-WorkBoard (Get-WorkLeft) 'After transcription'
            $lastAction = 'transcribe'
        }
        # 2. Optional: refresh listings before processing new pages.
        elseif ($DiscoverFirst -and -not $didDiscover -and -not $Url) {
            if (Invoke-DiscoveryPass -Deadline $deadline -Common $common `
                    -Reason 'listings only; -DiscoverFirst after on-disk drain') {
                $didDiscover = $true
            }
            $lastAction = 'discover'
        }
        # 3. Download queued audio once the on-disk transcribe queue is clear.
        elseif ($needDl -gt 0) {
            Write-Section "NOW: download $($needDl) queued audio file(s)"
            Write-Log 'Transcribe queue is empty, so downloads may proceed.'
            Invoke-Worker -Module 'workers.download_audio' `
                -Arguments (@('--batch-size', "$BatchSize") + $deadline + $common) | Out-Null
            $lastAction = 'download'
        }
        # 4. Fetch message pages that still need crawling.
        elseif ($needCrawl -gt 0 -and -not $Url) {
            Write-Section "NOW: fetch $needCrawl message page(s)"
            $batchCrawl = @('--messages-only', '--batch-size', "$BatchSize") +
                          $deadline + $common
            if ($MaxPages -gt 0) { $batchCrawl += @('--max-pages', "$MaxPages") }
            Invoke-Worker -Module 'workers.crawl' -Arguments $batchCrawl | Out-Null
            $lastAction = 'crawl'
        }
        # 5. Discover listings when the processing frontier is empty.
        elseif (-not $didDiscover -and -not $Url) {
            if (Invoke-DiscoveryPass -Deadline $deadline -Common $common `
                    -Reason 'listings only') {
                $didDiscover = $true
            }
            $lastAction = 'discover'
        }
        # 6. Index last — never blocks crawl/download/discover.
        elseif ($needIdx -gt 0) {
            Invoke-IndexBestEffort -Deadline $deadline -Common $common `
                -Label "index $($needIdx) item(s)" | Out-Null
            $lastAction = 'index'
        }
        elseif ($SkipTranscribe -and $needTx -gt 0) {
            Write-Log 'Skipping transcription (-SkipTranscribe). Audio stays on disk; nothing else queued.'
            break
        }
        else {
            Write-Log 'No matching action for the current queue; stopping.' 'WARN'
            break
        }

        $after = Get-WorkLeft
        if ($null -eq $after) { break }
        $signature = Get-WorkSignature $after
        if ($signature -eq $lastSignature) {
            # Sticky index must not end the night while crawl/download remain.
            if ($lastAction -eq 'index' -and (Test-FrontierRemaining $after $didDiscover)) {
                Write-Log 'Index made no progress; continuing with remaining crawl/download work.' 'WARN'
            } else {
                $stalled++
            }
        } else {
            $stalled = 0
        }
        $lastSignature = $signature
        if ($stalled -ge 2) {
            Write-Log 'No progress across two consecutive cycles; stopping.' 'WARN'
            Write-Log 'Check Status for failed items; they are retried on the next run.' 'WARN'
            break
        }
    }

    Write-Section "Run complete after $cycle cycle(s)"
    Invoke-Status
    return 0
}

function Invoke-Status {
    Write-Section 'Status'
    Invoke-Worker -Module 'workers.status' | Out-Null

    if (Test-Path $LockPath) {
        if (Test-LockHeld) {
            $info = [System.IO.File]::ReadAllText($LockPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
            Write-Host "Run in progress (PID $($info.pid), started $($info.started))" -ForegroundColor Yellow
        }
    } else {
        Write-Host 'No run in progress.' -ForegroundColor DarkGray
    }
}

function Invoke-Stop {
    Write-Section 'Stop'

    if (-not (Test-Path $LockPath)) {
        Write-Log 'Nothing running.'
        return 0
    }

    try {
        $info = [System.IO.File]::ReadAllText($LockPath, [System.Text.Encoding]::UTF8) | ConvertFrom-Json
        $pidValue = [int]$info.pid
    } catch {
        Remove-Item $LockPath -Force
        Write-Log 'Removed unreadable lock file.'
        return 0
    }

    $proc = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    if ($proc) {
        Write-Log "Stopping PID $pidValue"
        Stop-Process -Id $pidValue -Force
        Start-Sleep -Seconds 2
    }

    Remove-Item $LockPath -Force -ErrorAction SilentlyContinue
    Write-Log 'Stopped. State is intact; the next run resumes from the last finished item.'
    return 0
}

function Invoke-Reindex {
    <#
        Rebuild the LanceDB index from the Bible file and transcripts already
        on disk. Does not crawl, download, or transcribe. The live index is
        replaced only when the rebuild finishes.
    #>
    if (Test-LockHeld) {
        Write-Log 'Another run is active. Exiting.' 'WARN'
        return 1
    }

    Write-Section 'Rebuild the index'
    Write-Log 'Reading the Bible file and transcripts already on disk.'
    Write-Log 'No crawl, no download, no transcription.'
    Write-Log 'The live index is replaced only if the rebuild finishes.'
    if ($Until)      { Write-Log "Will stop at $Until and leave the live index in place." }
    if ($MaxMinutes) { Write-Log "Will stop after $MaxMinutes minutes and leave the live index in place." }

    $code = Invoke-Worker -Module 'workers.index_build' `
        -Arguments (@('--rebuild') + (Get-DeadlineArgs))
    if ($code -eq 0) {
        Write-Log 'Index rebuilt. Export again to refresh the Mac bundle.'
    } else {
        Write-Log 'Rebuild did not finish. The previous index is unchanged.' 'WARN'
    }
    return $code
}

function Invoke-Export {
    Write-Section 'Export portable bundle'

    if (-not $Destination) {
        throw 'Export needs -Destination, for example: -Destination E:\mosaic-portable'
    }

    $args = @('--destination', $Destination)
    if ($Full) {
        $args += '--full'
        Write-Log 'Full export: index, content, and models'
        Write-Log 'If models\chat.gguf is missing, it is downloaded before the bundle is copied.'
    } else {
        $args += '--index-only'
        Write-Log 'Incremental export: index and content only (models unchanged)'
    }

    $code = Invoke-Worker -Module 'workers.export_bundle' -Arguments $args
    if ($code -eq 0) {
        Write-Log "Bundle written to $Destination"
        Write-Log 'On the Mac: ./mosaic.sh verify   then   ./mosaic.sh start'
    }
    return $code
}

# ------------------------------------------------------------------- main --

try {
    switch ($Action) {
        'EnsureDeps' { Invoke-EnsureDeps }
        'Run'        { Invoke-Run | Out-Null }
        'Status'     { Invoke-Status }
        'Stop'       { Invoke-Stop | Out-Null }
        'Export'     { Invoke-Export | Out-Null }
        'Reindex'    { Invoke-Reindex | Out-Null }
    }
}
catch {
    Write-Log $_.Exception.Message 'ERROR'
    Write-Log 'State is append-only; re-running continues from the last finished item.' 'INFO'
    exit 1
}
