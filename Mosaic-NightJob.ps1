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
    .\Mosaic-NightJob.ps1 -Action Export -Destination E:\mosaic-portable -Full
#>

[CmdletBinding()]
param(
    [ValidateSet('EnsureDeps', 'Run', 'Status', 'Stop', 'Export')]
    [string] $Action = 'Run',

    # Wall-clock stop, checked between items so work halts on a boundary.
    [string] $Until,
    [int]    $MaxMinutes,

    [int]    $BatchSize = 20,

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

    # Workers stream progress for hours, so they run inline rather than being
    # captured. 'Continue' is what stops an ordinary library warning on stderr
    # from aborting the whole night job.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        & $VenvPython @argv
        $code = $LASTEXITCODE
    }
    finally {
        $ErrorActionPreference = $previous
    }

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

    Install-WithWinget -Id 'Python.Python.3.12' -Command 'python' -Friendly 'Python'   | Out-Null
    Install-WithWinget -Id 'Gyan.FFmpeg'        -Command 'ffmpeg' -Friendly 'ffmpeg'   | Out-Null
    Install-WithWinget -Id 'Git.Git'            -Command 'git'    -Friendly 'git'      | Out-Null

    Initialize-PythonEnv
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
    if ($Until)      { Write-Log "Will stop at $Until" }
    if ($MaxMinutes) { Write-Log "Will stop after $MaxMinutes minutes" }

    $deadline = Get-DeadlineArgs
    $common = @()
    if ($Force) { $common += '--force' }

    # Stray .part files and state that ran ahead of its artifacts are both
    # normal after a kill. Reconcile before doing any new work.
    Invoke-Worker -Module 'workers.reconcile' | Out-Null

    if ($Url) {
        Write-Log "Queueing single URL: $Url"
        Invoke-Worker -Module 'workers.crawl' -Arguments (@('--single', $Url) + $common) | Out-Null
    }
    else {
        Write-Log 'Step 1/5  crawl site'
        $crawlArgs = @('--batch-size', "$BatchSize") + $deadline + $common
        if ($MaxPages -gt 0) { $crawlArgs += @('--max-pages', "$MaxPages") }
        Invoke-Worker -Module 'workers.crawl' -Arguments $crawlArgs | Out-Null
    }

    Write-Log 'Step 2/5  ingest Bible'
    Invoke-Worker -Module 'workers.ingest_bible' -Arguments $common | Out-Null

    Write-Log 'Step 3/5  download audio'
    Invoke-Worker -Module 'workers.download_audio' `
        -Arguments (@('--batch-size', "$BatchSize") + $deadline + $common) | Out-Null

    if ($SkipTranscribe) {
        Write-Log 'Skipping transcription (-SkipTranscribe)'
    } else {
        Write-Log 'Step 4/5  transcribe'
        Invoke-Worker -Module 'workers.transcribe' `
            -Arguments (@('--batch-size', "$BatchSize") + $deadline + $common) | Out-Null
    }

    Write-Log 'Step 5/5  index'
    Invoke-Worker -Module 'workers.index_build' -Arguments ($deadline + $common) | Out-Null

    Write-Section 'Run complete'
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

function Invoke-Export {
    Write-Section 'Export portable bundle'

    if (-not $Destination) {
        throw 'Export needs -Destination, for example: -Destination E:\mosaic-portable'
    }

    $args = @('--destination', $Destination)
    if ($Full) {
        $args += '--full'
        Write-Log 'Full export: index, content, and models'
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
    }
}
catch {
    Write-Log $_.Exception.Message 'ERROR'
    Write-Log 'State is append-only; re-running continues from the last finished item.' 'INFO'
    exit 1
}