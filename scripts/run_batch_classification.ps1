# Dell-AITC Batch Classification Runner
# Runs batch classification of use cases with proper environment setup

# Parameters
param(
    [int]$BatchSize = 20,
    [switch]$DryRun,
    [double]$MinConfidence = 0.45
)

# Script setup
$ErrorActionPreference = "Stop"
$StartTime = Get-Date

# Ensure we're in the project root
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

# Create log directory if it doesn't exist
$LogDir = Join-Path $ProjectRoot "logs" "batch_processing"
if (-not (Test-Path $LogDir)) {
    New-Item -ItemType Directory -Path $LogDir | Out-Null
}

# Log file setup
$LogFile = Join-Path $LogDir ("batch_run_" + (Get-Date -Format "yyyyMMdd_HHmmss") + ".log")

# Function to write to both console and log file
function Write-Log {
    param($Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogMessage = "$Timestamp - $Message"
    Write-Host $LogMessage
    Add-Content -Path $LogFile -Value $LogMessage
}

# Verify Python environment
try {
    Write-Log "Verifying Python environment..."
    python -c "import sys; print(f'Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"
} catch {
    Write-Log "ERROR: Python not found or not in PATH"
    exit 1
}

# Verify required environment variables
$RequiredEnvVars = @(
    "NEO4J_URI",
    "NEO4J_USER",
    "NEO4J_PASSWORD",
    "OPENAI_API_KEY"
)

$MissingVars = @()
foreach ($Var in $RequiredEnvVars) {
    if (-not (Get-Item "env:$Var" -ErrorAction SilentlyContinue)) {
        $MissingVars += $Var
    }
}

if ($MissingVars.Count -gt 0) {
    Write-Log "ERROR: Missing required environment variables: $($MissingVars -join ', ')"
    exit 1
}

# Build command arguments
$CmdArgs = @(
    "-m", "backend.app.services.batch.run_batch_processing",
    "--batch-size", $BatchSize,
    "--min-confidence", $MinConfidence
)

if ($DryRun) {
    $CmdArgs += "--dry-run"
}

# Run batch processing
try {
    Write-Log "Starting batch processing with size $BatchSize..."
    Write-Log "Command: python $($CmdArgs -join ' ')"
    
    $Process = Start-Process -FilePath "python" -ArgumentList $CmdArgs -NoNewWindow -PassThru -Wait
    
    if ($Process.ExitCode -ne 0) {
        Write-Log "ERROR: Batch processing failed with exit code $($Process.ExitCode)"
        exit $Process.ExitCode
    }
    
    $EndTime = Get-Date
    $Duration = $EndTime - $StartTime
    
    Write-Log "Batch processing completed successfully"
    Write-Log "Total duration: $($Duration.TotalMinutes.ToString('F2')) minutes"
    
} catch {
    Write-Log "ERROR: Failed to run batch processing: $_"
    exit 1
} 