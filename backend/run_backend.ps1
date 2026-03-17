 
 #!/usr/bin/env pwsh
# Land Chatbot Backend Startup Script

$ErrorActionPreference = "Stop"

Write-Host "`n=================================" -ForegroundColor Green
Write-Host "Land Chatbot Backend Startup" -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green

$backendPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$venvPath = Join-Path (Split-Path -Parent (Split-Path -Parent (Split-Path -Parent $backendPath))) ".venv"

Write-Host "`n📁 Backend Path: $backendPath"
Write-Host "🐍 Virtual Env: $venvPath"

# Change to backend directory
Set-Location $backendPath

# Activate virtual environment
Write-Host "`n✓ Activating virtual environment..."
$activateScript = Join-Path $venvPath "Scripts\Activate.ps1"
if (-Not (Test-Path $activateScript)) {
    Write-Host "❌ Virtual environment not found at $venvPath"
    Write-Host "Creating new virtual environment..."
    python -m venv $venvPath
    & $activateScript
} else {
    & $activateScript
}

Write-Host "✓ Virtual environment activated"

# Verify modules
Write-Host "`n🔍 Checking required modules..."
python -c @"
try:
    import fastapi
    import uvicorn
    import chromadb
    import sentence_transformers
    print('✓ All required modules found')
except ImportError as e:
    print(f'❌ Missing module: {e}')
    exit(1)
"@

if ($LASTEXITCODE -ne 0) {
    Write-Host "`n⚠️  Installing missing packages..."
    pip install -r requirements.txt
}

Write-Host "`n🔐 Configuration Check:"
python -c @"
import config
print(f'  ✓ LLM_ENABLED: {config.LLM_ENABLED}')
print(f'  ✓ RERANK_ENABLED: {config.RERANK_ENABLED}')
print(f'  ✓ SUPPRESS_CUDA_WARNINGS: {config.SUPPRESS_CUDA_WARNINGS}')
"@

Write-Host "`n" 
Write-Host "=================================" -ForegroundColor Green
Write-Host "🎯 Starting FastAPI Server..." -ForegroundColor Green
Write-Host "=================================" -ForegroundColor Green
Write-Host "`n📍 API will be available at: http://localhost:8000"
Write-Host "📚 Swagger Docs: http://localhost:8000/docs"
Write-Host "🛑 Press Ctrl+C to stop`n"

# Start the server
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --reload
