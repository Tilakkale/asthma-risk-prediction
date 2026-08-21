$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot\..

if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
} else {
    Write-Host "Virtual environment not found. Please create one first." -ForegroundColor Yellow
    exit 1
}

Write-Host "Starting Streamlit app on 0.0.0.0:8501..." -ForegroundColor Cyan
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
