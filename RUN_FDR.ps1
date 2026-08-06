cd "C:\Users\akala\Documents\Codex\2026-06-04\create-a-python-project-named-sarah\fdr-v1-agent"
if (-not (Test-Path ".\.venv")) {
    py -m venv .venv
}
.\.venv\Scripts\Activate.ps1
python -m pip install fastapi uvicorn pydantic python-dotenv httpx
$env:AGENT_NAME="FDR v1.0"
$env:AGENT_MODEL="gpt-5.2"
python -m app.ui.web_app --port 8010
