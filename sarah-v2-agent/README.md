# Sarah v2.0 Agent

 Sarah v2.0 is a local conversational agent. The Windows-friendly CLI accepts typed text and can optionally capture microphone input for transcription. The local web UI supports browser microphone input and browser speaker playback.

Tools, evaluations, local document ingestion, retrieval, optional microphone input, and the local web UI are implemented. CLI voice output is not implemented; web voice output uses the browser's speech synthesis.

## Requirements

- Python 3.11 or newer
- Windows PowerShell or Command Prompt

## Setup

From this project folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
copy .env.example .env
```

Edit `.env` and fill in values as needed.

At minimum, set:

```text
OPENAI_API_KEY=...
SARAH_MODEL=gpt-5.2
TRANSCRIPTION_MODEL=gpt-4o-mini-transcribe
```

The external source documents currently live at:

```text
C:\Users\akala\Documents\Codex\2026-06-03\files-mentioned-by-the-user-pasted\data\source_docs
```

This project also includes a local placeholder folder at `data\source_docs` for future ingestion workflows.

## Run

```powershell
python -m app.main
```

Type a message and press Enter. Sarah retrieves local document chunks for each message, sends a compact context packet to OpenAI, and replies in text only.

To speak instead of typing, press Enter on an empty line. Sarah records from the PC microphone until silence or `Ctrl+C`, transcribes the audio, shows `You said: <transcript>`, then sends that transcript to Sarah.

Commands:

```text
/sources  show chunks used for the last answer
/debug_prompt  show a redacted prompt assembly summary
/reset    clear active conversation history
/remember <text>  store a durable long-term memory
/forget <keyword> deactivate matching memories
/memory   list active memories
/quit     exit
```

Conversation transcripts are saved under `data\conversations\YYYY-MM-DD\*.jsonl`.

`/debug_prompt` prints layer names, approximate token counts, source counts, and budget/compression notes for the last answer. It does not print hidden prompts, memory text, source excerpts, or conversation content.

Long-term memory is stored in `data\memory\sarah_memory.jsonl`. Sarah should not store every conversation. Use `/remember <text>` for durable facts, preferences, recurring projects, relationship continuity, canon facts, or unresolved questions. Use `/forget <keyword>` to deactivate matching memories.

## Happy Path

Ingest the source documents:

```powershell
python -m app.rag.ingest --source "C:\Users\akala\Documents\Codex\2026-06-03\files-mentioned-by-the-user-pasted\data\source_docs"
```

Run the demo:

```powershell
python -m app.demo
```

Run interactive chat:

```powershell
python -m app.main
```

Run the local web UI:

```powershell
python -m app.ui.web_app
```

Then open `http://127.0.0.1:8000`. The web UI binds to `127.0.0.1` by default and does not expose API keys to the browser.

For the voice webpage:

- Use Microsoft Edge or Chrome for browser speech recognition.
- Click `Speak`, talk into the PC microphone, then pause. The browser transcribes your speech and sends it to Sarah.
- Sarah's answer appears as text and is read over the speakers when `Speaker on` is enabled.
- Use `Sarah voice` to choose the browser voice. The page prefers Microsoft Ava first, then other neutral US English women's voices such as Jenny, Aria, or Zira when available.
- Use `Stop voice` to cancel playback.
- If the browser asks for microphone permission, allow it for `127.0.0.1`.

Use microphone input:

```powershell
python -m pip install -e ".[mic]"
python -m app.main
```

Then press Enter on an empty `You:` prompt to speak. Sarah still replies in text only.

Sarah also has a web/current-data layer. For questions that look current, geopolitical, market-related, leader-related, or likely to have changed recently, she retrieves from open RSS/news and official sources where relevant, caches results for 30 minutes under `data\web_cache`, and cites web/news sources with title, publisher, date, and URL. If that layer fails, she should say the current-data layer failed instead of pretending to have fresh information.

## Microphone Input On Windows

Typed chat works with the base install. For microphone capture, install the optional audio packages:

```powershell
python -m pip install -e ".[mic]"
```

For local transcription, install faster-whisper:

```powershell
python -m pip install -e ".[local-whisper]"
```

Sarah will prefer `faster-whisper` when installed. You can choose the local model size in `.env`:

```text
WHISPER_MODEL_SIZE=base
```

If you use whisper.cpp instead, set these in `.env`:

```text
WHISPER_CPP_EXE=C:\path\to\whisper-cli.exe
WHISPER_CPP_MODEL=C:\path\to\ggml-base.en.bin
```

If neither local option is available, Sarah falls back to the OpenAI Transcription API when `OPENAI_API_KEY` is set. The default model is `gpt-4o-mini-transcribe`, and WAV files are supported by the API.

Windows CLI audio notes:

- Make sure Windows Settings allows microphone access for desktop apps.
- If `sounddevice` cannot find an input device, install or update your microphone/audio interface driver.
- The CLI still replies in text only. The web UI can read Sarah's answers aloud through browser speech synthesis.

## Ingest Documents

Install the project first so PDF, DOCX, and OpenAI dependencies are available:

```powershell
python -m pip install -e .
```

Then ingest the local source documents:

```powershell
python -m app.rag.ingest --source "C:\Users\akala\Documents\Codex\2026-06-03\files-mentioned-by-the-user-pasted\data\source_docs"
```

The vector store is written to:

```text
data\vector_store
```

If `OPENAI_API_KEY` is set, Sarah uses the configured OpenAI embedding model. Without an API key, ingestion uses a deterministic local hashing embedder so the retrieval path remains runnable offline.

## Retrieve Evidence

```powershell
python -m app.rag.retriever --query "Who is Sarah Nelson?"
```

Each result includes the source filename, page or section when available, chunk id, and similarity score. If the best evidence is weak, the retriever prints that evidence is insufficient so Sarah should not answer from the documents.

## Run Evals

Sarah's behavioral eval cases live in `app\evals\sarah_eval_cases.yaml`.

```powershell
python -m app.evals.run_evals
```

Useful options:

```powershell
python -m app.evals.run_evals --limit 1
python -m app.evals.run_evals --jsonl data\conversations\eval-results.jsonl
python -m app.evals.run_evals --llm-judge
```

The default scorer is heuristic and reports missing expected traits plus forbidden-trait hits. `--llm-judge` adds an optional model-based judgment pass.

## Compile Sarah Canon

After ingesting source documents, compile a source-cited canon digest:

```powershell
python -m app.persona.compile_sarah_canon
```

The output is written to `data\sarah_canon.md`. Confirmed claims should include source filename references; weak or missing evidence is marked `UNCONFIRMED`.

## Master Prompt

Sarah's master identity and voice prompt lives at:

```text
config\sarah_master_prompt.md
```

The prompt builder loads this file as the immutable identity layer, then adds the constitution, safety boundaries, style directives, memory, canon/RAG context, web context, conversation history, and the user message.

## Project Layout

```text
app/
  core/       configuration, logging, application services
  persona/    Sarah persona definitions
  rag/        retrieval and vector store code
  tools/      local tools Sarah can call
  ui/         user interfaces
  evals/      evaluation helpers
data/
  source_docs/
  vector_store/
config/
tests/
```
