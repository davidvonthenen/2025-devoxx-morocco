# Local MCP-Enabled LLM Agent

This project provides a lightweight OpenAI-compatible chat completions API backed by a
local `llama-cpp-python` model. The agent integrates with Model Context Protocol (MCP)
tools and ships with a financial data service that surfaces SEC filing context. The MCP
server aggregates Finnhub's earnings calendar alongside Alpha Vantage earnings history,
exposing tools that answer questions about recent quarterly performance.

## Prerequisites
- Python 3.10+
- A quantised GGUF model accessible on disk (see `LLAMA_MODEL_PATH`)

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # update paths and overrides as needed
```

## Usage

### Launch the MCP SEC filings server (optional but recommended)
```bash
make mcp-server
```
The server exposes the following tools over SSE on `http://127.0.0.1:8765/mcp`:
- `sec-filings.get_earnings_calendar`
- `sec-filings.get_recent_quarters_results`
- `sec-filings.get_recent_quarters_improvement`

### Run the agent API
```bash
make agent
```
This starts the OpenAI-style endpoint on `0.0.0.0:8000` (configurable via environment
variables). At startup the agent discovers MCP tools and enables function calling for
them automatically.

## Configuration
All configuration is driven by environment variables (see `.env.example`). Key values:
- `LLAMA_MODEL_PATH`, `LLAMA_CTX`, `LLAMA_N_THREADS`, `LLAMA_N_GPU_LAYERS`,
  `LLAMA_N_BATCH`, `LLAMA_N_UBATCH`, `LLAMA_LOW_VRAM`
- `SERVER_HOST`, `SERVER_PORT`
- `MCP_ENABLED`, `MCP_TARGETS`, `MCP_CONNECT_TIMEOUT_SEC`,
  `MCP_INVOCATION_TIMEOUT_SEC`, `MCP_MAX_TOOL_CALL_DEPTH`

The agent automatically discovers tools declared by each MCP target and exposes them to
the model via OpenAI function-calling metadata.

## Notes
- There is no retrieval-augmented generation pipeline—responses rely on the LLM and
  any MCP tools it calls during the conversation.
- Streaming responses are supported when `stream: true` is supplied in the payload.
- The SEC filings server requires valid `FINNHUB_API_KEY`
  credentials in the environment to service requests.
