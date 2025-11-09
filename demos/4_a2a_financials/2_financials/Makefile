PYTHON ?= python
HOST ?= 0.0.0.0
PORT ?= 8000
MCP_PORT ?= 8765
QUESTION ?= How was Apple's (ticker: $AAPL) stock performance in the last quarter?
# QUESTION ?= How was Apple's (ticker: $AAPL) stock performance improvement compared to the previous quarter?

.PHONY: agent server query client env mcp-server mcp-server-stdio mcp-server-sse


mcp-server:
	$(PYTHON) -m src.1_mcp_server.sec_filings_server --port $(MCP_PORT)

mcp-server-stdio:
	$(PYTHON) -m src.1_mcp_server.sec_filings_server --stdio

mcp-server-sse: mcp-server

agent:
	MCP_TARGETS="sse:http://127.0.0.1:$(MCP_PORT)/mcp" $(PYTHON) -m src.2_agent.server

server: agent

query:
	$(PYTHON) -m src.3_client.client --question "$(QUESTION)"

client: query

env:
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-./models/llama.gguf}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-8192}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:--1}"
	@echo "LLAMA_N_BATCH=$${LLAMA_N_BATCH:-256}"
	@echo "LLAMA_N_UBATCH=$${LLAMA_N_UBATCH:-256}"
	@echo "LLAMA_LOW_VRAM=$${LLAMA_LOW_VRAM:-1}"
	@echo "SERVER_HOST=$${SERVER_HOST:-$(HOST)}"
	@echo "SERVER_PORT=$${SERVER_PORT:-$(PORT)}"
	@echo "MCP_ENABLED=$${MCP_ENABLED:-1}"
	@echo "MCP_TARGETS=$${MCP_TARGETS:-sse:http://127.0.0.1:$(MCP_PORT)/mcp}"
	@echo "MCP_CONNECT_TIMEOUT_SEC=$${MCP_CONNECT_TIMEOUT_SEC:-5.0}"
	@echo "MCP_INVOCATION_TIMEOUT_SEC=$${MCP_INVOCATION_TIMEOUT_SEC:-15.0}"
	@echo "MCP_MAX_TOOL_CALL_DEPTH=$${MCP_MAX_TOOL_CALL_DEPTH:-3}"
