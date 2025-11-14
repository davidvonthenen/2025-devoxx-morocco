PYTHON ?= python
INDEX ?= bbc
DATA_DIR ?= ./bbc
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION ?= How much did OpenAI purchase Windsurf for?

.PHONY: ingest agent server query client env mcp-server mcp-server-stdio mcp-server-sse

ingest:
	$(PYTHON) -m src.1_ingest.ingest --data-dir $(DATA_DIR) --index-name $(INDEX)

mcp-server:
	$(PYTHON) -m src.2_mcp_server.news_server --port 8765

mcp-server-stdio:
	$(PYTHON) -m src.2_mcp_server.news_server --stdio

mcp-server-sse: mcp-server

agent:
	MCP_TARGETS="sse:http://127.0.0.1:8765/mcp" $(PYTHON) -m src.3_rag_agent.server

server: agent

query:
	$(PYTHON) -m src.4_rag_client.client --question "$(QUESTION)"

client: query

env:
	@echo "OPENSEARCH_HOST=$${OPENSEARCH_HOST:-127.0.0.1}"
	@echo "OPENSEARCH_PORT=$${OPENSEARCH_PORT:-9200}"
	@echo "OPENSEARCH_INDEX=$${OPENSEARCH_INDEX:-$(INDEX)}"
	@echo "EMBEDDING_MODEL_NAME=$${EMBEDDING_MODEL_NAME:-thenlper/gte-small}"
	@echo "LLAMA_MODEL_PATH=$${LLAMA_MODEL_PATH:-./models/llama.gguf}"
	@echo "LLAMA_CTX=$${LLAMA_CTX:-8192}"
	@echo "LLAMA_N_THREADS=$${LLAMA_N_THREADS:-$$($(PYTHON) -c 'import os; print(os.cpu_count() or 1)')}"
	@echo "LLAMA_N_GPU_LAYERS=$${LLAMA_N_GPU_LAYERS:--1}"
	@echo "RAG_TOP_K=$${RAG_TOP_K:-5}"
	@echo "RAG_NUM_CANDIDATES=$${RAG_NUM_CANDIDATES:-50}"
	@echo "SERVER_HOST=$${SERVER_HOST:-$(HOST)}"
	@echo "SERVER_PORT=$${SERVER_PORT:-$(PORT)}"
