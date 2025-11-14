# Devoxx Morocco 2025: Rethinking RAG: How MCP and Agent2Agent Will Transform the Future of Intelligent Search

Welcome to the landing page for the workshop `Rethinking RAG: How MCP and Agent2Agent Will Transform the Future of Intelligent Search` at the `Devoxx Morocco 2025`.

## What to Expect

This workshop intends to provide an introduction to:

- Introduction to using [Agent2Agent (A2A) Protocol](https://github.com/a2aproject/A2A) to connect multiple agents together
- Introduction to using [Model Context Protocol (MCP)](https://github.com/modelcontextprotocol) to use with an AI Agent
- How to Build an Alternative to a RAG Agent That Provides Similar Functionality
- How To Connect This Agent With a Traditional RAG Agent (Using OpenSearch)

All of this using Small Language Models (SLM) which require less resources at inference time.

## Software Prerequisites

- A Linux or Mac-based Developer’s Laptop with enough memory to run a database (ie OpenSearch) plus Intel's neural-chat SLM (below).
  - Windows Users should use a VM or Cloud Instance
- Python Installed: version 3.12 or higher
- (Recommended) Using a miniconda or venv virtual environment
- Docker (Linux or MacOS) Installed: for running a local OpenSearch instance
- Basic familiarity with shell operations

Docker images you should pre-pull in your environment:

- `docker image pull opensearchproject/opensearch:3`

### LLM to pre-download:

This is the official one used in today's workshop:

- Intel's [neural-chat-7B-v3-3-GGUF](https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF)

OR

- Huggingface [bartowski/Meta-Llama-3-8B-Instruct-GGUF](https://huggingface.co/bartowski/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf)

OR

- Alternatively, using [ollama](https://ollama.com/)
  - Llama 3B: [https://ollama.com/library/llama3:8b](https://ollama.com/library/llama3:8b)

## Participation Options

There are 4 separate demos:

- [demo/1_contradictory](./demos/1_contradictory/README.md)
- [demo/2_mcp_whats_relevant](./demos/2_mcp_whats_relevant/README.md)
- [demo/3_a2a_simple](./demos/3_a2a_simple/README.md)
- [demo/4_a2a_financials](./demos/4_a2a_financials/README.md)

The instructions and purpose for each demo is contained within their respective folders.
