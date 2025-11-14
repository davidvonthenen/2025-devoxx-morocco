# Hotel remote agent

This service exposes an A2A-compatible agent that fabricates example Hotel listings using the OpenAI API. The host agent contacts it once policy checks allow lodging suggestions.

## Prerequisites

* Install dependencies from the repository root.
* Export `OPENAI_API_KEY`. Optionally override `OPENAI_MODEL` or `OPENAI_HOTEL_MODEL` to target a specific chat model.
* (Optional) Set `APP_URL` when publishing the service behind a reverse proxy so the generated agent card advertises the correct public URL.

## Run the server

From the project root:

```bash
python -m src.hotel_agent
# or
make hotel_agent
```

The service listens on `http://0.0.0.0:10002` by default and serves an agent card describing the Hotel search capability.

## Disclaimer

Important: The sample code provided is for demonstration purposes and illustrates the mechanics of the Agent-to-Agent (A2A) protocol. When building production applications, it is critical to treat any agent operating outside of your direct control as a potentially untrusted entity.

All data received from an external agent—including but not limited to its AgentCard, messages, artifacts, and task statuses—should be handled as untrusted input. For example, a malicious agent could provide an AgentCard containing crafted data in its fields (e.g., description, name, skills.description). If this data is used without sanitization to construct prompts for a Large Language Model (LLM), it could expose your application to prompt injection attacks.  Failure to properly validate and sanitize this data before use can introduce security vulnerabilities into your application.

Developers are responsible for implementing appropriate security measures, such as input validation and secure handling of credentials to protect their systems and users.
