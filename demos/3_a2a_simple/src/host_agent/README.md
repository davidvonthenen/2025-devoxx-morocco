# Host agent front end

This host agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to coordinate conversations between the weather and Hotel specialists. The routing graph enforces a simple travel policy:

* Hotel lodging questions always trigger a weather check first.
* Forecasts containing hazardous terms stop downstream lodging suggestions, and the user is told why the plan was paused.

Running through LangGraph keeps every decision reproducible and auditable because each policy branch is represented by an explicit edge in the graph.

## Prerequisites

* Install the repository dependencies (see the root `README.md`).
* Export `OPENAI_API_KEY` and launch the remote agents (`python -m src.weather_agent` and `python -m src.hotel_agent`).
* Optionally set `AIR_AGENT_URL` and `WEA_AGENT_URL` if the specialists are hosted somewhere other than `http://localhost:10001` and `http://localhost:10002`.

## Running the host UI

From the project root:

```bash
python -m src.host_agent
# or using the helper target:
make host_agent
```

Then open the Gradio UI at <http://127.0.0.1:11000> and try a prompt such as:

> Plan a spring hotel stay in Denver for four friends.

The host fetches a weather outlook first and only shares rental ideas if the forecast looks safe.
