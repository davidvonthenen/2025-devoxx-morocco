# Host agent front end

This host agent uses [LangGraph](https://github.com/langchain-ai/langgraph) to coordinate conversations between the News and Financial specialists.

Running through LangGraph keeps every decision reproducible and auditable because each policy branch is represented by an explicit edge in the graph.

## Prerequisites

* Install the repository dependencies (see the root `README.md`).
* Export `OPENAI_API_KEY` and launch the remote agents (`make news_agent` and `make finacial_agent`).

## Running the host UI

From the project root:

```bash
python -m src.host_agent
# or using the helper target:
make host_agent
```

Then open the Gradio UI at <http://127.0.0.1:11000> and try a prompt such as:

> How much did Google purchase Windsurf for based on news articles?
> How was Google's (ticker symbol: GOOG) quarterly results for last quarter?
> How much did Google's (ticker symbol: GOOG) quarterly results improve for the previous quarter?

The host fetches a weather outlook first and only shares rental ideas if the forecast looks safe.
