.PHONY: hotel_agent weather_agent host_agent test

hotel_agent:
	python -m src.hotel_agent

weather_agent:
	python -m src.weather_agent

host_agent:
	python -m src.host_agent

client: host_agent

test:
	python -m pytest
