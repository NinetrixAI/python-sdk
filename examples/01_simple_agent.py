"""
Example 1 — Simple agent with a custom @Tool.

Demonstrates:
  - Defining tools with @Tool
  - Creating an Agent with a local tool
  - Running synchronously with .run() and async with .arun()

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import os
import json
from typing import Any

from ninetrix import Agent, Tool, enable_debug


enable_debug()
# ---------------------------------------------------------------------------
# 1. Define tools the agent can call
# ---------------------------------------------------------------------------


print(os.environ.get("NINETRIX_RUNNER_TOKEN"))  #

@Tool
def get_weather(city: str, unit: str = "celsius") -> dict:
    """Fetch current weather for a city.

    Args:
        city: City name, e.g. "London".
        unit: Temperature unit — "celsius" or "fahrenheit".
    """
    # Replace with a real weather API call
    fake_data = {"city": city, "temp": 22, "unit": unit, "condition": "sunny"}
    return fake_data


@Tool
def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    """Convert an amount between two currencies using live rates.

    Args:
        amount: Amount to convert.
        from_currency: ISO 4217 source currency code, e.g. "USD".
        to_currency: ISO 4217 target currency code, e.g. "EUR".
    """
    # Replace with a real FX API call
    rates = {"USD_EUR": 0.92, "EUR_USD": 1.09, "GBP_USD": 1.27}
    key = f"{from_currency}_{to_currency}"
    rate = rates.get(key, 1.0)
    return {"amount": amount, "from": from_currency, "to": to_currency, "result": round(amount * rate, 2)}


# ---------------------------------------------------------------------------
# 2. Build the agent
# ---------------------------------------------------------------------------

agent = Agent(
    name="travel-assistant",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",   # fast + cheap for demos
    role="You are a helpful travel assistant. Use your tools to answer questions.",
    tools=[get_weather, convert_currency]
)

# ---------------------------------------------------------------------------
# 3. Inspect before running (no API call)
# ---------------------------------------------------------------------------

async def main() -> None:
    info = agent.info()
    print(f"Agent  : {info.name}")
    print(f"Model  : {info.model}")
    print(f"Tools  : {info.local_tools}")

    issues = await agent.validate()
    if issues:
        for issue in issues:
            print(f"  [{issue.level}] {issue.message}")
    else:
        print("Validation: OK")

    # Async run
    result = await agent.arun("What's the weather in Tokyo, and how much is 100 USD in EUR?")

    print(f"\nOutput : {result.output}")
    print(f"Tokens : {result.tokens_used}  |  Cost: ${result.cost_usd:.4f}")
    print(f"Thread : {result.thread_id}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
