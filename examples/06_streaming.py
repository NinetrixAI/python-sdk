"""
Example 6 — Real-time streaming output.

Demonstrates:
  - agent.stream() as an async generator
  - StreamEvent types: token, tool_start, tool_end, turn_end, done, error
  - Printing tokens as they arrive (no buffering)
  - Showing tool call / result summaries inline

Prerequisites:
  pip install ninetrix anthropic
  export ANTHROPIC_API_KEY=sk-...
"""

from __future__ import annotations

import asyncio

from ninetrix import Agent, Tool

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@Tool
def get_stock_price(ticker: str) -> dict:
    """Get the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL", "MSFT", "GOOGL".
    """
    # Replace with a real market data API
    fake_prices = {"AAPL": 189.30, "MSFT": 415.20, "GOOGL": 175.80, "AMZN": 198.50}
    price = fake_prices.get(ticker.upper())
    if price is None:
        return {"error": f"Unknown ticker '{ticker}'"}
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


@Tool
def compare_stocks(tickers: list[str]) -> dict:
    """Compare multiple stocks by price and rank them highest to lowest.

    Args:
        tickers: List of ticker symbols to compare.
    """
    fake_prices = {"AAPL": 189.30, "MSFT": 415.20, "GOOGL": 175.80, "AMZN": 198.50}
    results = {t.upper(): fake_prices.get(t.upper(), 0.0) for t in tickers}
    ranked = sorted(results.items(), key=lambda x: x[1], reverse=True)
    return {"ranked": ranked}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

analyst = Agent(
    name="stock-analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a helpful stock market analyst. "
        "Use your tools to look up prices, then give a brief plain-English summary."
    ),
    tools=[get_stock_price, compare_stocks],
)


# ---------------------------------------------------------------------------
# Streaming loop
# ---------------------------------------------------------------------------

async def stream_analysis(question: str) -> None:
    print(f"Question: {question}\n")
    print("─" * 50)

    total_tokens = 0

    async for event in analyst.stream(question):

        if event.type == "token":
            # Print each token immediately without a newline buffer
            print(event.content, end="", flush=True)

        elif event.type == "tool_start":
            print(f"\n⚙ calling {event.tool_name}({event.tool_args})", flush=True)

        elif event.type == "tool_end":
            print(f"  → {event.tool_result}\n", flush=True)

        elif event.type == "turn_end":
            total_tokens = event.tokens_used

        elif event.type == "done":
            print(f"\n{'─' * 50}")
            print(f"tokens: {event.tokens_used}  |  cost: ${event.cost_usd:.5f}")

        elif event.type == "error":
            print(f"\n[error] {event.error}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    asyncio.run(
        stream_analysis(
            "What are the current prices of AAPL, MSFT, and GOOGL? "
            "Which one is the most expensive and which is the cheapest?"
        )
    )
