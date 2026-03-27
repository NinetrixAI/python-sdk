"""
Example 7 — OpenTelemetry tracing for production observability.

Demonstrates:
  - configure_otel() to send spans to any OTEL-compatible backend
    (Jaeger, Grafana Tempo, Datadog, Honeycomb, AWS X-Ray, …)
  - Automatic span emission: ninetrix.agent.run, ninetrix.agent.turn,
    ninetrix.tool.call — zero per-agent setup after configure_otel()
  - Graceful no-op when opentelemetry-api is not installed
  - enable_debug() as a lightweight alternative for local dev
  - Combining OTEL traces with the debug pretty-printer

Use case: a production agent service where every run is traced,
          enabling latency analysis, error alerting, and cost dashboards.

Prerequisites:
  pip install ninetrix anthropic

  # For real OTEL export (optional):
  pip install 'ninetrix[otel]'
  docker run -p 4317:4317 -p 16686:16686 jaegertracing/all-in-one

  export ANTHROPIC_API_KEY=sk-...

Spans emitted per run:
  ninetrix.agent.run          root span — agent name, model, thread_id
  ninetrix.agent.turn         child span per LLM call — turn index, tokens
  ninetrix.tool.call          child span per tool — tool name, success/failure
"""

from __future__ import annotations

import asyncio

from ninetrix import Agent, Tool, enable_debug


# ---------------------------------------------------------------------------
# 1. Configure OpenTelemetry (call once at process startup)
# ---------------------------------------------------------------------------

def setup_telemetry() -> None:
    """Wire OTEL to a local Jaeger instance.

    After this call every Agent in the process automatically emits spans —
    no per-agent configuration needed.

    If opentelemetry-api is not installed, configure_otel() is a no-op
    and a warning is printed.  The agent runs normally without tracing.
    """
    from ninetrix import configure_otel

    configure_otel(
        endpoint="http://localhost:4317",   # gRPC OTLP endpoint (Jaeger / Tempo)
        service_name="ninetrix-demo",
        # Pass extra headers for managed backends, e.g.:
        #   headers={"x-honeycomb-team": "your-api-key"}
        #   headers={"DD-API-KEY": "your-datadog-key"}
    )
    print("OTEL configured → sending spans to http://localhost:4317")
    print("Open http://localhost:16686 to view traces in Jaeger\n")


# ---------------------------------------------------------------------------
# 2. Define tools
# ---------------------------------------------------------------------------

@Tool
def get_stock_price(ticker: str) -> dict:
    """Fetch the current stock price for a ticker symbol.

    Args:
        ticker: Stock ticker symbol, e.g. "AAPL".
    """
    # Replace with a real market data API
    prices = {"AAPL": 189.50, "MSFT": 415.20, "GOOG": 175.30}
    price = prices.get(ticker.upper(), 0.0)
    return {"ticker": ticker.upper(), "price": price, "currency": "USD"}


@Tool
def get_company_info(ticker: str) -> dict:
    """Fetch basic company information for a ticker symbol.

    Args:
        ticker: Stock ticker symbol.
    """
    companies = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "market_cap": "2.9T"},
        "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "market_cap": "3.1T"},
        "GOOG": {"name": "Alphabet Inc.", "sector": "Technology", "market_cap": "2.1T"},
    }
    info = companies.get(ticker.upper(), {"name": "Unknown", "sector": "Unknown"})
    return {"ticker": ticker.upper(), **info}


# ---------------------------------------------------------------------------
# 3. Build the agent
# ---------------------------------------------------------------------------

analyst = Agent(
    name="equity-analyst",
    provider="anthropic",
    model="claude-haiku-4-5-20251001",
    role=(
        "You are a concise equity research analyst. "
        "Use your tools to look up stock data and provide a brief 2-sentence summary."
    ),
    tools=[get_stock_price, get_company_info],
)


# ---------------------------------------------------------------------------
# 4. Run with full observability
# ---------------------------------------------------------------------------

async def run_with_otel() -> None:
    """Run the agent with OTEL tracing active."""

    queries = [
        "What's the current price and market cap of Apple?",
        "Compare Microsoft and Google — which has the higher stock price?",
    ]

    for query in queries:
        print(f"Query: {query}")
        result = await analyst.arun(query)
        print(f"→ {result.output}")
        print(f"  tokens={result.tokens_used}  cost=${result.cost_usd:.4f}  thread={result.thread_id}")
        print()


async def run_with_debug_printer() -> None:
    """Same run but using the pretty-printer instead of OTEL.

    Useful for local development when you don't have an OTEL collector running.
    The pretty-printer writes directly to stderr so it doesn't pollute stdout.
    """
    # Attach the pretty-printer to this specific agent
    enable_debug(agent=analyst)

    print("=== Debug pretty-printer active (output → stderr) ===\n")
    result = await analyst.arun("What sector is Microsoft in?")
    print(f"\nFinal output: {result.output}")


# ---------------------------------------------------------------------------
# 5. Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    import os

    use_otel = os.environ.get("NINETRIX_USE_OTEL", "0") == "1"

    if use_otel:
        # Real OTEL run — requires 'pip install ninetrix[otel]' and a collector
        setup_telemetry()
        await run_with_otel()
        print("\nSpans sent. Open your trace backend to inspect.")
    else:
        # Local dev — pretty-printer only
        print("Running with debug pretty-printer (set NINETRIX_USE_OTEL=1 for real OTEL)\n")
        await run_with_debug_printer()

    # ── What spans look like (for documentation) ────────────────────────────
    print("\n" + "=" * 60)
    print("Spans emitted per agent.run() call:")
    print()
    print("  ninetrix.agent.run")
    print("  ├─ ninetrix.agent.turn  [turn=0, in_tokens=420, out_tokens=110]")
    print("  │  ├─ ninetrix.tool.call  [tool=get_stock_price, success=true]")
    print("  │  └─ ninetrix.tool.call  [tool=get_company_info, success=true]")
    print("  └─ ninetrix.agent.turn  [turn=1, in_tokens=680, out_tokens=95]")
    print()
    print("Span attributes include: agent_name, model, thread_id, cost_usd")


if __name__ == "__main__":
    asyncio.run(main())
