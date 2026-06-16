import os
from typing import TypedDict, Annotated, Sequence
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
import json

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]
    forecast_data: str
    current_price: float
    trading_signal: str
    reasoning: str
    agent_engine: str

def format_forecast_for_llm(forecast_df: pd.DataFrame) -> str:
    """
    Formats the ensemble forecast into a readable string for the LLM.
    """
    summary = f"Next 24h Average Price: {forecast_df['ensemble_pred'].mean():.2f}\n"
    summary += f"Next 24h Max Price: {forecast_df['ensemble_pred'].max():.2f}\n"
    summary += f"Next 24h Min Price: {forecast_df['ensemble_pred'].min():.2f}\n"
    return summary

# Default model for the trading analyst. Sonnet balances cost/latency for a
# dashboard that calls the agent on every render; override with ENERGY_AGENT_MODEL.
DEFAULT_MODEL = os.environ.get("ENERGY_AGENT_MODEL", "claude-sonnet-4-6")

ANALYST_SYSTEM_PROMPT = (
    "You are a disciplined energy-market trading analyst. Given the current "
    "spot price and a short-horizon ensemble forecast summary, decide a single "
    "trading signal: BUY, SELL, or HOLD. Be concise and quantitative. "
    "Respond ONLY with a JSON object of the form "
    '{"signal": "BUY|SELL|HOLD", "reasoning": "<2-3 sentence rationale>"}.'
)


def _parse_avg_predicted(forecast_data: str, default: float = 50.0) -> float:
    try:
        for line in forecast_data.split("\n"):
            if "Average Price" in line:
                return float(line.split(": ")[1])
    except Exception:
        pass
    return default


def _rule_based_analysis(forecast_data: str, current_price: float) -> dict:
    """Deterministic fallback used when no LLM is available."""
    avg_predicted = _parse_avg_predicted(forecast_data, current_price)

    if avg_predicted > current_price * 1.05:
        signal = "BUY"
        reason = (f"The ensemble forecast predicts a significant increase in average price "
                  f"({avg_predicted:.2f}) compared to the current price ({current_price:.2f}). "
                  f"Upward trend expected.")
    elif avg_predicted < current_price * 0.95:
        signal = "SELL"
        reason = (f"The ensemble forecast predicts a significant decrease in average price "
                  f"({avg_predicted:.2f}) compared to the current price ({current_price:.2f}). "
                  f"Downward trend expected.")
    else:
        signal = "HOLD"
        reason = (f"The ensemble forecast predicts a stable price ({avg_predicted:.2f}) close to "
                  f"the current price ({current_price:.2f}). Market volatility is low.")

    return {"trading_signal": signal, "reasoning": reason, "agent_engine": "rule-based"}


def _llm_analysis(forecast_data: str, current_price: float) -> dict:
    """Ask Claude for a signal. Raises on any failure so the caller can fall back."""
    from langchain_anthropic import ChatAnthropic

    llm = ChatAnthropic(model=DEFAULT_MODEL, temperature=0, max_tokens=512)
    user_msg = (
        f"Current spot price: {current_price:.2f}\n\n"
        f"Forecast summary (next 24h):\n{forecast_data}\n\n"
        "Provide your trading signal as JSON."
    )
    resp = llm.invoke([
        {"role": "system", "content": ANALYST_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ])
    text = resp.content if isinstance(resp.content, str) else str(resp.content)

    # Extract the JSON object even if wrapped in prose/code fences.
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object in LLM response: {text!r}")
    parsed = json.loads(text[start:end + 1])

    signal = str(parsed.get("signal", "")).upper().strip()
    if signal not in ("BUY", "SELL", "HOLD"):
        raise ValueError(f"Invalid signal from LLM: {signal!r}")

    return {
        "trading_signal": signal,
        "reasoning": parsed.get("reasoning", "").strip(),
        "agent_engine": f"claude:{DEFAULT_MODEL}",
    }


def analyze_market_node(state: AgentState):
    """
    Analyze the market and emit a trading signal.

    Uses Claude (ChatAnthropic) when ANTHROPIC_API_KEY is set; otherwise — or if
    the LLM call fails for any reason — falls back to deterministic rule-based logic.
    """
    forecast_data = state.get("forecast_data", "")
    current_price = state.get("current_price", 50.0)

    if os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return _llm_analysis(forecast_data, current_price)
        except Exception as e:
            fallback = _rule_based_analysis(forecast_data, current_price)
            fallback["reasoning"] += f"  (LLM unavailable, used rule-based fallback: {e})"
            fallback["agent_engine"] = "rule-based (LLM failed)"
            return fallback

    return _rule_based_analysis(forecast_data, current_price)

def generate_report_node(state: AgentState):
    """
    Finalizes the report generation state.
    """
    # In a real LangGraph setup, this might route to a report formatter LLM.
    # We will pass the state to our report_generator module.
    return state

# Build the Graph
workflow = StateGraph(AgentState)

workflow.add_node("analyze_market", analyze_market_node)
workflow.add_node("generate_report", generate_report_node)

workflow.set_entry_point("analyze_market")
workflow.add_edge("analyze_market", "generate_report")
workflow.add_edge("generate_report", END)

app_graph = workflow.compile()

def run_trading_agent(forecast_df: pd.DataFrame, current_price: float) -> dict:
    """
    Entry point to run the agent.
    """
    forecast_str = format_forecast_for_llm(forecast_df)
    
    initial_state = {
        "messages": [HumanMessage(content="Analyze the energy market and provide a trading signal.")],
        "forecast_data": forecast_str,
        "current_price": current_price,
        "trading_signal": "",
        "reasoning": ""
    }
    
    final_state = app_graph.invoke(initial_state)
    return final_state
