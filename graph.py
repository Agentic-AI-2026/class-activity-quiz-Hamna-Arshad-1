### Graph
import sys
import json
import re
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
# from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

# llm = ChatAnthropic(model="claude-3-5-haiku-20241022", temperature=0)
llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

PLAN_SYSTEM = """Break the user goal into an ordered JSON list of steps.
Each step MUST follow this EXACT schema:
  {"step": int, "description": str, "tool": str or null, "args": dict or null}

Available tools and their EXACT argument names:
  - calculator(expression: str)                    -> evaluate math like '15 * 8 + 20'
  - search_web(query: str)                         -> search the web for information
  - search_news(query: str)                        -> search for recent news
  - get_current_weather(city: str)                 -> get current weather for a city
  - get_weather_forecast(city: str, days: int)     -> get forecast for N days

Use null for tool/args on synthesis or writing steps.
Return ONLY a valid JSON array. No markdown, no explanation."""


class AgentState(TypedDict):
    goal: str
    plan: List[dict]
    current_step: int
    results: List[dict]


def planner_node(state: AgentState) -> AgentState:
    goal = state["goal"]
    response = llm.invoke([
        SystemMessage(content=PLAN_SYSTEM),
        HumanMessage(content=goal)
    ])
    raw = response.content if isinstance(response.content, str) else response.content[0].get("text", "")
    clean = re.sub(r"```json|```", "", raw).strip()
    plan = json.loads(clean)
    return {**state, "plan": plan, "current_step": 0, "results": []}


def _call_tool(tool_name: str, args: dict) -> str:
    try:
        if tool_name == "calculator":
            import math as mathlib
            expression = args.get("expression", "")
            safe_globals = {
                "__builtins__": {},
                "sqrt": mathlib.sqrt, "log": mathlib.log, "log2": mathlib.log2,
                "log10": mathlib.log10, "sin": mathlib.sin, "cos": mathlib.cos,
                "tan": mathlib.tan, "ceil": mathlib.ceil, "floor": mathlib.floor,
                "pi": mathlib.pi, "e": mathlib.e, "abs": abs, "round": round, "pow": pow,
            }
            result = eval(expression, safe_globals)
            return f"{expression} = {round(float(result), 6)}"

        elif tool_name in ("search_web", "search_news"):
            from tavily import TavilyClient
            TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "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")
            tavily = TavilyClient(api_key=TAVILY_API_KEY)
            query = args.get("query", "")
            if tool_name == "search_news":
                resp = tavily.search(query=query, topic="news", search_depth="basic", max_results=3)
            else:
                resp = tavily.search(query=query, search_depth="basic", max_results=3)
            items = resp.get("results", [])
            if not items:
                return f"No results found for: '{query}'"
            return "\n\n".join([f"[{i+1}] {r['title']}\n    {r['content']}" for i, r in enumerate(items)])

        elif tool_name == "get_current_weather":
            import requests as req
            CITY_COORDS = {
                "london": (51.5074, -0.1278), "paris": (48.8566, 2.3522),
                "new york": (40.7128, -74.0060), "tokyo": (35.6762, 139.6503),
                "karachi": (24.8607, 67.0011), "lahore": (31.5204, 74.3587),
                "islamabad": (33.6844, 73.0479), "rawalpindi": (33.5651, 73.0169),
                "dubai": (25.2048, 55.2708), "berlin": (52.5200, 13.4050),
                "sydney": (-33.8688, 151.2093), "chicago": (41.8781, -87.6298),
            }
            city = args.get("city", "")
            coords = CITY_COORDS.get(city.lower().strip())
            if not coords:
                return f"City '{city}' not found."
            lat, lon = coords
            url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   f"&current_weather=true&hourly=relativehumidity_2m,apparent_temperature")
            data = req.get(url, timeout=5).json()
            cw = data.get("current_weather", {})
            temp = cw.get("temperature", "N/A")
            wind = cw.get("windspeed", "N/A")
            wcode = cw.get("weathercode", 0)
            cond = "Sunny" if wcode < 3 else "Cloudy" if wcode < 50 else "Rainy"
            humidity = data.get("hourly", {}).get("relativehumidity_2m", ["N/A"])[0]
            feels = data.get("hourly", {}).get("apparent_temperature", ["N/A"])[0]
            return (f"Current weather in {city.title()}:\n  Condition : {cond}\n"
                    f"  Temp      : {temp} C\n  Feels like: {feels} C\n"
                    f"  Wind      : {wind} km/h\n  Humidity  : {humidity}%")

        elif tool_name == "get_weather_forecast":
            import requests as req
            CITY_COORDS = {
                "london": (51.5074, -0.1278), "paris": (48.8566, 2.3522),
                "new york": (40.7128, -74.0060), "tokyo": (35.6762, 139.6503),
                "karachi": (24.8607, 67.0011), "lahore": (31.5204, 74.3587),
                "islamabad": (33.6844, 73.0479), "rawalpindi": (33.5651, 73.0169),
                "dubai": (25.2048, 55.2708), "berlin": (52.5200, 13.4050),
                "sydney": (-33.8688, 151.2093), "chicago": (41.8781, -87.6298),
            }
            city = args.get("city", "")
            days = int(args.get("days", 3))
            coords = CITY_COORDS.get(city.lower().strip())
            if not coords:
                return f"City '{city}' not found."
            lat, lon = coords
            url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
                   f"&daily=temperature_2m_max,temperature_2m_min,weathercode"
                   f"&forecast_days={days}&timezone=auto")
            data = req.get(url, timeout=5).json()
            daily = data.get("daily", {})
            dates = daily.get("time", [])
            maxts = daily.get("temperature_2m_max", [])
            mints = daily.get("temperature_2m_min", [])
            wcodes = daily.get("weathercode", [])
            lines = [f"Forecast for {city.title()} ({days} days):"]
            for i in range(min(days, len(dates))):
                wc = wcodes[i] if i < len(wcodes) else 0
                cond = "Sunny" if wc < 3 else "Cloudy" if wc < 50 else "Rainy"
                lines.append(f"  {dates[i]} : {cond}, High {maxts[i]}C / Low {mints[i]}C")
            return "\n".join(lines)

        else:
            return f"Unknown tool: {tool_name}"

    except Exception as e:
        return f"Tool error ({tool_name}): {e}"


def executor_node(state: AgentState) -> AgentState:
    plan = state["plan"]
    idx = state["current_step"]
    results = list(state["results"])
    step = plan[idx]
    tool_name = step.get("tool")
    args = step.get("args") or {}

    if tool_name:
        result = _call_tool(tool_name, args)
    else:
        context = "\n".join([f"Step {r['step']}: {r['result']}" for r in results])
        resp = llm.invoke([HumanMessage(content=f"{step['description']}\n\nContext:\n{context}")])
        result = resp.content if isinstance(resp.content, str) else resp.content[0].get("text", "")

    results.append({"step": step["step"], "description": step["description"], "result": str(result)})
    return {**state, "current_step": idx + 1, "results": results}


def should_continue(state: AgentState) -> str:
    return "end" if state["current_step"] >= len(state["plan"]) else "executor"


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner_node)
    graph.add_node("executor", executor_node)
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", should_continue, {"executor": "executor", "end": END})
    return graph.compile()
