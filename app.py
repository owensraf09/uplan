import os
import streamlit as st
import requests
from dotenv import load_dotenv
from typing import Annotated, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

# --- FIX 1: Actually call load_dotenv() ---
load_dotenv()

# ─────────────────────────────────────────
# 1. AGENT STATE
# ─────────────────────────────────────────
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    # FIX 2: Track inferred user preferences across the conversation
    user_preferences: List[str]
    # FIX 3: Hold the draft response so the vetting node can review it
    draft_response: str


# ─────────────────────────────────────────
# 2. TICKETMASTER TOOL
# ─────────────────────────────────────────
@tool
def get_ticketmaster_events(query: str, city: str = "") -> str:
    """Search for upcoming live events on Ticketmaster."""
    api_key = os.getenv("TMAPI")
    if not api_key:
        return "Error: Ticketmaster API key not configured."

    params = {"keyword": query, "apikey": api_key, "size": 5, "sort": "date,asc"}
    if city:
        params["city"] = city

    try:
        response = requests.get(
            "https://app.ticketmaster.com/discovery/v2/events.json",
            params=params,
            timeout=8,
        )
        response.raise_for_status()
        data = response.json()
        events = data.get("_embedded", {}).get("events", [])
        if not events:
            return f"No upcoming '{query}' events found."

        lines = []
        for e in events[:3]:
            name  = e.get("name", "Unknown Event")
            date  = e.get("dates", {}).get("start", {}).get("localDate", "TBA")
            venue = e.get("_embedded", {}).get("venues", [{}])[0].get("name", "Unknown")
            url   = e.get("url", "")
            lines.append(f"• {name}\n  📅 {date} | 📍 {venue}\n  🔗 {url}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"Error fetching events: {str(e)}"


tools     = [get_ticketmaster_events]
tool_node = ToolNode(tools)

# Two separate model instances keep concerns clean
main_model    = ChatOpenAI(model="gpt-4o", temperature=0,   max_tokens=1024).bind_tools(tools)
vetting_model = ChatOpenAI(model="gpt-4o", temperature=0.3, max_tokens=1024)


# ─────────────────────────────────────────
# 3. GRAPH NODES
# ─────────────────────────────────────────

def call_model(state: AgentState) -> dict:
    """Main agent node — plans, calls tools, and drafts a response."""
    prefs = state.get("user_preferences", [])
    pref_str = (
        "Known user preferences: " + "; ".join(prefs)
        if prefs
        else "No explicit user preferences noted yet."
    )

    sys_msg = SystemMessage(content=(
        "You are UPlan, a warm and friendly AI assistant who helps people discover "
        "fun hobbies and events they might enjoy. Always respond with enthusiasm, "
        "encouragement, and a personable touch. Make users feel welcome and supported "
        "in exploring new interests.\n\n"
        f"{pref_str}"
    ))

    # FIX 4: Pass the FULL conversation history, not just the latest message
    response = main_model.invoke([sys_msg] + state["messages"])
    return {"messages": [response], "draft_response": response.content}


def vet_response(state: AgentState) -> dict:
    """
    Sub-agent / reflection node.
    Reads the draft, checks it against known preferences, and either
    approves it or rewrites it to be better tailored to the user.
    """
    draft = state.get("draft_response", "")
    prefs = state.get("user_preferences", [])

    # Nothing to vet if the main agent is about to call a tool
    last_msg = state["messages"][-1]
    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return {}   # pass through unchanged

    pref_context = (
        "Known user preferences:\n- " + "\n- ".join(prefs)
        if prefs
        else "No user preferences have been established yet."
    )

    vetting_prompt = (
        f"{pref_context}\n\n"
        f"Draft response to review:\n{draft}\n\n"
        "Your job:\n"
        "1. Check whether the draft is well-tailored to the user's known preferences.\n"
        "2. If it is good, return it verbatim.\n"
        "3. If it can be improved (e.g. it ignores a known preference, is too generic, "
        "   or misses an opportunity to personalise), rewrite it — keeping the same "
        "   facts but making it more relevant and encouraging.\n"
        "Return ONLY the final response text, no commentary."
    )

    vetted = vetting_model.invoke([
        SystemMessage(content="You are a quality-control sub-agent for UPlan. "
                               "Your sole job is to review and, if needed, improve "
                               "AI responses so they feel personal and preference-aware."),
        HumanMessage(content=vetting_prompt),
    ])

    # Replace the last AIMessage with the vetted version
    vetted_msg = AIMessage(content=vetted.content)
    return {"messages": [vetted_msg], "draft_response": vetted.content}


def update_preferences(state: AgentState) -> dict:
    """
    Lightweight preference-extraction node.
    Runs after every user turn to keep the preference list fresh.
    """
    prefs = state.get("user_preferences", [])

    # Collect all user messages so far
    user_msgs = [
        m.content for m in state["messages"] if isinstance(m, HumanMessage)
    ]
    if not user_msgs:
        return {}

    extraction_prompt = (
        f"Existing preferences: {prefs}\n\n"
        f"All user messages so far:\n" +
        "\n".join(f"- {m}" for m in user_msgs) +
        "\n\nExtract a concise updated list of the user's hobbies, interests, "
        "dislikes, location, and any other relevant preferences mentioned. "
        "Return them as a plain Python list of short strings, e.g. "
        '["likes jazz", "based in Manchester", "dislikes crowds"]. '
        "Return ONLY the list, nothing else."
    )

    result = vetting_model.invoke([HumanMessage(content=extraction_prompt)])

    try:
        # Safely parse the returned list string
        import ast
        new_prefs = ast.literal_eval(result.content.strip())
        if isinstance(new_prefs, list):
            return {"user_preferences": new_prefs}
    except Exception:
        pass  # If parsing fails, keep existing preferences

    return {}


# ─────────────────────────────────────────
# 4. ROUTING LOGIC
# ─────────────────────────────────────────

def should_continue(state: AgentState) -> str:
    """After the main agent: go to tools if there are tool calls, else vet."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "vet"


# ─────────────────────────────────────────
# 5. BUILD THE GRAPH
# ─────────────────────────────────────────
workflow = StateGraph(AgentState)

workflow.add_node("update_prefs", update_preferences)
workflow.add_node("agent",        call_model)
workflow.add_node("tools",        tool_node)
workflow.add_node("vet",          vet_response)

# Flow:
#   START → update_prefs → agent → (tools → agent)* → vet → END
workflow.add_edge(START,          "update_prefs")
workflow.add_edge("update_prefs", "agent")
workflow.add_conditional_edges("agent", should_continue, ["tools", "vet"])
workflow.add_edge("tools",        "agent")
workflow.add_edge("vet",          END)

uplan_app = workflow.compile()


# ─────────────────────────────────────────
# 6. STREAMLIT UI
# ─────────────────────────────────────────
st.set_page_config(page_title="UPlan AI", page_icon="🎯")
st.title("🎯 UPlan, Hobby & Event Assistant")

# FIX 5: Persist both the display messages AND the LangGraph message history
if "display_messages"  not in st.session_state:
    st.session_state.display_messages  = []   # [{role, content}]  for rendering
if "graph_messages"    not in st.session_state:
    st.session_state.graph_messages    = []   # [BaseMessage]       for the agent
if "user_preferences"  not in st.session_state:
    st.session_state.user_preferences  = []

# Sidebar: show inferred preferences so the user can see what UPlan knows
with st.sidebar:
    st.header("🧠 What I know about you")
    prefs = st.session_state.user_preferences
    if prefs:
        for p in prefs:
            st.markdown(f"- {p}")
    else:
        st.caption("Nothing yet — just start chatting!")
    if st.button("🗑️ Reset conversation"):
        st.session_state.display_messages = []
        st.session_state.graph_messages   = []
        st.session_state.user_preferences = []
        st.rerun()

# Render chat history
for message in st.session_state.display_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Tell me what you enjoy, or ask about events near you…"):

    # Show the user's message immediately
    st.session_state.display_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # FIX 6: Append to the running LangGraph history (not just the latest message)
    st.session_state.graph_messages.append(HumanMessage(content=prompt))

    with st.chat_message("assistant"):
        with st.spinner("UPlan is thinking…"):
            state_in = {
                "messages":         st.session_state.graph_messages,
                "user_preferences": st.session_state.user_preferences,
                "draft_response":   "",
            }
            final_state = uplan_app.invoke(state_in)

        # Extract the final (vetted) answer
        answer = final_state["messages"][-1].content
        st.markdown(answer)

    # Persist the assistant reply and updated state
    st.session_state.graph_messages    = final_state["messages"]
    st.session_state.user_preferences  = final_state.get("user_preferences", [])
    st.session_state.display_messages.append({"role": "assistant", "content": answer})

    # Refresh the sidebar preferences without a full page reload
    st.rerun()