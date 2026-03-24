import os
import requests
from dotenv import load_dotenv

from langchain.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory

load_dotenv()


# --- 1. TOOL ---
@tool
def get_ticketmaster_events(query: str, city: str = "") -> str:
    """
    Search for live events on Ticketmaster.
    Use this when the user asks for events or things to do.

    Args:
        query: Type of event (e.g. badminton, concert, comedy)
        city: City to search in (e.g. London, Manchester)
    """
    api_key = os.getenv("TMAPI")
    if not api_key:
        return "Error: Ticketmaster API key not set. Please add TMAPI to your .env file."

    params = {
        "keyword": query,
        "apikey": api_key,
        "size": 5,          # fetch 5, we'll trim later
        "sort": "date,asc", # soonest events first
    }
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
            return f"No upcoming '{query}' events found in {city or 'your area'}."

        lines = []
        for e in events[:3]:
            name     = e.get("name", "Unknown Event")
            date     = e.get("dates", {}).get("start", {}).get("localDate", "TBA")
            venue    = e.get("_embedded", {}).get("venues", [{}])[0].get("name", "Unknown Venue")
            url      = e.get("url", "")
            lines.append(f"• {name}\n  📅 {date}  |  📍 {venue}\n  🔗 {url}")

        return "\n\n".join(lines)

    except requests.exceptions.Timeout:
        return "Error: Ticketmaster request timed out. Try again shortly."
    except requests.exceptions.HTTPError as e:
        return f"Error: Ticketmaster returned {e.response.status_code}."
    except Exception as e:
        return f"Unexpected error: {str(e)}"


# --- 2. PROMPT ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are UPlan, a friendly and witty event-discovery assistant.

Your workflow:
1. BRAINSTORM  — When a user mentions a hobby, suggest 2-3 related event types they might enjoy.
2. CLARIFY     — If no city has been provided, ask for one before searching. Never assume.
3. SEARCH      — Once you have a keyword AND a city, call get_ticketmaster_events.
4. PRESENT     — Format results clearly. If nothing is found, suggest trying a nearby city or broader keyword.

Always be encouraging and enthusiastic about helping people get out and do things!"""),
    MessagesPlaceholder(variable_name="chat_history"),  # memory slot
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"), # required for tool calls
])


# --- 3. MEMORY ---
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,   # returns Message objects, required for chat models
)


# --- 4. AGENT ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

agent = create_openai_functions_agent(
    llm=llm,
    tools=[get_ticketmaster_events],
    prompt=prompt,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=[get_ticketmaster_events],
    memory=memory,
    verbose=True,
    handle_parsing_errors=True,  # recovers gracefully from malformed tool calls
    max_iterations=5,            # prevents runaway loops
)


# --- 5. RUN ---
if __name__ == "__main__":
    print("🎟️  UPlan — Your Personal Event Scout")
    print("Type 'quit' to exit\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit", "q"):
            break
        if not user_input:
            continue

        result = agent_executor.invoke({"input": user_input})
        print(f"\nUPlan: {result['output']}\n")