from dataclasses import dataclass
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain.tools import tool, ToolRuntime
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents.structured_output import ToolStrategy

SYSTEM_PROMPT="""You are an expert weather forecaster who speaks in puns.

you have access to two tools:

-get_weather_for_location:use this to get the weather for a specific location
-get_user_location: use this to get the users location

if a user asks you for the weather make sure you know the location. If you can tell
from the question that they mean whereevre they are, use get_user_location
tool to find their location"""

@dataclass
class Context:
    """Custom runtime context schema"""
    user_id: str

@tool
def get_weather_for_location(city: str) -> str:
    """Get weather for a given city"""
    return f"Its always sunny in {city}!"

@tool
def get_user_location(runtime: ToolRuntime[Context]) -> str:
    """Retrieve the users location based on user_id"""
    user_id=runtime.context.user_id
    return "Florida" if user_id == 1 else "SF"

# Configure model
model = init_chat_model(
    "claude-sonnet-4-5-20250929",
    temperature=0
)

# Define response format
@dataclass
class ResponseFormat:
    """Response schema for the agent"""
    # A punny response (always required)
    punny_response: str
    # Any interesting information about the weather if available
    weather_conditions: str | None = None

# set up memory
checkpointer = InMemorySaver()

# create agent
agent = create_agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    tools=[get_user_location, get_weather_for_location],
    context_schema=Context,
    response_format=ToolStrategy(ResponseFormat),
    checkpointer=checkpointer
)

# Run agent
# thread id is a unique identifier for a given conversation
config = {"configurable": {"thread_id":"1"}}

response = agent.invoke(
    {"messages":[{"role":"user","content":"what is the weather outside"}]},
    config=config,
    context=Context(user_id="1")
)

print(response["structured_response"])

# Note that we can continue the conversation using the same thread_id
response = agent.invoke(
    {"messages":[{"role":"user","content":"thank you!"}]},
    config=config,
    context=Context(user_id="1")
)

print(response['structured_response'])

