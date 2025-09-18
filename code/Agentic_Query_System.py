import os
import json
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# --- Part 1: Import and Load Data ---
# Import the data loading function and semantic mapping from your first script.
# This makes the loaded data available to our tools.
from data_loader import load_data, SEMANTIC_MAPPING

# Load all data into memory when the application starts.
all_data = load_data()
if not all_data:
    raise ValueError("Data could not be loaded. Please check your data files and paths.")

camera_feeds_df = all_data["camera_feeds"]
encoder_schema = all_data["encoder_schema"]
decoder_schema = all_data["decoder_schema"]
# Note: encoder/decoder_parameters are loaded but not used in these specific tools.
# They could be used in new tools you might add later.


# --- Part 2: Define the Tools (The Agent's "Hands") ---

@tool
def find_cameras(location: str = None, min_resolution: str = None, min_bitrate: int = None, codec: str = None) -> str:
    """
    Finds cameras based on a set of optional filter criteria.
    Args:
        location: The geographical location of the camera (e.g., 'pacific', 'atlantic').
        min_resolution: The minimum resolution required (e.g., '1080p', '4K').
        min_bitrate: The minimum bitrate in kbps (e.g., 5000).
        codec: The codec used by the camera (e.g., 'H.264', 'H.265').
    """
    query_df = camera_feeds_df.copy()
    if location:
        query_df = query_df[query_df['location'].str.lower() == location.lower()]
    if codec:
        query_df = query_df[query_df['codec'].str.lower() == codec.lower()]
    if min_bitrate:
        query_df = query_df[query_df['bitrate'] >= min_bitrate]
    if min_resolution:
        # Handle resolution ordering
        resolutions = ['720p', '1080p', '4K']
        if min_resolution in resolutions:
            allowed_resolutions = resolutions[resolutions.index(min_resolution):]
            query_df = query_df[query_df['resolution'].isin(allowed_resolutions)]

    if query_df.empty:
        return "No cameras found matching the criteria."

    # Return the result as a JSON string
    return query_df.to_json(orient='records')


@tool
def get_schema_definition(schema_type: str, parameter_name: str) -> str:
    """
    Retrieves the description of a specific parameter from the encoder or decoder schema.
    Args:
        schema_type: The type of schema to query ('encoder' or 'decoder').
        parameter_name: The name of the parameter to define (e.g., 'gop_size', 'bitrate').
    """
    schema_map = {
        "encoder": encoder_schema,
        "decoder": decoder_schema
    }
    selected_schema = schema_map.get(schema_type.lower())

    if not selected_schema:
        return f"Error: Invalid schema_type '{schema_type}'. Must be 'encoder' or 'decoder'."

    param_info = selected_schema.get("properties", {}).get(parameter_name)
    if not param_info or "description" not in param_info:
        return f"Parameter '{parameter_name}' not found in the {schema_type} schema."

    return json.dumps({"parameter": parameter_name, "description": param_info["description"]})


# --- Part 3: Configure the Agent ---

# Tie the tools together for the agent to use
tools = [find_cameras, get_schema_definition]
tool_executor = ToolExecutor(tools)

# Set up the Gemini model
# Make sure your GEMINI_API_KEY is set in your .env file
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model = model.bind_tools(tools)

# --- Part 4: Define the Agent State and Graph ---

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

# This is the primary node of our graph. It calls the model and decides what to do.
def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# This node executes the tools.
def call_tool(state):
    last_message = state['messages'][-1]
    # We found a tool call
    action = last_message.tool_calls[0]
    tool_output = tool_executor.invoke(action)
    return {"messages": [ToolMessage(content=str(tool_output), tool_call_id=action['id'])]}

# This conditional edge decides whether to continue using tools or to finish.
def should_continue(state):
    last_message = state['messages'][-1]
    if not last_message.tool_calls:
        return "end" # If there are no tool calls, we're done.
    else:
        return "continue" # Otherwise, we continue to the tool execution node.

# --- Part 5: Build and Run the Graph ---

# 1. Create the graph
workflow = StateGraph(AgentState)

# 2. Define the nodes
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# 3. Set the entry point
workflow.set_entry_point("agent")

# 4. Add the conditional edge
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "action",
        "end": END,
    },
)

# 5. Add the edge from the action back to the agent
workflow.add_edge("action", "agent")

# 6. Compile the graph into a runnable app
app = workflow.compile()

# --- Main execution block ---
if __name__ == "__main__":
    print("Agent is ready. Type 'exit' to quit.")
    # Create the system prompt
    system_prompt = f"""
    You are an expert AI assistant for a video analyst. Your goal is to answer questions about a dataset of camera feeds.
    You have access to a set of tools to find information.

    When interpreting user queries, use the following semantic definitions:
    {json.dumps(SEMANTIC_MAPPING, indent=2)}

    Always provide your final answer in a clear, easy-to-read format.
    If you use the find_cameras tool and it returns a lot of data, summarize it unless asked for full details.
    """

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        # We start the conversation with the system prompt and the user's query
        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]

        # Invoke the agent
        final_state = app.invoke({"messages": messages})

        # The final answer is the last message from the agent
        final_answer = final_state['messages'][-1].content
        print(f"Agent: {final_answer}")
