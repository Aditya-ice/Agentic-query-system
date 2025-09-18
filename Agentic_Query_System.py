import json
import os
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode

# --- Part 1: Import and Load Data ---
from data_loader import load_data, SEMANTIC_MAPPING

all_data = load_data()
if not all_data:
    raise SystemExit(
        "Exiting: Data could not be loaded. Please check the 'datasets' folder and file paths.")

camera_feeds_df = all_data["camera_feeds"]
table_definitions_df = all_data["table_definitions"]


# --- Part 2: Define the Agent's Tools ---
@tool
def find_cameras(location: str = None, min_resolution: str = None, min_bitrate: int = None, codec: str = None, feed_id: str = None) -> str:
    """
    Finds cameras based on a set of optional filter criteria. Can also be used to look up a single camera by its feed_id.
    Args:
        location: The geographical theater of the camera (e.g., 'PAC', 'EUR', 'ME').
        min_resolution: The minimum resolution required (e.g., '1080p', '4K').
        min_bitrate: The minimum estimated bitrate in kbps (e.g., 4000).
        codec: The codec used by the camera (e.g., 'H264', 'H265').
        feed_id: The exact ID of a specific camera feed to look up (e.g., 'FD-ML64LG').
    """
    query_df = camera_feeds_df.copy()
    if feed_id:
        query_df = query_df[query_df['feed_id'].str.upper() == feed_id.upper()]
        return query_df.to_json(orient='records')
    if location:
        query_df = query_df[query_df['location'].str.upper()
                            == location.upper()]
    if codec:
        query_df = query_df[query_df['codec'].str.upper() == codec.upper()]
    if min_bitrate:
        query_df = query_df[query_df['bitrate'] >= min_bitrate]
    if min_resolution:
        resolutions = ['480p', '720p', '1080p', '1440p', '4K']
        if min_resolution in resolutions:
            allowed_resolutions = resolutions[resolutions.index(
                min_resolution):]
            query_df = query_df[query_df['resolution'].isin(
                allowed_resolutions)]
    if query_df.empty:
        return "No cameras found matching the criteria."
    return query_df[['feed_id', 'location', 'resolution', 'bitrate', 'codec']].to_json(orient='records')


@tool
def get_parameter_definition(parameter_name: str) -> str:
    """
    Retrieves the description of a specific parameter from the table definitions file.
    Use this to understand what columns like 'LAT_MS', 'CIV_OK', or 'THEATER' mean.
    Args:
        parameter_name: The exact name of the parameter/column header to define (e.g., 'LAT_MS').
    """
    try:
        definition = table_definitions_df.loc[parameter_name.upper(
        )]['description']
        return json.dumps({"parameter": parameter_name, "description": definition})
    except KeyError:
        return f"Parameter '{parameter_name}' not found in the definitions table."


# --- Part 3: Configure the Agent ---
tools = [find_cameras, get_parameter_definition]
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError(
        "GEMINI_API_KEY not found. Please set it in your .env file.")
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=api_key,
    temperature=0
)
model = model.bind_tools(tools)

# --- Part 4: Define Agent State and Graph ---

# --- MEMORY FIX ---
# By annotating the `messages` field with `add_messages`, we tell the graph
# to always APPEND new messages to the list, rather than replacing it.
# This preserves the conversation history.


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# The call_model function is now simpler. It just invokes the model.
# The graph takes care of adding the response to the state.


def call_model(state):
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def should_continue(state):
    last_message = state['messages'][-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    else:
        return "continue"


# --- Part 5: Build the LangGraph Workflow ---
workflow = StateGraph(AgentState)
tool_node = ToolNode(tools)
workflow.add_node("agent", call_model)
workflow.add_node("action", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\nAgent is ready. Type 'exit' to quit.")
    system_prompt_text = f"""
    You are an expert AI assistant for a video analyst. Your goal is to answer questions about a dataset of 100 camera feeds.
    You have access to a set of tools to find information. First, understand the user's question. Then, decide which tool, if any, is appropriate.
    You have the following domain knowledge to help you translate user requests:
    {json.dumps(SEMANTIC_MAPPING, indent=2)}
    - The user might use 'region' or 'theater'; both map to the 'location' parameter which uses abbreviations like 'PAC', 'EUR', etc.
    - If asked for details of a single camera, use the `feed_id` parameter in the `find_cameras` tool.
    - Always provide your final answer in a clear, easy-to-read format.
    - If a tool search returns a large number of cameras, summarize the result (e.g., "I found 15 cameras matching your criteria. Here are the first 3:") and show a few examples. Do not return a giant list unless asked.
    """
    while True:
        user_query = input("\nYou: ")
        if user_query.lower() == 'exit':
            break

        # The chat history is now properly initialized with the system prompt
        # and the user's first question.
        messages = [
            SystemMessage(content=system_prompt_text),
            HumanMessage(content=user_query)
        ]

        try:
            # We now pass a config to allow for recursion
            final_state = app.invoke({"messages": messages}, config={
                                     "recursion_limit": 10})
            final_answer = final_state['messages'][-1].content
            print(f"\nAgent: {final_answer}")
        except Exception as e:
            print(f"An error occurred: {e}")
