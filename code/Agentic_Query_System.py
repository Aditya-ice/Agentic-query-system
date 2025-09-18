import os
import json
from typing import TypedDict, Sequence
import pandas as pd
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolExecutor

# --- Part 1: Import and Load Data ---
# Changed import to match corrected filename
from data_loader import load_data, SEMANTIC_MAPPING

all_data = load_data()
if not all_data:
    raise ValueError("Data could not be loaded. Please check your data files and paths.")

# Use the processed dataframes
camera_feeds_df = all_data["camera_feeds"]
table_definitions_df = all_data["table_definitions"]


# --- Part 2: Define the Tools (Corrected and Improved) ---

@tool
def find_cameras(location: str = None, min_resolution: str = None, min_bitrate: int = None, codec: str = None) -> str:
    """
    Finds cameras based on a set of optional filter criteria.
    Args:
        location: The geographical theater of the camera (e.g., 'PAC', 'EUR', 'ME').
        min_resolution: The minimum resolution required (e.g., '1080p', '4K').
        min_bitrate: The minimum estimated bitrate in kbps (e.g., 4000).
        codec: The codec used by the camera (e.g., 'H264', 'H265').
    """
    # The tool now works with the transformed and corrected column names
    query_df = camera_feeds_df.copy()
    if location:
        # Match short codes like 'PAC'
        query_df = query_df[query_df['location'].str.upper() == location.upper()]
    if codec:
        query_df = query_df[query_df['codec'].str.upper() == codec.upper()]
    if min_bitrate:
        query_df = query_df[query_df['bitrate'] >= min_bitrate]
    if min_resolution:
        resolutions = ['480p', '720p', '1080p', '1440p', '4K']
        if min_resolution in resolutions:
            allowed_resolutions = resolutions[resolutions.index(min_resolution):]
            query_df = query_df[query_df['resolution'].isin(allowed_resolutions)]

    if query_df.empty:
        return "No cameras found matching the criteria."

    # Return a subset of columns to avoid overwhelming the model
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
        # The tool now correctly looks in the definitions dataframe
        definition = table_definitions_df.loc[parameter_name.upper()]['description']
        return json.dumps({"parameter": parameter_name, "description": definition})
    except KeyError:
        return f"Parameter '{parameter_name}' not found in the definitions table."


# --- Part 3: Configure the Agent ---

tools = [find_cameras, get_parameter_definition]
tool_executor = ToolExecutor(tools)

model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
model = model.bind_tools(tools)

# --- Part 4: Define the Agent State and Graph ---

class AgentState(TypedDict):
    messages: Sequence[BaseMessage]

def call_model(state):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

def call_tool(state):
    last_message = state['messages'][-1]
    # Handle multiple tool calls in the future if needed
    tool_messages = []
    for tool_call in last_message.tool_calls:
        action = tool_call
        tool_output = tool_executor.invoke(action)
        tool_messages.append(ToolMessage(content=str(tool_output), tool_call_id=action['id']))
    return {"messages": tool_messages}

def should_continue(state):
    last_message = state['messages'][-1]
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "end"
    else:
        return "continue"

# --- Part 5: Build and Run the Graph ---

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)
workflow.set_entry_point("agent")
workflow.add_conditional_edges(
    "agent",
    should_continue,
    {"continue": "action", "end": END},
)
workflow.add_edge("action", "agent")
app = workflow.compile()

# --- Main execution block ---
if __name__ == "__main__":
    print("Agent is ready. Type 'exit' to quit.")
    system_prompt = f"""
    You are an expert AI assistant for a video analyst. Your goal is to answer questions about a dataset of 100 camera feeds.
    You have access to a set of tools to find information.

    When interpreting user queries, use the following semantic definitions which map natural language to technical parameters:
    {json.dumps(SEMANTIC_MAPPING, indent=2)}

    The user might use terms like 'region' or 'theater'; both map to the 'location' parameter in your tools, which uses abbreviations like 'PAC', 'EUR', etc.
    Always provide your final answer in a clear, easy-to-read format.
    If you use the find_cameras tool and it returns a lot of data, summarize it unless asked for full details. For example, state the number of cameras found and show the first 3.
    """

    while True:
        user_query = input("You: ")
        if user_query.lower() == 'exit':
            break

        messages = [
            HumanMessage(content=system_prompt),
            HumanMessage(content=user_query)
        ]

        try:
            final_state = app.invoke({"messages": messages})
            final_answer = final_state['messages'][-1].content
            print(f"Agent: {final_answer}")
        except Exception as e:
            print(f"An error occurred: {e}")
