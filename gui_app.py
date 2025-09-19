import streamlit as st
from Agentic_Query_System import app as agent_app, system_prompt_text
from langchain_core.messages import HumanMessage, SystemMessage

# --- Page Configuration ---
st.set_page_config(
    page_title="Agentic Query System",
    page_icon="🤖",
    layout="centered",
)

# --- Title and Description ---
st.title("🤖 Agentic Query System")
st.markdown("""
Welcome! Ask me anything about the camera feeds.
I can help you with questions about camera locations, quality, codecs, and what different technical terms mean.
""")

# --- Session State Initialization ---
# This is where we will store the conversation history for the session.
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append(
        {"role": "assistant", "content": "How can I help you today?"}
    )

# --- Display Chat History ---
# This loop will display all the messages in the history.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Handle User Input ---
if prompt := st.chat_input("Ask a question..."):
    # Add user message to the chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # --- Get Agent's Response ---
    with st.spinner("Agent is thinking..."):
        try:
            # Construct the messages list for the agent, including the system prompt
            agent_messages = [
                SystemMessage(content=system_prompt_text),
                HumanMessage(content=prompt)
            ]
            
            # Invoke the agent with the proper history
            final_state = agent_app.invoke(
                {"messages": agent_messages},
                config={"recursion_limit": 10}
            )
            agent_response = final_state['messages'][-1].content

            # Display the agent's response
            with st.chat_message("assistant"):
                st.markdown(agent_response)
            
            # Add agent's response to the chat history
            st.session_state.messages.append({"role": "assistant", "content": agent_response})

        except Exception as e:
            error_message = f"An error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append({"role": "assistant", "content": error_message})