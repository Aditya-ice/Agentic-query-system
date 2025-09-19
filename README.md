### Agentic Query System for Camera Feed Analysis

This project is a sophisticated agentic query system designed to help a non-expert analyst ask natural language questions about a complex video camera dataset. The system uses a Large Language Model (Google Gemini) to understand user intent, select the appropriate tools to query the data, and provide clear, human-readable answers.

The project was developed as a technical assessment for an AI Intern position, demonstrating proficiency in cutting-edge agentic workflows, LLM integration, and rapid application development.

### Features
Natural Language Understanding: Users can ask questions in plain English (e.g., "show me the clearest cameras") instead of writing complex technical queries.

Agentic Workflow Orchestration: The agent's reasoning loop is powered by LangGraph, providing a robust, stateful, and debuggable workflow for complex multi-step tasks.

Modular "MCP" Tools: All data operations are encapsulated in secure, stand-alone Python functions (@tool decorators), which the agent can intelligently select and combine to answer questions.

Semantic-to-Technical Translation: The agent is equipped with a "semantic mapping" that allows it to translate abstract concepts like "best clarity" or "high efficiency" into concrete data filters.

Dual Interfaces: The system can be run via a traditional command-line interface or a user-friendly web-based GUI built with Streamlit.

### 🛠️ Tech Stack
Language: Python 3.10+

LLM: Google Gemini 1.5 Flash

Core AI/Agent Libraries: LangChain, LangGraph

Data Handling: Pandas

GUI: Streamlit

Environment Management: venv, pip

### 📂 Project Structure
The repository is organized as follows:

/Agentic-query-system
|
|-- 📄 .env.example             # Example environment file for API key
|-- 📄 Agentic_Query_System.py   # Main agent logic and terminal interface
|-- 📄 data_loader.py             # Data loading, cleaning, and preprocessing script
|-- 📄 gui_app.py                 # Streamlit GUI application
|-- 📄 README.md                  # This file
|-- 📄 requirements.txt           # Python package dependencies
|
|-- 📁 datasets/
|   |-- 📄 Table_defs_v2.csv
|   |-- 📄 Table_feeds_v2.csv
|   |-- ... (and other data files)
|
`-- 📁 agent_env/                  # Python virtual environment (created on setup)

### 🚀 Setup and Installation
Follow these steps to get the project running on your local machine.

####1. Prerequisites
Python 3.10 or higher installed on your system.

A Google Gemini API key. You can get one from Google AI Studio.

#### 2. Clone the Repository
git clone [https://docs.github.com/en/get-started/using-github/connecting-to-github](https://docs.github.com/en/get-started/using-github/connecting-to-github)
cd Agentic-query-system

#### 3. Create and Activate the Virtual Environment
It is highly recommended to use a virtual environment to keep dependencies isolated.

#### Create the virtual environment
python -m venv agent_env

#### Activate the environment
#### On macOS/Linux:
source agent_env/bin/activate
#### On Windows:
#### .\agent_env\Scripts\activate

#### 4. Install Dependencies
Install all the required Python packages from the requirements.txt file.

pip install -r requirements.txt

#### 5. Set Up Your API Key
Create a .env file in the root of the project to securely store your Gemini API key.

Rename the example file: mv .env.example .env (or just create a new file named .env).

Open the .env file and add your API key:

GEMINI_API_KEY="YOUR_API_KEY_GOES_HERE"

### ▶️ How to Run the Application
You can run the agent in two ways:

#### 1. Run the GUI (Recommended)
To launch the user-friendly Streamlit web interface, run the following command in your terminal:

streamlit run gui_app.py

Your web browser will automatically open a new tab with the chat application.

#### 2. Run in the Terminal
To interact with the agent directly in your command line, run:

python Agentic_Query_System.py
