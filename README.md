# Multi-Agent AI Demo Project

This project demonstrates several multi-agent architectures using the LangChain, LangGraph, and OpenAI APIs. It includes examples of hierarchical agent routing, supervisor-agent architectures, and creative writing/reasoning workflows. The agents are coordinated by a supervisor, which delegates tasks to specialized agents (e.g., for booking, writing, math, research, etc.).

## Features

- **Supervisor-Agent Architectures**: Modular design with a supervisor agent that routes user requests to the appropriate specialist agent.
- **Specialist Agents**: Includes agents for booking (flights/hotels), math, writing, research, editing, and more.
- **Hierarchical Agent Graphs**: Demonstrates multi-level agent teams (e.g., research team, content team) coordinated by a top-level supervisor.
- **Creative Writing Workflow**: Multi-step story writing, editing, and critique using LLMs.
- **Tool-Calling and State Management**: Uses LangGraph's stateful agent orchestration and tool-calling patterns.
- **Jupyter Notebook Examples**: Interactive notebooks for experimentation and demonstration.

## Project Structure

- `supervisor.py` — Booking assistant demo with supervisor routing between flight and hotel agents.
- `supervisor-toolcall.py` — Supervisor agent that routes to math, writing, or research specialists.
- `hierarchical_agent_architecture.py` — Hierarchical agent system with research and content teams, each with their own supervisor and agents.
- `network.py` — Multi-agent creative writing workflow (story writer, editor, critic) using a state graph.
- `network.ipynb` — Jupyter notebook for interactive experimentation with the booking supervisor.
- `.env` — Environment variables (API keys, project config).

## Requirements

- Python 3.10+
- [OpenAI API key](https://platform.openai.com/account/api-keys)
- [LangChain](https://python.langchain.com/)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [langchain-openai](https://github.com/langchain-ai/langchain-openai)
- [langgraph-supervisor](https://github.com/langchain-ai/langgraph-supervisor)
- [python-dotenv](https://pypi.org/project/python-dotenv/)
- [pydantic](https://pydantic-docs.helpmanual.io/)

Install all dependencies with:

```bash
pip install -U langchain langgraph langchain-openai langgraph-supervisor python-dotenv pydantic
```

## Usage

1. **Set your OpenAI API key**
   - Create a `.env` file in the project root:
     ```
     OPENAI_API_KEY=your-openai-key-here
     ```
2. **Run a demo script**
   - For booking assistant:
     ```bash
     python supervisor.py
     ```
   - For creative writing workflow:
     ```bash
     python network.py
     ```
   - For hierarchical agent architecture:
     ```bash
     python hierarchical_agent_architecture.py
     ```
   - For tool-calling supervisor:
     ```bash
     python supervisor-toolcall.py
     ```
3. **Try the Jupyter notebook**
   - Open `network.ipynb` in VS Code or JupyterLab for interactive demos.

## Customization
- Add or modify agents in the Python scripts to suit your use case.
- Change prompts, tools, or agent logic for different workflows.

## License
This project is for educational and demonstration purposes. See individual package licenses for dependencies.

---

**Author:** Vishwajith
