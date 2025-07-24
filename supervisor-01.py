import getpass
import os
from typing import Annotated
from langchain_tavily import TavilySearch
from langgraph.prebuilt import create_react_agent, InjectedState
from langchain_core.messages import convert_to_messages
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.graph import StateGraph, START, MessagesState, END
from langgraph.types import Command, Send
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")

_set_if_undefined("OPENAI_API_KEY")
_set_if_undefined("TAVILY_API_KEY")

def _set_if_undefined(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"Please provide your {var}")


def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def pretty_print_messages(update, last_message=False):
    is_subgraph = False
    if isinstance(update, tuple):
        ns, update = update
        if len(ns) == 0:
            return
        graph_id = ns[-1].split(":")[0]
        print(f"Update from subgraph {graph_id}:")
        print("\n")
        is_subgraph = True

    for node_name, node_update in update.items():
        update_label = f"Update from node {node_name}:"
        if is_subgraph:
            update_label = "\t" + update_label
        print(update_label)
        print("\n")

        messages = convert_to_messages(node_update["messages"])
        if last_message:
            messages = messages[-1:]

        for m in messages:
            pretty_print_message(m, indent=is_subgraph)
        print("\n")


def add(a: float, b: float):
    """Add two numbers."""
    return a + b


def multiply(a: float, b: float):
    """Multiply two numbers."""
    return a * b


def divide(a: float, b: float):
    """Divide two numbers."""
    return a / b


def create_task_description_handoff_tool(*, agent_name: str, description: str | None = None):
    name = f"transfer_to_{agent_name}"
    description = description or f"Ask {agent_name} for help."

    @tool(name, description=description)
    def handoff_tool(
        task_description: Annotated[
            str,
            "Description of what the next agent should do, including all of the relevant context.",
        ],
        state: Annotated[MessagesState, InjectedState],
    ) -> Command:
        task_description_message = {"role": "user", "content": task_description}
        agent_input = {**state, "messages": [task_description_message]}
        return Command(
            goto=[Send(agent_name, agent_input)],
            graph=Command.PARENT,
        )

    return handoff_tool


def main():
    # Set up API keys
    _set_if_undefined("OPENAI_API_KEY")
    _set_if_undefined("TAVILY_API_KEY")

    # Create web search tool
    web_search = TavilySearch(max_results=3)

    # Create research agent
    research_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[web_search],
        prompt=(
            "You are a research agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with research-related tasks, DO NOT do any math\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="research_agent",
    )

    # Create math agent
    math_agent = create_react_agent(
        model="openai:gpt-4o",
        tools=[add, multiply, divide],
        prompt=(
            "You are a math agent.\n\n"
            "INSTRUCTIONS:\n"
            "- Assist ONLY with math-related tasks\n"
            "- After you're done with your tasks, respond to the supervisor directly\n"
            "- Respond ONLY with the results of your work, do NOT include ANY other text."
        ),
        name="math_agent",
    )

    # Create handoff tools
    assign_to_research_agent_with_description = create_task_description_handoff_tool(
        agent_name="research_agent",
        description="Assign task to a researcher agent.",
    )

    assign_to_math_agent_with_description = create_task_description_handoff_tool(
        agent_name="math_agent",
        description="Assign task to a math agent.",
    )

    # Create supervisor agent
    supervisor_agent_with_description = create_react_agent(
        model="openai:gpt-4o-mini",
        tools=[
            assign_to_research_agent_with_description,
            assign_to_math_agent_with_description,
        ],
        prompt=(
            "You are a supervisor managing two agents:\n"
            "- a research agent. Assign research-related tasks to this assistant\n"
            "- a math agent. Assign math-related tasks to this assistant\n"
            "Assign work to one agent at a time, do not call agents in parallel.\n"
            "Do not do any work yourself."
        ),
        name="supervisor",
    )

    # Create supervisor graph
    supervisor_with_description = (
        StateGraph(MessagesState)
        .add_node(
            supervisor_agent_with_description, destinations=("research_agent", "math_agent")
        )
        .add_node(research_agent)
        .add_node(math_agent)
        .add_edge(START, "supervisor")
        .add_edge("research_agent", "supervisor")
        .add_edge("math_agent", "supervisor")
        .compile()
    )

    # Run the multi-agent system
    print("Running multi-agent supervisor system...")
    for chunk in supervisor_with_description.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "find US and New York state GDP in 2024. what % of US GDP was New York state?",
                }
            ]
        },
        subgraphs=True,
    ):
        pretty_print_messages(chunk, last_message=True)


if __name__ == "__main__":
    main()