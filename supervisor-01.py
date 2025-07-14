from mailbox import BabylMessage
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv

import os
import json

# Load environment variables (from .env file)
load_dotenv()


# Initialize the LLM
model = ChatOpenAI(model="gpt-4o", temperature=0)

def supervisor(state: MessagesState) -> Command[Literal["agent_1", "agent_2", END]]:
    """Supervisor uses LLM to decide which agent to call next"""
    messages = state.get("messages", [])
    
    print("🎯 SUPERVISOR: Using LLM to decide which agent to call...")
    
    if not messages:
        return Command(goto="agent_1")
    
    # Get the user's request
    user_request = state["messages"][-1].content if messages else "hello"

    messages: list[BabylMessage] = state["messages"]
    
    
    # Use LLM to make routing decision
    supervisor_prompt = f"""
    You are a supervisor managing two agents:
    - agent_1: Math Agent (handles math, calculations, numbers)
    - agent_2: Joke Agent (handles jokes, humor, entertainment)
    
    User request: "{user_request}"


    Which agent should handle this? Respond with ONLY one word:
    - "agent_1" for math/calculation requests
    - "agent_2" for joke/humor requests  
    - "END" if the request doesn't fit either category 
       
    """
    
    response = model.invoke([SystemMessage(content=supervisor_prompt)])
    AIMessage(content=response.content)
    decision = response.content.strip().lower()
    
    if "agent_1" in decision:
        print("🎯 SUPERVISOR: LLM chose Math Agent")
        return Command(goto="agent_1")
    elif "agent_2" in decision:
        print("🎯 SUPERVISOR: LLM chose Joke Agent")
        return Command(goto="agent_2")
    else:
        print("🎯 SUPERVISOR: LLM chose to END")
        return Command(goto=END)

def agent_1(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Math Agent uses LLM to solve math problems"""
    messages = state.get("messages", [])
    user_request = messages[0] if messages else ""
    
    print("🔢 MATH AGENT: Using LLM to solve math problem...")
    
    # Use LLM for math
    math_prompt = f"""
    You are a math expert. Solve this problem clearly and concisely:
    "{user_request}"
    
    Provide the answer with a brief explanation.
    """
    
    response = model.invoke([SystemMessage(content=math_prompt)])
    
    print(f"🔢 MATH AGENT: {response.content}")
    
    return Command(
        goto="supervisor",
        update={"messages": messages + [f"Math Agent: {response.content}"]},
    )

def agent_2(state: MessagesState) -> Command[Literal["supervisor"]]:
    """Joke Agent uses LLM to tell jokes"""
    messages = state.get("messages", [])
    user_request = messages[0] if messages else ""
    
    print("😄 JOKE AGENT: Using LLM to create a joke...")
    
    # Use LLM for jokes
    joke_prompt = f"""
    You are a comedian. Create a funny, clean joke based on this request:
    "{user_request}"
    
    Make it clever and family-friendly.
    """
    
    response = model.invoke([SystemMessage(content=joke_prompt)])
    
    print(f"😄 JOKE AGENT: {response.content}")
    
    return Command(
        goto="supervisor",
        update={"messages": messages + [f"Joke Agent: {response.content}"]},
    )

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("supervisor", supervisor)
builder.add_node("agent_1", agent_1)
builder.add_node("agent_2", agent_2)

# Start with supervisor
builder.add_edge(START, "supervisor")
builder.add_edge("supervisor",END)

# Compile the graph
graph = builder.compile()

def run_demo():
    """Run demo with LLM-powered agents"""
    print("=" * 60)
    print("🤖 SUPERVISOR DEMO WITH LLM")
    print("=" * 60)
    print("Note: Make sure to set OPENAI_API_KEY environment variable")
    print("=" * 60)
    
    # Test cases
    test_cases = "Tell me a story about computers"

    initial_state = {
        "messages": [HumanMessage(content=test_cases)]
    }

    invoke_result = graph.invoke(initial_state)
    
    

if __name__ == "__main__":
    run_demo()