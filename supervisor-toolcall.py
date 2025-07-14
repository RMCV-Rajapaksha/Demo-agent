import os
from typing import Annotated
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import InjectedState, create_react_agent
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os
import json
load_dotenv()

# Initialize the model
model = ChatOpenAI(
    model="gpt-4o",
    temperature=0.1,
    # api_key=os.getenv("OPENAI_API_KEY")  # Make sure to set this
)

# Agent 1: Math Specialist
def math_agent(state: Annotated[dict, InjectedState]) -> str:
    """
    Specialized agent for mathematical calculations and problems.
    """

    print(state)
    print("🧮 Math Agent activated!")
    print(state)
    # Get the latest message from state
    messages = state.get("messages", [])
    if messages:
        user_input = messages[-1].content
    else:
        user_input = "No input provided"
    
    # Create a specialized prompt for math problems
    math_prompt = f"""
    You are a math specialist. Solve this mathematical problem step by step:
    {user_input}
    
    Provide a clear, step-by-step solution.
    """
    
    response = model.invoke([HumanMessage(content=math_prompt)])
    return f"Math Agent: {response.content}"

# Agent 2: Writing Specialist
def writing_agent(state: Annotated[dict, InjectedState]) -> str:
    """
    Specialized agent for writing tasks, editing, and language help.
    """
    print("✍️ Writing Agent activated!")
    
    # Get the latest message from state
    messages = state.get("messages", [])
    if messages:
        user_input = messages[-1].content
    else:
        user_input = "No input provided"
    
    # Create a specialized prompt for writing tasks
    writing_prompt = f"""
    You are a writing specialist. Help with this writing task:
    {user_input}
    
    Provide helpful writing assistance, grammar corrections, or creative content.
    """
    
    response = model.invoke([HumanMessage(content=writing_prompt)])
    return f"Writing Agent: {response.content}"

# Agent 3: General Research Assistant
def research_agent(state: Annotated[dict, InjectedState]) -> str:
    """
    Specialized agent for research, fact-checking, and general information.
    """
    print("🔍 Research Agent activated!")
    
    # Get the latest message from state
    messages = state.get("messages", [])
    if messages:
        user_input = messages[-1].content
    else:
        user_input = "No input provided"
    
    # Create a specialized prompt for research tasks
    research_prompt = f"""
    You are a research specialist. Provide comprehensive information about:
    {user_input}
    
    Give factual, well-organized information with key points and context.
    """
    
    response = model.invoke([HumanMessage(content=research_prompt)])
    return f"Research Agent: {response.content}"

# Define the tools (our specialized agents)
tools = [math_agent, writing_agent, research_agent]

# Create the supervisor using the prebuilt ReAct agent
# The supervisor will decide which agent to call based on the user's request
supervisor = create_react_agent(model, tools)

def demo_supervisor():
    """
    Demonstrate the supervisor agent in action
    """
    print("=== Supervisor Agent Demo ===")
    print("The supervisor will analyze your request and delegate to the appropriate specialist agent.\n")
    
    # Test cases to demonstrate different agents
    test_cases = [
        "What is the derivative of x^2 + 3x + 5?",
      
    ]
    
    for i, query in enumerate(test_cases, 1):
        print(f"\n--- Test Case {i} ---")
        print(f"User Query: {query}")
        print("-" * 50)
        
        # Create input for the supervisor
        inputs = {"messages": [HumanMessage(content=query)]}
        
        # Let the supervisor decide and execute
        try:
            result = supervisor.invoke(inputs)
            
            # Extract the final response
            final_message = result["messages"][-1].content
            print(f"Supervisor Decision & Result:\n{final_message}")
            
        except Exception as e:
            print(f"Error: {e}")
            print("Make sure your OpenAI API key is set correctly!")
        
        print("\n" + "="*60)



if __name__ == "__main__":
    # Check if API key is available
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️  Warning: OPENAI_API_KEY not found in environment variables!")
        print("Please set your OpenAI API key to run this demo.")
        print("\nYou can set it by:")
        print("1. Creating a .env file with: OPENAI_API_KEY=your_key_here")
        print("2. Or setting environment variable: export OPENAI_API_KEY=your_key_here")
        exit(1)
    
    # Run the demo
    demo_supervisor()

"""
How it works:
=============

1. **Supervisor Agent**: Uses create_react_agent() with a tool-calling LLM
   - Analyzes user input
   - Decides which specialist agent to call
   - Routes the request appropriately

2. **Specialist Agents**: Each agent has a specific domain
   - math_agent: Handles mathematical calculations
   - writing_agent: Helps with writing tasks
   - research_agent: Provides research and general information

3. **Tool-Calling Pattern**: 
   - User query → Supervisor → Agent selection → Specialized response
   - The supervisor acts as a router/coordinator
   - Each agent receives the full state via InjectedState

4. **State Management**:
   - State contains conversation history
   - Agents can access previous messages
   - Responses are automatically converted to ToolMessages

Key Benefits:
- Modular design with specialized agents
- Automatic routing based on content
- Scalable architecture for adding more agents
- Built-in state management and message handling
"""