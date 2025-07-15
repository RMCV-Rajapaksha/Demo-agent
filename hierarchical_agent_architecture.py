import os
from typing import Literal, TypedDict, List
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.types import Command
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv()



# Initialize the model
model = ChatOpenAI(model="gpt-4o", temperature=0)

# Define structured output for routing decisions
class RoutingDecision(BaseModel):
    next_agent: Literal["research_agent", "writing_agent", "__end__"]
    reasoning: str

class TeamRoutingDecision(BaseModel):
    next_team: Literal["research_team", "content_team", "__end__"]
    reasoning: str

# =============================================================================
# TEAM 1: RESEARCH TEAM
# =============================================================================

class ResearchTeamState(MessagesState):
    """State for the research team with routing information"""
    pass

def research_supervisor(state: ResearchTeamState) -> Command[Literal["research_agent", "fact_checker", "__end__"]]:
    """Supervises the research team - decides between research agent and fact checker"""
    
    print("🔍 Research Supervisor is analyzing the conversation...")


    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Create routing prompt
    routing_prompt = f"""
    You are a research team supervisor. Based on the conversation history, decide what to do next:
    
    - research_agent: For gathering information, conducting research, finding data
    - fact_checker: For verifying information accuracy, checking sources
    - __end__: If the research task is complete
    
    Last message: {last_message.content if last_message else "No previous messages"}
    
    Decide the next step and provide reasoning.
    """
    
    response = model.with_structured_output(RoutingDecision).invoke([
        SystemMessage(content=routing_prompt)
    ])
    
    print(f"🔍 Research Supervisor: Routing to {response.next_agent} - {response.reasoning}")
    return Command(goto=response.next_agent)

def research_agent(state: ResearchTeamState) -> Command[Literal["research_supervisor"]]:
    """Conducts research and gathers information"""
    
    print("🔎 Research Agent is gathering information...")

    messages = state["messages"]
    
    research_prompt = """
    You are a research agent. Your job is to gather information and provide detailed research on topics.
    Based on the user's request, provide comprehensive research findings.
    """
    
    response = model.invoke([
        SystemMessage(content=research_prompt),
        *messages
    ])
    
    print(f"📚 Research Agent: {response.content[:100]}...")
    return Command(
        goto="research_supervisor", 
        update={"messages": [AIMessage(content=f"[Research Agent] {response.content}")]}
    )

def fact_checker(state: ResearchTeamState) -> Command[Literal["research_supervisor"]]:
    """Verifies and fact-checks information"""
    

    print("✅ Fact Checker is verifying information...")

    messages = state["messages"]
    
    fact_check_prompt = """
    You are a fact-checking agent. Review the previous research and verify its accuracy.
    Highlight any potential issues or confirm the reliability of the information.
    """
    
    response = model.invoke([
        SystemMessage(content=fact_check_prompt),
        *messages
    ])
    
    print(f"✅ Fact Checker: {response.content[:100]}...")
    return Command(
        goto="research_supervisor", 
        update={"messages": [AIMessage(content=f"[Fact Checker] {response.content}")]}
    )

# Build research team graph
research_team_builder = StateGraph(ResearchTeamState)
research_team_builder.add_node("research_supervisor", research_supervisor)
research_team_builder.add_node("research_agent", research_agent)
research_team_builder.add_node("fact_checker", fact_checker)
research_team_builder.add_edge(START, "research_supervisor")
research_team_graph = research_team_builder.compile()

# =============================================================================
# TEAM 2: CONTENT CREATION TEAM
# =============================================================================

class ContentTeamState(MessagesState):
    """State for the content creation team"""
    pass

def content_supervisor(state: ContentTeamState) -> Command[Literal["writer_agent", "editor_agent", "__end__"]]:
    """Supervises the content team - decides between writer and editor"""
    

    print("✍️ Content Supervisor is analyzing the conversation...")

    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    routing_prompt = f"""
    You are a content team supervisor. Based on the conversation, decide what to do next:
    
    - writer_agent: For creating content, writing articles, drafting text
    - editor_agent: For editing, proofreading, improving existing content
    - __end__: If the content creation task is complete
    
    Last message: {last_message.content if last_message else "No previous messages"}
    
    Decide the next step and provide reasoning.
    """
    
    response = model.with_structured_output(RoutingDecision).invoke([
        SystemMessage(content=routing_prompt)
    ])
    
    print(f"✍️ Content Supervisor: Routing to {response.next_agent} - {response.reasoning}")
    return Command(goto=response.next_agent)

def writer_agent(state: ContentTeamState) -> Command[Literal["content_supervisor"]]:
    """Creates written content based on requirements"""
    
    print("📝 Writer Agent is drafting content...")

    messages = state["messages"]
    
    writing_prompt = """
    You are a content writer. Create engaging, well-structured content based on the user's request.
    Use any research provided to create informative and accurate content.
    """
    
    response = model.invoke([
        SystemMessage(content=writing_prompt),
        *messages
    ])
    
    print(f"📝 Writer Agent: {response.content[:100]}...")
    return Command(
        goto="content_supervisor", 
        update={"messages": [AIMessage(content=f"[Writer Agent] {response.content}")]}
    )

def editor_agent(state: ContentTeamState) -> Command[Literal["content_supervisor"]]:
    """Edits and improves existing content"""
    
    print("✏️ Editor Agent is reviewing content...")

    messages = state["messages"]
    
    editing_prompt = """
    You are an editor. Review the content and improve it for clarity, accuracy, and engagement.
    Provide suggestions or a revised version.
    """
    
    response = model.invoke([
        SystemMessage(content=editing_prompt),
        *messages
    ])
    
    print(f"✏️ Editor Agent: {response.content[:100]}...")
    return Command(
        goto="content_supervisor", 
        update={"messages": [AIMessage(content=f"[Editor Agent] {response.content}")]}
    )

# Build content team graph
content_team_builder = StateGraph(ContentTeamState)
content_team_builder.add_node("content_supervisor", content_supervisor)
content_team_builder.add_node("writer_agent", writer_agent)
content_team_builder.add_node("editor_agent", editor_agent)
content_team_builder.add_edge(START, "content_supervisor")
content_team_graph = content_team_builder.compile()

# =============================================================================
# TOP-LEVEL SUPERVISOR
# =============================================================================

def top_level_supervisor(state: MessagesState) -> Command[Literal["research_team", "content_team", "__end__"]]:
    """Top-level supervisor that coordinates between teams"""
    
    messages = state["messages"]
    last_message = messages[-1] if messages else None
    
    # Analyze the conversation to determine which team should handle the request
    routing_prompt = f"""
    You are the top-level supervisor coordinating between specialized teams:
    
    - research_team: For research tasks, fact-finding, data gathering, analysis
    - content_team: For writing, editing, content creation tasks
    - __end__: If the overall task is complete and satisfactory
    
    Current conversation context:
    {[msg.content for msg in messages[-3:]] if len(messages) >= 3 else [msg.content for msg in messages]}
    
    Last message: {last_message.content if last_message else "No previous messages"}
    
    Determine which team should handle this next, or if we're done.
    """
    
    response = model.with_structured_output(TeamRoutingDecision).invoke([
        SystemMessage(content=routing_prompt)
    ])
    
    print(f"🎯 Top Supervisor: Routing to {response.next_team} - {response.reasoning}")
    return Command(goto=response.next_team)

# =============================================================================
# MAIN HIERARCHICAL GRAPH
# =============================================================================

# Build the main graph
main_builder = StateGraph(MessagesState)
main_builder.add_node("top_level_supervisor", top_level_supervisor)
main_builder.add_node("research_team", research_team_graph)
main_builder.add_node("content_team", content_team_graph)

# Define the flow
main_builder.add_edge(START, "top_level_supervisor")
main_builder.add_edge("research_team", "top_level_supervisor")
main_builder.add_edge("content_team", "top_level_supervisor")

# Compile the main graph
hierarchical_graph = main_builder.compile()

# =============================================================================
# DEMO FUNCTION
# =============================================================================

def run_demo():
    """Run a demonstration of the hierarchical agent system"""
    
    print("🚀 Starting Hierarchical Agent Architecture Demo")
    print("=" * 60)
    
    # Test case 1: Research task
    print("\n📋 Test Case 1: Research Request")
    print("-" * 40)
    
    initial_state = {
        "messages": [
            HumanMessage(content="I need to research the benefits of renewable energy and then write a short article about it.")
        ]
    }
    
    try:
        result = hierarchical_graph.invoke(initial_state)
        
        print(f"\n🎉 Final Result:")
        print("-" * 40)
        for msg in result["messages"]:
            if isinstance(msg, AIMessage):
                print(f"{msg.content[:200]}...")
                print()
    
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Note: Make sure to set your OPENAI_API_KEY environment variable")

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Uncomment the line below to run the demo
    # Make sure to set your OpenAI API key first!
    
    print("To run this demo:")
    print("1. Set your OpenAI API key: os.environ['OPENAI_API_KEY'] = 'your-key'")
    print("2. Install requirements: pip install langchain-openai langgraph")
    print("3. Uncomment the run_demo() call below")
    print("4. Run the script")
    

    run_demo()
