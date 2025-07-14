from typing import Literal, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from pydantic import BaseModel
import json
from dotenv import load_dotenv

import os
import json

# Load environment variables (from .env file)
load_dotenv()

# Configure the model
model = ChatOpenAI(model="gpt-4o",temperature=0.7)

class RouteDecision(BaseModel):
    """Structured output for routing decisions"""
    next_agent: Literal["story_writer", "editor", "critic", "__end__"]
    reasoning: str

def story_writer(state: MessagesState) -> Command[Literal["editor", "critic", END]]:
    """Agent that writes creative stories"""
    print("📝 Story Writer is working...")
    
    # Get the latest message content
    last_message = state["messages"][-1].content if state["messages"] else "Write a short story"
    
    # Create a system message for the story writer
    story_prompt = [
        SystemMessage(content="""You are a creative story writer. Your job is to:
1. Write engaging short stories (2-3 paragraphs)
2. Create vivid characters and settings
3. Include compelling plot elements
4. After writing, decide if the story needs editing, criticism, or is complete

Respond with a JSON object containing:
- content: your story
- next_agent: "editor" (if needs editing), "critic" (if needs feedback), or "__end__" (if complete)
- reasoning: why you chose that next step"""),
        HumanMessage(content=last_message)
    ]
    
    response = model.invoke(story_prompt)
  
    
    try:
        # Parse the JSON response
        result = json.loads(response.content)
        content = result.get("content", response.content)
        next_agent = result.get("next_agent", "editor")
        reasoning = result.get("reasoning", "Continuing workflow")
        
        print(f"✅ Story Writer completed. Next: {next_agent}")
        print(f"Reasoning: {reasoning}")
        
        return Command(
            goto=next_agent,
            update={"messages": [AIMessage(content=content)]},
        )
    except json.JSONDecodeError:
        # Fallback if JSON parsing fails
        return Command(
            goto="editor",
            update={"messages": [AIMessage(content=response.content)]},
        )

def editor(state: MessagesState) -> Command[Literal["story_writer", "critic", END]]:
    """Agent that edits and improves content"""
    print("✏️ Editor is working...")
    
    last_message = state["messages"][-1].content
    
    edit_prompt = [
        SystemMessage(content="""You are a professional editor. Your job is to:
1. Review the provided content for grammar, style, and flow
2. Make improvements while preserving the original voice
3. Suggest structural changes if needed
4. Decide next steps in the workflow

Respond with a JSON object containing:
- content: your edited version
- next_agent: "story_writer" (if major rewrites needed), "critic" (for final review), or "__end__" (if polished)
- reasoning: explanation of your edits and next step"""),
        HumanMessage(content=f"Please edit this content:\n\n{last_message}")
    ]
    
    response = model.invoke(edit_prompt)
    
    try:
        result = json.loads(response.content)
        content = result.get("content", response.content)
        next_agent = result.get("next_agent", "critic")
        reasoning = result.get("reasoning", "Continuing workflow")
        
        print(f"✅ Editor completed. Next: {next_agent}")
        print(f"Reasoning: {reasoning}")
        
        return Command(
            goto=next_agent,
            update={"messages": [AIMessage(content=content)]},
        )
    except json.JSONDecodeError:
        return Command(
            goto="critic",
            update={"messages": [AIMessage(content=response.content)]},
        )

def critic(state: MessagesState) -> Command[Literal["story_writer", "editor", END]]:
    """Agent that provides feedback and quality assessment"""
    print("🔍 Critic is analyzing...")
    
    last_message = state["messages"][-1].content
    
    critic_prompt = [
        SystemMessage(content="""You are a literary critic. Your job is to:
1. Analyze the content for strengths and weaknesses
2. Provide constructive feedback
3. Rate the overall quality (1-10)
4. Decide if more work is needed or if it's ready

Respond with a JSON object containing:
- content: your critical analysis and final version (if approving)
- next_agent: "story_writer" (if needs major changes), "editor" (if needs minor fixes), or "__end__" (if approved)
- reasoning: your assessment and recommendation
- quality_score: numerical rating 1-10"""),
        HumanMessage(content=f"Please review this content:\n\n{last_message}")
    ]
    
    response = model.invoke(critic_prompt)
    
    try:
        result = json.loads(response.content)
        content = result.get("content", response.content)
        next_agent = result.get("next_agent", "__end__")
        reasoning = result.get("reasoning", "Review complete")
        quality_score = result.get("quality_score", "N/A")
        
        print(f"✅ Critic completed. Quality Score: {quality_score}/10")
        print(f"Next: {next_agent}")
        print(f"Reasoning: {reasoning}")
        
        return Command(
            goto=next_agent,
            update={"messages": [AIMessage(content=content)]},
        )
    except json.JSONDecodeError:
        return Command(
            goto="__end__",
            update={"messages": [AIMessage(content=response.content)]},
        )

# Build the graph
builder = StateGraph(MessagesState)
builder.add_node("story_writer", story_writer)
builder.add_node("editor", editor)
builder.add_node("critic", critic)

# Set the entry point
builder.add_edge(START, "story_writer")

# Compile the network
network = builder.compile()

def run_demo(initial_prompt: str = "Write a short story about a time traveler who gets stuck in a mundane moment"):
    """Run the multi-agent network demo"""
    print("🚀 Starting Multi-Agent Creative Writing Network")
    print("=" * 60)
    print(f"Initial prompt: {initial_prompt}")
    print("=" * 60)
    
    # Initialize state with the user's prompt
    initial_state = {
        "messages": [HumanMessage(content=initial_prompt)]
    }
    
    # Run the network
    result = network.invoke(initial_state)
    
    print("\n" + "=" * 60)
    print("🎉 FINAL RESULT")
    print("=" * 60)
    
    # Print the final content
    if result["messages"]:
        final_content = result["messages"][-1].content
        print(final_content)
    
    return result

# Example usage
if __name__ == "__main__":
    # You'll need to set your OpenAI API key
    # import os
    # os.environ["OPENAI_API_KEY"] = "your-api-key-here"
    
    # Run the demo
    try:
        result = run_demo()
        print("\n🔄 Network execution completed successfully!")
        print(f"Total messages exchanged: {len(result['messages'])}")
    except Exception as e:
        print(f"❌ Error running demo: {e}")
        print("Make sure to set your OPENAI_API_KEY environment variable")

# Advanced usage example
def run_custom_demo():
    """Run with a custom prompt"""
    custom_prompt = "Write a mystery story about a detective who can only solve crimes on Tuesdays"
    return run_demo(custom_prompt)

# Uncomment to run custom demo
# run_custom_demo()