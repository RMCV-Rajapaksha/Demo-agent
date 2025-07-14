from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor
from dotenv import load_dotenv
from langsmith import traceable
import os
import json

# Load environment variables (from .env file)
load_dotenv()

# Define your booking functions
def book_hotel(hotel_name: str):
    """Book a hotel"""
    return f"Successfully booked a stay at {hotel_name}."

def book_flight(from_airport: str, to_airport: str):
    """Book a flight"""
    return f"Successfully booked a flight from {from_airport} to {to_airport}."


# Create individual agent assistants
flight_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_flight],
    prompt="You are a flight booking assistant",
    name="flight_assistant"
)

hotel_assistant = create_react_agent(
    model="openai:gpt-4o",
    tools=[book_hotel],
    prompt="You are a hotel booking assistant",
    name="hotel_assistant"
)

# Create the supervisor agent
supervisor = create_supervisor(
    agents=[flight_assistant, hotel_assistant],
    model=ChatOpenAI(model="gpt-4o"),
    prompt=(
        "You manage a hotel booking assistant, a flight booking assistant, "
       )
).compile()

# Trace the whole execution using LangSmith
@traceable(name="Booking Flow Execution")
def run_booking_flow():
    results = []
    for chunk in supervisor.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": "book a flight from BOS to JFK and a stay at McKittrick Hotel"
                }
            ]
        }
    ):
        print(json.dumps(chunk, indent=2, default=str))
        print("\n")
        results.append(chunk)
    return results

if __name__ == "__main__":
   
    
    run_booking_flow()