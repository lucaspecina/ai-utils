""" 
Based on this https://langchain-ai.github.io/langgraph/how-tos/memory/add-summary-conversation-history/#build-the-chatbot
and this https://langchain-ai.github.io/langgraph/tutorials/workflows/
and this https://langchain-ai.github.io/langgraph/concepts/memory/
"""

from typing import Literal, Annotated, Sequence
from langchain_community.llms import Ollama
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, RemoveMessage
from langgraph.graph import StateGraph, MessagesState, START, END
from langgraph.checkpoint.memory import MemorySaver
import operator

# Define the custom state with a summary field
class State(MessagesState):
    summary: str

# Initialize the Ollama model (DeepSeek-R1:8B)
llm = Ollama(model="deepseek-r1:8b", base_url="http://localhost:11434")

# Memory saver for persisting state
memory = MemorySaver()

# Function to call the model for conversation
def call_model(state: State) -> dict:
    summary = state.get("summary", "")
    if summary:
        system_message = SystemMessage(content=f"Summary of conversation so far: {summary}")
        messages = [system_message] + state["messages"]
    else:
        messages = state["messages"]
    
    # Convert messages to a string format suitable for Ollama
    prompt = "\n".join([f"{m.type}: {m.content}" for m in messages])
    response = llm.invoke(prompt)
    
    return {"messages": [AIMessage(content=response)]}

# Function to determine the next step in the workflow
def should_continue(state: State) -> Literal["summarize_conversation", END]:
    messages = state["messages"]
    if len(messages) > 6:  # Summarize if more than 6 messages
        return "summarize_conversation"
    return END

# Function to summarize the conversation
def summarize_conversation(state: State) -> dict:
    summary = state.get("summary", "")
    if summary:
        summary_prompt = (
            f"Previous summary: {summary}\n\n"
            "Extend this summary based on the new messages:\n" +
            "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
        )
    else:
        summary_prompt = (
            "Create a summary of the following conversation:\n" +
            "\n".join([f"{m.type}: {m.content}" for m in state["messages"]])
        )
    
    response = llm.invoke(summary_prompt)
    
    # Keep only the last 2 messages, remove the rest
    delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-2]]
    
    return {
        "summary": response,
        "messages": delete_messages
    }

# Define the workflow
workflow = StateGraph(State)

# Add nodes
workflow.add_node("conversation", call_model)
workflow.add_node("summarize_conversation", summarize_conversation)

# Define edges
workflow.add_edge(START, "conversation")
workflow.add_conditional_edges(
    "conversation",
    should_continue,
    {
        "summarize_conversation": "summarize_conversation",
        END: END
    }
)
workflow.add_edge("summarize_conversation", END)

# Compile the graph with memory
app = workflow.compile(checkpointer=memory)

# Example usage
def run_conversation():
    config = {"configurable": {"thread_id": "example_thread_1"}}
    
    # Simulate a multi-turn conversation
    inputs = [
        "Hello, how can you assist me today?",
        "Can you tell me about the weather?",
        "What about tomorrow's forecast?",
        "Any chance of rain?",
        "How about the weekend?",
        "Anything else I should know?",
        "Thanks, one more question: what's the temperature?"
    ]
    
    for user_input in inputs:
        print(f"\nUser: {user_input}")
        input_message = {"messages": [HumanMessage(content=user_input)]}
        output = app.invoke(input_message, config=config)
        
        # Print the latest AI response
        last_message = output["messages"][-1]
        print(f"AI: {last_message.content}")
        
        # Print summary if it exists
        if "summary" in output and output["summary"]:
            print(f"Summary: {output['summary']}")

if __name__ == "__main__":
    run_conversation()