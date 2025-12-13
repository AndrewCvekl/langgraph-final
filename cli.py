#!/usr/bin/env python3
"""CLI runner for the customer support bot.

Provides an interactive terminal interface with:
- Token streaming
- Node transition display
- Tool call display
- HITL interrupt handling
"""

import os
import sys
import uuid
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables
load_dotenv()

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.graph import compile_graph
from src.state import get_initial_state
from src.db import initialize_database, DEMO_CUSTOMER_ID


# ANSI color codes for terminal output
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'


def print_header():
    """Print the CLI header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}║     🎵 Music Store Customer Support Bot (LangGraph Demo)     ║{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}╚══════════════════════════════════════════════════════════════╝{Colors.ENDC}")
    print(f"{Colors.DIM}Type 'quit' or 'exit' to end the conversation.{Colors.ENDC}")
    print(f"{Colors.DIM}Type 'help' for example commands.{Colors.ENDC}")
    print()


def print_help():
    """Print example commands."""
    print(f"\n{Colors.BOLD}Example things you can ask:{Colors.ENDC}")
    print(f"  {Colors.GREEN}• What genres do you have?{Colors.ENDC}")
    print(f"  {Colors.GREEN}• Show me artists in Rock{Colors.ENDC}")
    print(f"  {Colors.GREEN}• What albums does AC/DC have?{Colors.ENDC}")
    print(f"  {Colors.GREEN}• Find tracks with 'Back in Black'{Colors.ENDC}")
    print(f"  {Colors.GREEN}• Show me my profile{Colors.ENDC}")
    print(f"  {Colors.GREEN}• What are my recent purchases?{Colors.ENDC}")
    print(f"  {Colors.GREEN}• I want to change my email{Colors.ENDC}")
    print(f"  {Colors.GREEN}• I remember some lyrics: 'back in black I hit the sack'{Colors.ENDC}")
    print(f"  {Colors.GREEN}• I want to buy track 1{Colors.ENDC}")
    print()


def print_node_event(node_name: str, event_type: str):
    """Print node transition events."""
    if event_type == "start":
        print(f"\n{Colors.DIM}[{node_name}] Starting...{Colors.ENDC}")
    else:
        print(f"{Colors.DIM}[{node_name}] Finished{Colors.ENDC}")


def print_tool_call(tool_name: str, tool_args: dict):
    """Print tool invocation."""
    print(f"\n{Colors.YELLOW}🔧 Tool: {tool_name}{Colors.ENDC}")
    if tool_args:
        args_str = ", ".join(f"{k}={v}" for k, v in tool_args.items() if k != "config")
        print(f"{Colors.DIM}   Args: {args_str}{Colors.ENDC}")


def print_tool_result(tool_name: str, result: str):
    """Print tool result (truncated if too long)."""
    max_len = 200
    result_str = str(result)
    if len(result_str) > max_len:
        result_str = result_str[:max_len] + "..."
    print(f"{Colors.DIM}   Result: {result_str}{Colors.ENDC}")


def handle_interrupt(interrupt_value: dict) -> str:
    """Handle HITL interrupt and get user input."""
    interrupt_type = interrupt_value.get("type", "confirm")
    title = interrupt_value.get("title", "Input Required")
    message = interrupt_value.get("message", "Please respond:")
    options = interrupt_value.get("options", [])
    
    print(f"\n{Colors.BOLD}{Colors.YELLOW}⏸️  {title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{message}{Colors.ENDC}")
    
    if options:
        print(f"{Colors.DIM}Options: {', '.join(options)}{Colors.ENDC}")
    
    while True:
        response = input(f"{Colors.GREEN}Your response: {Colors.ENDC}").strip()
        if response:
            return response
        print(f"{Colors.RED}Please enter a response.{Colors.ENDC}")


def run_cli():
    """Main CLI loop."""
    print_header()
    
    # Initialize database
    print(f"{Colors.DIM}Initializing database...{Colors.ENDC}")
    initialize_database()
    
    # Compile the graph
    print(f"{Colors.DIM}Loading the support bot...{Colors.ENDC}")
    graph = compile_graph()
    
    # Create a thread ID for this session
    thread_id = str(uuid.uuid4())
    config = {
        "configurable": {
            "thread_id": thread_id,
            "customer_id": DEMO_CUSTOMER_ID,
        }
    }
    
    print(f"\n{Colors.GREEN}Ready! Customer ID: {DEMO_CUSTOMER_ID} (Demo Account){Colors.ENDC}")
    print(f"{Colors.DIM}Thread ID: {thread_id}{Colors.ENDC}\n")
    
    # Initialize state
    state = get_initial_state(customer_id=DEMO_CUSTOMER_ID)
    
    while True:
        try:
            # Get user input
            user_input = input(f"{Colors.BOLD}You: {Colors.ENDC}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\n{Colors.CYAN}Thanks for using the Music Store Support Bot! 🎵{Colors.ENDC}\n")
                break
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Add user message to state (always include customer_id for state consistency)
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "customer_id": DEMO_CUSTOMER_ID,
            }
            
            print(f"\n{Colors.BOLD}Bot:{Colors.ENDC} ", end="", flush=True)
            
            # Stream the response
            final_response = ""
            current_node = None
            
            try:
                from langgraph.types import Command
                
                def process_stream(stream_input, is_resume=False):
                    """Process a stream of events, handling interrupts recursively."""
                    nonlocal final_response, current_node
                    
                    for event in graph.stream(
                        stream_input,
                        config=config,
                        stream_mode="updates"
                    ):
                        # Check for interrupts
                        if "__interrupt__" in event:
                            interrupts = event["__interrupt__"]
                            for interrupt_info in interrupts:
                                interrupt_value = interrupt_info.value if hasattr(interrupt_info, 'value') else interrupt_info
                                response = handle_interrupt(interrupt_value)
                                
                                # Resume with the user's response (recursive call handles nested interrupts)
                                process_stream(Command(resume=response), is_resume=True)
                            continue
                        
                        # Process regular events
                        for node_name, node_output in event.items():
                            if node_name.startswith("__"):
                                continue
                            
                            if node_name != current_node:
                                if current_node is not None:
                                    print_node_event(current_node, "end")
                                current_node = node_name
                                print_node_event(node_name, "start")
                            
                            if not node_output:
                                continue
                            
                            # Handle messages
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    if isinstance(msg, AIMessage):
                                        if msg.tool_calls:
                                            for tc in msg.tool_calls:
                                                print_tool_call(tc["name"], tc.get("args", {}))
                                        elif msg.content:
                                            print(f"\n{msg.content}")
                                            final_response = msg.content
                                    elif isinstance(msg, ToolMessage):
                                        print_tool_result(msg.name, msg.content)
                
                # Start processing the stream
                process_stream(input_state)
                
                if current_node:
                    print_node_event(current_node, "end")
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Interrupted. Type 'quit' to exit.{Colors.ENDC}")
                continue
            
            print()  # Add newline after response
            
        except KeyboardInterrupt:
            print(f"\n\n{Colors.CYAN}Goodbye! 🎵{Colors.ENDC}\n")
            break
        except Exception as e:
            print(f"\n{Colors.RED}Error: {e}{Colors.ENDC}")
            import traceback
            traceback.print_exc()
            continue


if __name__ == "__main__":
    run_cli()

