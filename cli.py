#!/usr/bin/env python3
"""CLI runner for the customer support bot.

DEBUG MODE: Shows exactly what each node produces for message debugging.
"""

import os
import sys
import uuid
import logging
from typing import Any

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

# Load environment variables FIRST (before any service imports)
load_dotenv()

# Configure logging to show service initialization
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)

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
    MAGENTA = '\033[35m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    BG_BLUE = '\033[44m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_RED = '\033[41m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'


def print_header():
    """Print the CLI header."""
    print(f"\n{Colors.BOLD}{Colors.CYAN}╔══════════════════════════════════════════════════════════════╗{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.CYAN}║     🎵 Music Store Customer Support Bot (DEBUG MODE)         ║{Colors.ENDC}")
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
    print(f"  {Colors.GREEN}• What song is 'back in black I hit the sack'{Colors.ENDC}")
    print(f"  {Colors.GREEN}• I want to buy track 1{Colors.ENDC}")
    print()


def print_node_box(node_name: str, event_type: str, message_count: int = 0):
    """Print a clear node boundary box."""
    width = 60
    if event_type == "start":
        print(f"\n{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} {'─' * width} {Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE}  ▶ NODE: {node_name.upper():50} {Colors.ENDC}")
        print(f"{Colors.BOLD}{Colors.BG_BLUE}{Colors.WHITE} {'─' * width} {Colors.ENDC}")
    else:
        if message_count > 0:
            summary = f"({message_count} message(s) emitted)"
        else:
            summary = "(no messages)"
        print(f"{Colors.DIM}└── {node_name} finished {summary}{Colors.ENDC}")


def print_message_debug(msg: Any, node_name: str, status: str, msg_id: str = None):
    """Print a message with full debug info in a clear box."""
    # Determine the status color and symbol
    if status == "NEW":
        status_color = Colors.BG_GREEN
        symbol = "✓"
    elif status == "SKIPPED":
        status_color = Colors.BG_YELLOW
        symbol = "⊘"
    else:
        status_color = Colors.BG_MAGENTA
        symbol = "?"
    
    msg_type = type(msg).__name__
    short_id = msg_id[-8:] if msg_id else "no-id"
    
    # Header line
    print(f"\n{Colors.BOLD}{status_color}{Colors.WHITE} {symbol} {status} MESSAGE {Colors.ENDC} {Colors.DIM}[{msg_type}] id:...{short_id}{Colors.ENDC}")
    
    if status == "SKIPPED":
        # Just show it was skipped
        content_preview = str(getattr(msg, 'content', ''))[:50]
        print(f"    {Colors.DIM}Content: \"{content_preview}...\" (already seen){Colors.ENDC}")
        return
    
    # Show the actual content in a highlighted box
    content = getattr(msg, 'content', None)
    if content:
        print(f"    {Colors.BOLD}┌{'─' * 56}┐{Colors.ENDC}")
        # Word wrap content for display
        lines = content.split('\n')
        for line in lines:
            # Truncate very long lines
            if len(line) > 54:
                line = line[:51] + "..."
            print(f"    {Colors.BOLD}│{Colors.ENDC} {line:<54} {Colors.BOLD}│{Colors.ENDC}")
        print(f"    {Colors.BOLD}└{'─' * 56}┘{Colors.ENDC}")
    
    # Show tool calls if any
    tool_calls = getattr(msg, 'tool_calls', None)
    if tool_calls:
        print(f"    {Colors.YELLOW}Tool Calls: {len(tool_calls)}{Colors.ENDC}")
        for tc in tool_calls:
            print(f"      → {tc['name']}({tc.get('args', {})})")


def print_tool_result_debug(msg: ToolMessage, node_name: str):
    """Print tool result with debug info."""
    content = str(msg.content)[:100] + "..." if len(str(msg.content)) > 100 else str(msg.content)
    print(f"    {Colors.CYAN}🔧 Tool Result [{msg.name}]:{Colors.ENDC}")
    print(f"       {Colors.DIM}{content}{Colors.ENDC}")


def handle_interrupt(interrupt_value: dict) -> str:
    """Handle HITL interrupt and get user input."""
    interrupt_type = interrupt_value.get("type", "confirm")
    title = interrupt_value.get("title", "Input Required")
    message = interrupt_value.get("message", "Please respond:")
    options = interrupt_value.get("options", [])
    
    print(f"\n{Colors.BOLD}{Colors.BG_YELLOW}{Colors.WHITE} ⏸ INTERRUPT {Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.YELLOW}{title}{Colors.ENDC}")
    print(f"{Colors.CYAN}{message}{Colors.ENDC}")
    
    if options:
        print(f"{Colors.DIM}Options: {', '.join(options)}{Colors.ENDC}")
    
    while True:
        response = input(f"{Colors.GREEN}Your response: {Colors.ENDC}").strip()
        if response:
            return response
        print(f"{Colors.RED}Please enter a response.{Colors.ENDC}")


def run_cli():
    """Main CLI loop with enhanced debug output."""
    print_header()
    
    # Initialize database
    print(f"{Colors.DIM}Initializing database...{Colors.ENDC}")
    initialize_database()
    
    # Compile the graph
    print(f"{Colors.DIM}Loading the support bot...{Colors.ENDC}")
    graph = compile_graph()
    
    # Initialize and show service status
    print(f"\n{Colors.BOLD}API Service Status:{Colors.ENDC}")
    from src.tools.services import get_genius_service, get_youtube_service, get_twilio_service
    
    genius = get_genius_service()
    youtube = get_youtube_service()
    twilio = get_twilio_service()
    
    def status_icon(is_live: bool) -> str:
        return f"{Colors.GREEN}✓ LIVE{Colors.ENDC}" if is_live else f"{Colors.YELLOW}⚠ MOCK{Colors.ENDC}"
    
    print(f"  Genius:  {status_icon(genius.is_live)}")
    print(f"  YouTube: {status_icon(youtube.is_live)}")
    print(f"  Twilio:  {status_icon(twilio.is_live)}")
    
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
            print(f"\n{Colors.BOLD}{'═' * 62}{Colors.ENDC}")
            user_input = input(f"{Colors.BOLD}{Colors.GREEN}You: {Colors.ENDC}").strip()
            print(f"{Colors.BOLD}{'═' * 62}{Colors.ENDC}")
            
            if not user_input:
                continue
            
            if user_input.lower() in ("quit", "exit", "q"):
                print(f"\n{Colors.CYAN}Thanks for using the Music Store Support Bot! 🎵{Colors.ENDC}\n")
                break
            
            if user_input.lower() == "help":
                print_help()
                continue
            
            # Add user message to state
            input_state = {
                "messages": [HumanMessage(content=user_input)],
                "customer_id": DEMO_CUSTOMER_ID,
            }
            
            # Stream the response with debug output
            current_node = None
            node_message_count = 0
            seen_message_ids: set[str] = set()
            
            try:
                from langgraph.types import Command
                
                def process_stream(stream_input, is_resume=False):
                    """Process a stream of events with debug output."""
                    nonlocal current_node, node_message_count, seen_message_ids
                    
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
                                process_stream(Command(resume=response), is_resume=True)
                            continue
                        
                        # Process regular events
                        for node_name, node_output in event.items():
                            if node_name.startswith("__"):
                                continue
                            
                            # Node transition
                            if node_name != current_node:
                                if current_node is not None:
                                    print_node_box(current_node, "end", node_message_count)
                                current_node = node_name
                                node_message_count = 0
                                print_node_box(node_name, "start")
                            
                            if not node_output:
                                continue
                            
                            # Handle messages with DEBUG output
                            if "messages" in node_output:
                                for msg in node_output["messages"]:
                                    msg_id = getattr(msg, 'id', None)
                                    
                                    # Check if we've seen this message
                                    if msg_id and msg_id in seen_message_ids:
                                        # SKIPPED - already seen
                                        print_message_debug(msg, node_name, "SKIPPED", msg_id)
                                        continue
                                    
                                    if msg_id:
                                        seen_message_ids.add(msg_id)
                                    
                                    # NEW message
                                    if isinstance(msg, AIMessage):
                                        if msg.tool_calls:
                                            print_message_debug(msg, node_name, "NEW", msg_id)
                                        elif msg.content:
                                            print_message_debug(msg, node_name, "NEW", msg_id)
                                            node_message_count += 1
                                    elif isinstance(msg, ToolMessage):
                                        print_tool_result_debug(msg, node_name)
                
                # Start processing
                process_stream(input_state)
                
                if current_node:
                    print_node_box(current_node, "end", node_message_count)
                    
            except KeyboardInterrupt:
                print(f"\n{Colors.YELLOW}Interrupted. Type 'quit' to exit.{Colors.ENDC}")
                continue
            
            # Summary
            print(f"\n{Colors.BOLD}{Colors.CYAN}{'─' * 62}{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.CYAN}  TURN COMPLETE - {len(seen_message_ids)} unique messages processed{Colors.ENDC}")
            print(f"{Colors.BOLD}{Colors.CYAN}{'─' * 62}{Colors.ENDC}")
            
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
