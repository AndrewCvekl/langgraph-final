"""Main graph definition for the customer support bot.

Assembles all nodes into a StateGraph with conditional routing.

Architecture:
- Router: Classifies intent and routes to appropriate node
- QA Nodes (catalog_qa, account_qa, lyrics_qa): LLM + Tools for agentic behavior
- Workflow Nodes (email_change, purchase_flow): HITL flows with interrupts
- ToolNode: Executes tool calls from QA nodes
"""

from typing import Literal

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

from src.state import SupportState
from src.nodes.guardrails import guardrails_node
from src.nodes.router import router_node
from src.nodes.catalog_qa import catalog_qa_node, CATALOG_TOOLS
from src.nodes.account_qa import account_qa_node, ACCOUNT_TOOLS
from src.nodes.lyrics_qa import lyrics_qa_node, LYRICS_TOOLS
from src.nodes.email_change import email_change_node
from src.nodes.purchase_flow import purchase_flow_node


# Create tool node with ALL tools from all QA nodes
ALL_TOOLS = CATALOG_TOOLS + ACCOUNT_TOOLS + LYRICS_TOOLS
tool_node = ToolNode(ALL_TOOLS)


def route_after_router(state: SupportState) -> Literal[
    "catalog_qa",
    "account_qa", 
    "email_change",
    "lyrics_qa",
    "purchase_flow",
    END
]:
    """Route based on the router's decision."""
    route = state.get("route")
    
    if route == "catalog_qa":
        return "catalog_qa"
    elif route == "account_qa":
        return "account_qa"
    elif route == "email_change":
        return "email_change"
    elif route == "lyrics_flow":
        # Map old route name to new node name
        return "lyrics_qa"
    elif route == "purchase_flow":
        return "purchase_flow"
    elif route == "final":
        return END
    else:
        # Default to catalog_qa for music-related questions
        return "catalog_qa"


def route_after_guardrails(state: SupportState) -> Literal["router", END]:
    """If guardrails blocked, end; otherwise continue to router."""
    if state.get("guardrail_blocked"):
        return END
    return "router"


def should_continue_qa(state: SupportState) -> Literal["tools", "router", END]:
    """Generic QA continuation logic - works for catalog, account, and lyrics QA.
    
    Decide whether to call tools, return to router, or end.
    """
    messages = state["messages"]
    if not messages:
        return "router"
    
    last_message = messages[-1]
    
    # If the LLM made tool calls, route to tools
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    
    # If route changed (e.g., to purchase_flow or email_change), go back to router
    route = state.get("route")
    if route in ("purchase_flow", "email_change"):
        return "router"
    
    # Otherwise, we're done with this turn
    return END


def route_after_tools(state: SupportState) -> Literal["catalog_qa", "account_qa", "lyrics_qa"]:
    """Route back to the appropriate QA node after tool execution."""
    messages = state["messages"]
    
    # Build tool name -> node mapping
    catalog_tool_names = {t.name for t in CATALOG_TOOLS}
    account_tool_names = {t.name for t in ACCOUNT_TOOLS}
    lyrics_tool_names = {t.name for t in LYRICS_TOOLS}
    
    # Find the last AI message before tools to determine which QA called them
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            continue
        if isinstance(msg, AIMessage) and msg.tool_calls:
            # Check which tools were called
            tool_names = {tc["name"] for tc in msg.tool_calls}
            
            # Check for lyrics tools first (most specific)
            if tool_names & lyrics_tool_names:
                return "lyrics_qa"
            elif tool_names & catalog_tool_names:
                return "catalog_qa"
            elif tool_names & account_tool_names:
                return "account_qa"
    
    # Default to catalog_qa
    return "catalog_qa"


def build_graph() -> StateGraph:
    """Build and return the support bot graph."""
    
    # Create the graph
    builder = StateGraph(SupportState)
    
    # Add nodes
    builder.add_node("guardrails", guardrails_node)
    builder.add_node("router", router_node)
    builder.add_node("catalog_qa", catalog_qa_node)
    builder.add_node("account_qa", account_qa_node)
    builder.add_node("lyrics_qa", lyrics_qa_node)
    builder.add_node("email_change", email_change_node)
    builder.add_node("purchase_flow", purchase_flow_node)
    builder.add_node("tools", tool_node)
    
    # Entry point: start with guardrails, then router
    builder.add_edge(START, "guardrails")

    builder.add_conditional_edges(
        "guardrails",
        route_after_guardrails,
        {
            "router": "router",
            END: END,
        },
    )
    
    # Router routes to specialized nodes
    builder.add_conditional_edges(
        "router",
        route_after_router,
        {
            "catalog_qa": "catalog_qa",
            "account_qa": "account_qa",
            "email_change": "email_change",
            "lyrics_qa": "lyrics_qa",
            "purchase_flow": "purchase_flow",
            END: END,
        }
    )
    
    # All QA nodes use the same continuation logic
    for qa_node in ["catalog_qa", "account_qa", "lyrics_qa"]:
        builder.add_conditional_edges(
            qa_node,
            should_continue_qa,
            {
                "tools": "tools",
                "router": "router",
                END: END,
            }
        )
    
    # Tools route back to the appropriate QA node
    builder.add_conditional_edges(
        "tools",
        route_after_tools,
        {
            "catalog_qa": "catalog_qa",
            "account_qa": "account_qa",
            "lyrics_qa": "lyrics_qa",
        }
    )
    
    # Workflow nodes (email_change, purchase_flow) use Command
    # to specify their next destination, so no explicit edges needed
    
    return builder


def compile_graph(checkpointer=None):
    """Compile the graph with optional checkpointer.
    
    Args:
        checkpointer: Optional checkpointer for state persistence.
                     If None, uses MemorySaver.
    
    Returns:
        Compiled graph ready for invocation.
    """
    builder = build_graph()
    
    if checkpointer is None:
        checkpointer = MemorySaver()
    
    return builder.compile(checkpointer=checkpointer)


# Default compiled graph for import
# NOTE: Don't provide a checkpointer when using langgraph dev / LangGraph Studio
# The platform handles persistence automatically
graph = build_graph().compile()


if __name__ == "__main__":
    # Print the graph structure for debugging
    from IPython.display import Image, display
    
    builder = build_graph()
    compiled = builder.compile()
    
    try:
        # Try to generate a visualization
        png_data = compiled.get_graph().draw_mermaid_png()
        with open("graph_visualization.png", "wb") as f:
            f.write(png_data)
        print("Graph visualization saved to graph_visualization.png")
    except Exception as e:
        print(f"Could not generate visualization: {e}")
        print("\nGraph structure:")
        print(compiled.get_graph().draw_ascii())
