"""Main graph definition for the customer support bot.

Follows LangGraph "Thinking in LangGraph" design principles:
- Each agent has its OWN tool node (no shared tools)
- Nodes use Command for explicit routing
- Minimal edges - routing logic lives inside nodes
- Retry policies on tool nodes for resilience

Architecture:
- Router: Classifies intent and routes to appropriate agent
- QA Agents: Each has its own LLM node + dedicated tool node
  - catalog_qa ↔ catalog_tools
  - account_qa ↔ account_tools  
  - lyrics_qa ↔ lyrics_tools
- Workflow Nodes: HITL flows with interrupts (email_change, purchase_flow)
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

from src.state import SupportState
from src.nodes.router import router_node
from src.nodes.catalog_qa import catalog_qa_node, CATALOG_TOOLS
from src.nodes.account_qa import account_qa_node, ACCOUNT_TOOLS
from src.nodes.lyrics_qa import lyrics_qa_node, LYRICS_TOOLS
from src.nodes.email_change import email_change_node
from src.nodes.purchase_flow import purchase_flow_node


# ============================================================================
# TOOL NODES: Each agent gets its own dedicated tool node
# This follows the "discrete steps" principle - clear boundaries per agent
# ============================================================================
catalog_tools_node = ToolNode(CATALOG_TOOLS)
account_tools_node = ToolNode(ACCOUNT_TOOLS)
lyrics_tools_node = ToolNode(LYRICS_TOOLS)


# ============================================================================
# ROUTING: No complex routing functions needed!
# Each node now uses Command to specify its destination explicitly.
# Tools always route back to their owning agent via simple edges.
# ============================================================================


def build_graph() -> StateGraph:
    """Build and return the support bot graph.
    
    Architecture follows "Thinking in LangGraph" principles:
    - Each agent has its own tool node
    - Nodes use Command for routing (no external routing functions)
    - Simple edges where possible
    - Retry policies on tool nodes
    """
    
    # Create the graph
    builder = StateGraph(SupportState)
    
    # =========================================================================
    # ADD NODES
    # =========================================================================
    
    # Router node - classifies intent
    builder.add_node("router", router_node)
    
    # QA agent nodes - each uses Command for routing
    builder.add_node("catalog_qa", catalog_qa_node)
    builder.add_node("account_qa", account_qa_node)
    builder.add_node("lyrics_qa", lyrics_qa_node)
    
    # Tool nodes - each agent has its own, with retry policies for resilience
    builder.add_node(
        "catalog_tools",
        catalog_tools_node,
        retry=RetryPolicy(max_attempts=3)
    )
    builder.add_node(
        "account_tools",
        account_tools_node,
        retry=RetryPolicy(max_attempts=3)
    )
    builder.add_node(
        "lyrics_tools",
        lyrics_tools_node,
        retry=RetryPolicy(max_attempts=3)
    )
    
    # Workflow nodes - use Command with interrupt() for HITL
    builder.add_node("email_change", email_change_node)
    builder.add_node("purchase_flow", purchase_flow_node)
    
    # =========================================================================
    # ADD EDGES
    # =========================================================================
    
    # Entry point: always start with router
    builder.add_edge(START, "router")
    
    # Router uses Command to route - no conditional edges needed!
    # (The router_node returns Command(goto="catalog_qa") etc.)
    
    # Tool nodes always return to their owning agent - simple edges!
    builder.add_edge("catalog_tools", "catalog_qa")
    builder.add_edge("account_tools", "account_qa")
    builder.add_edge("lyrics_tools", "lyrics_qa")
    
    # QA nodes use Command to route to:
    # - their tools node (if tool calls)
    # - router (if route change needed)
    # - END (if done)
    # No conditional edges needed!
    
    # Workflow nodes (email_change, purchase_flow) use Command
    # to specify their next destination - no explicit edges needed
    
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
