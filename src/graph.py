"""Main graph definition for the customer support bot.

Clean architecture following LangGraph best practices:
- Moderator → Supervisor → Domain Experts → Subgraphs
- Supervisor is STATELESS - only routes based on intent (catalog vs account)
- catalog_qa is the "music brain" - detects lyrics intent and routes to subgraph
- All subgraphs use HITL (interrupt) for critical decisions
- Consistent Input/Output schemas prevent message accumulation

Key Design Principle:
  Supervisor = "What domain?" (stateless intent routing)
  Domain Expert = "What task in my domain?" (coordinates workflows)
  Subgraph = "Self-contained workflow with HITL" (clean state isolation)

Lyrics Flow (with HITL):
  1. User provides lyrics → supervisor → catalog_qa (detects lyrics) → lyrics subgraph
  2. Lyrics subgraph: identify_song → check_catalog → get_youtube → present_options
  3. present_options uses interrupt() to ask "Would you like to buy?"
  4. User responds via inline buttons → resume interrupt
  5. If confirmed, conditional edge routes to purchase subgraph

Architecture:
                    ┌─────────────┐
                    │  Moderator  │ ← Input safety guard
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Supervisor │ ← STATELESS: routes to domains only
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │CatalogQA │              │AccountQA │
        │ (music   │              │ (account │
        │  brain)  │              │  expert) │
        └────┬─────┘              └────┬─────┘
             │                         │
    ┌────────┼────────┐           ┌────┼─────┐
    ▼        ▼        ▼           ▼    ▼     ▼
 tools    lyrics   purchase    tools email_chg
          subgraph subgraph
"""

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import RetryPolicy

from src.state import SupportState
from src.nodes import (
    supervisor_node,
    moderator_node,
    catalog_qa_node,
    account_qa_node,
    CATALOG_TOOLS,
    ACCOUNT_TOOLS,
)
from src.subgraphs import purchase_subgraph, email_change_subgraph, lyrics_subgraph


# ============================================================================
# TOOL NODES: Each agent gets its own dedicated tool node
# ============================================================================
catalog_tools_node = ToolNode(CATALOG_TOOLS)
account_tools_node = ToolNode(ACCOUNT_TOOLS)


def route_after_lyrics(state: SupportState) -> str:
    """Route after lyrics subgraph completes.
    
    If lyrics_purchase_confirmed is True (user said yes to purchase),
    route to purchase subgraph. Otherwise, end.
    """
    if state.get("lyrics_purchase_confirmed") and state.get("pending_track_id"):
        return "purchase"
    return END


def build_graph() -> StateGraph:
    """Build and return the support bot graph.
    
    Architecture:
    - Moderator checks input safety
    - Supervisor routes to domain experts (or lyrics subgraph if awaiting response)
    - catalog_qa detects lyrics intent and routes to lyrics subgraph
    - Lyrics subgraph is self-contained: handles identification AND response
    - After lyrics, conditional edge routes to purchase if user confirmed
    """
    builder = StateGraph(SupportState)
    
    # =========================================================================
    # ADD NODES
    # =========================================================================
    
    # Entry nodes
    builder.add_node("moderator", moderator_node)
    builder.add_node("supervisor", supervisor_node)
    
    # Domain expert nodes
    builder.add_node("catalog_qa", catalog_qa_node)
    builder.add_node("account_qa", account_qa_node)
    
    # Tool nodes with retry policies
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
    
    # Subgraphs for workflows
    builder.add_node("lyrics", lyrics_subgraph)
    builder.add_node("purchase", purchase_subgraph)
    builder.add_node("email_change", email_change_subgraph)
    
    # =========================================================================
    # ADD EDGES
    # =========================================================================
    
    # Entry: start with moderator
    builder.add_edge(START, "moderator")
    
    # Moderator uses Command to route to supervisor or end
    # (No explicit edge needed - Command handles routing)
    
    # Supervisor uses Command to route to domain experts
    # (No explicit edge needed - Command handles routing)
    
    # Tool nodes return to their owning agent
    builder.add_edge("catalog_tools", "catalog_qa")
    builder.add_edge("account_tools", "account_qa")
    
    # Lyrics subgraph handles identification with HITL for purchase decision
    # After lyrics completes, check if user confirmed purchase
    builder.add_conditional_edges("lyrics", route_after_lyrics)
    
    # Purchase and email_change subgraphs go to END when complete
    builder.add_edge("purchase", END)
    builder.add_edge("email_change", END)
    
    # Domain experts use Command to route to:
    # - Their tools (if tool calls)
    # - Subgraphs (if workflow needed)
    # - END (if response complete)
    
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
