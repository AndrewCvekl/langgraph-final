"""Main graph definition for the customer support bot.

Clean architecture following LangGraph best practices:
- Moderator → Supervisor → Domain Experts → Subgraphs
- catalog_qa is the "music brain" (handles browsing AND lyrics detection)
- Lyrics subgraph is self-contained with internal router for both flows
- Purchase subgraph uses HITL confirmation for actual purchase

Lyrics Flow (CONVERSATIONAL - all self-contained in lyrics subgraph):

MODE 1 - New lyrics query (lyrics_awaiting_response = False):
  1. User provides lyrics → catalog_qa routes to lyrics subgraph
  2. Lyrics subgraph: router → identify_song → check_catalog → get_youtube → present_options
  3. present_options asks "Would you like to buy?" and sets lyrics_awaiting_response=True
  4. Turn ENDS - user sees the question

MODE 2 - User response (lyrics_awaiting_response = True):
  5. User responds "yes" or "no" → supervisor routes to lyrics subgraph
  6. Lyrics subgraph: router → handle_response → sets lyrics_purchase_confirmed
  7. After subgraph ends, conditional edge routes to purchase if confirmed

Architecture:
                    ┌─────────────┐
                    │  Moderator  │ ← Input safety guard
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │  Supervisor │ ← Routes to domain experts
                    └──────┬──────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
        ┌──────────┐              ┌──────────┐
        │CatalogQA │              │AccountQA │
        └────┬─────┘              └────┬─────┘
             │                         │
    ┌────────┴────────┐           ┌────┴─────┐
    ▼                 ▼           ▼          ▼
 tools             lyrics      tools    email_chg
                   subgraph
                      │
        ┌─────────────┴─────────────┐
        │    lyrics subgraph        │
        │  ┌─────────────────────┐  │
        │  │       router        │  │ ← checks lyrics_awaiting_response
        │  └──────────┬──────────┘  │
        │       ┌─────┴─────┐       │
        │       ▼           ▼       │
        │  identify_song  handle    │
        │       │         _response │
        │       ▼           │       │
        │  check_catalog    │       │
        │       │           │       │
        │       ▼           │       │
        │  get_youtube      │       │
        │       │           │       │
        │       ▼           │       │
        │  present_options  │       │
        └───────────────────┴───────┘
                      │
                      ▼
            ┌─────────────────┐
            │ if purchase     │──► purchase_subgraph
            │ confirmed       │
            └─────────────────┘
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
    
    # Lyrics subgraph handles both identification AND response handling
    # Its internal router decides which flow based on lyrics_awaiting_response
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
