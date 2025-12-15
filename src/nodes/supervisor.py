"""Supervisor node for routing between domains.

Simple, focused routing using structured output.
Routes to domain experts - catalog handles all music (including lyrics).

Key principles:
- Routes BETWEEN domains, not within them
- Uses structured output for reliable routing
- catalog_qa handles music browsing AND lyrics identification
- account_qa handles customer account info
"""

from typing import Literal

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from pydantic import BaseModel, Field

from src.state import SupportState


class RouteDecision(BaseModel):
    """Structured output for routing decisions."""
    
    route: Literal["catalog", "account", "end"] = Field(
        description="""The domain to route to:
        - catalog: Questions about music (browsing, searching, purchasing, lyrics identification)
        - account: Questions about profile, invoices, purchase history, email changes
        - end: Conversation is complete, user said goodbye
        """
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen"
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a customer service supervisor for a digital music store.

Your job is to route customers to the right specialist based on what they're asking about.

## Routes:

1. **catalog**: For anything about MUSIC:
   - Browsing genres, finding artists, searching for tracks
   - Purchasing songs, music recommendations
   - LYRICS IDENTIFICATION - when user provides song lyrics and wants to identify the song
   - Example: "What song goes 'Is this the real life, is this just fantasy'"

2. **account**: For anything about their ACCOUNT:
   - Profile info, invoices, purchase history
   - Changing their email address, billing questions

3. **end**: When the conversation is DONE - user says goodbye, thanks and leaves, etc.

## Guidelines:

- Route based on the user's latest message
- Greetings like "hi" or "hello" → catalog (default starting point for browsing)
- Any music-related query including lyrics → catalog
- "Change my email" → account (email changes are account management)

Be decisive. Pick the single best route."""


def supervisor_node(
    state: SupportState
) -> Command[Literal["catalog_qa", "account_qa", "__end__"]]:
    """Route to the appropriate domain expert.
    
    Pure intent-based routing - no state checks here.
    catalog_qa handles all music-related queries including lyrics.
    account_qa handles customer account info.
    
    Workflow state (like lyrics_awaiting_response) is handled by domain experts,
    not the supervisor. This keeps routing logic clean and stateless.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(RouteDecision)
    
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
    
    decision: RouteDecision = structured_model.invoke(messages)
    
    # Map route names to node names
    route_to_node = {
        "catalog": "catalog_qa",
        "account": "account_qa", 
        "end": "__end__",
    }
    
    goto_node = route_to_node[decision.route]
    
    return Command(goto=goto_node)
