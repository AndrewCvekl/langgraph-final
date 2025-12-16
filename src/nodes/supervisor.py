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
    
    route: Literal["catalog", "account", "chitchat"] = Field(
        description="""The domain to route to:
        - catalog: Questions about music (browsing, searching, purchasing, lyrics identification)
        - account: Questions about profile, invoices, purchase history, email changes
        - chitchat: Pleasantries, greetings, thanks, goodbyes, or non-actionable messages
        """
    )
    
    reasoning: str = Field(
        description="Brief explanation of why this route was chosen"
    )
    
    chitchat_response: str = Field(
        default="",
        description="If route is chitchat, provide a brief friendly response appropriate to the user's message"
    )


SUPERVISOR_SYSTEM_PROMPT = """You are a customer service supervisor for a digital music store.

Your job is to route customers to the right specialist OR respond directly to simple pleasantries.

## Routes:

1. **catalog**: For anything about MUSIC:
   - Browsing genres, finding artists, searching for tracks
   - Purchasing songs, music recommendations
   - LYRICS IDENTIFICATION - when user provides song lyrics and wants to identify the song
   - Example: "What song goes 'Is this the real life, is this just fantasy'"

2. **account**: For anything about their ACCOUNT:
   - Profile info, invoices, purchase history
   - Changing their email address, billing questions

3. **chitchat**: For pleasantries and non-actionable messages:
   - Greetings: "hi", "hello", "hey" → warm welcome, mention you can help with music or account questions
   - Thanks: "thanks", "thank you", "no problem", "ok", "cool", "great" → polite acknowledgment
   - Goodbyes: "bye", "goodbye", "see ya" → friendly farewell
   - Small talk or unclear intent → friendly redirect to what you can help with

## Guidelines:

- Route based on the user's latest message
- If the message has NO actionable intent (just pleasantries), use chitchat and provide a response
- Any music-related query including lyrics → catalog
- "Change my email" → account (email changes are account management)

When routing to chitchat, ALWAYS provide a brief, contextual response in chitchat_response.
Be decisive. Pick the single best route."""


def supervisor_node(
    state: SupportState
) -> Command[Literal["catalog_qa", "account_qa", "__end__"]]:
    """Route to the appropriate domain expert or handle chitchat directly.
    
    Pure intent-based routing - no state checks here.
    catalog_qa handles all music-related queries including lyrics.
    account_qa handles customer account info.
    chitchat is handled directly here with a friendly response.
    
    Workflow state (like lyrics_awaiting_response) is handled by domain experts,
    not the supervisor. This keeps routing logic clean and stateless.
    """
    model = ChatOpenAI(model="gpt-4o", temperature=0)
    structured_model = model.with_structured_output(RouteDecision)
    
    messages = [SystemMessage(content=SUPERVISOR_SYSTEM_PROMPT)] + state["messages"]
    
    decision: RouteDecision = structured_model.invoke(messages)
    
    # Handle chitchat directly - respond and end
    if decision.route == "chitchat":
        response = decision.chitchat_response or "Happy to help! Let me know if you have any questions about our music catalog or your account."
        return Command(
            update={"messages": [AIMessage(content=response)]},
            goto="__end__"
        )
    
    # Route to domain experts
    route_to_node = {
        "catalog": "catalog_qa",
        "account": "account_qa",
    }
    
    return Command(goto=route_to_node[decision.route])
