"""Node definitions for the customer support bot.

Exports:
- supervisor_node: Routes between domains
- moderator_node: Input safety guard
- catalog_qa_node: Music browsing, lyrics detection, and purchases
- account_qa_node: Account information
- Tool lists for each agent
"""

from src.nodes.supervisor import supervisor_node
from src.nodes.moderator import moderator_node
from src.nodes.catalog_qa import catalog_qa_node, CATALOG_TOOLS
from src.nodes.account_qa import account_qa_node, ACCOUNT_TOOLS

__all__ = [
    "supervisor_node",
    "moderator_node",
    "catalog_qa_node",
    "account_qa_node",
    "CATALOG_TOOLS",
    "ACCOUNT_TOOLS",
]
