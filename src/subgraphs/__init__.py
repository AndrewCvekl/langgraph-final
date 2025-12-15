"""Subgraphs for isolated workflows.

Each subgraph handles a specific workflow with its own internal state.
"""

from src.subgraphs.purchase import purchase_subgraph
from src.subgraphs.email_change import email_change_subgraph
from src.subgraphs.lyrics import lyrics_subgraph

__all__ = ["purchase_subgraph", "email_change_subgraph", "lyrics_subgraph"]
