"""Purchase tools for creating invoices.

These tools mutate the database and should only be called
after HITL confirmation in the purchase_flow node.
"""

import re
from datetime import datetime

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.db import get_db


def _get_customer_id(config: RunnableConfig) -> int:
    """Extract customer_id from the runnable config."""
    return config.get("configurable", {}).get("customer_id", 1)


@tool
def create_invoice_for_track(track_id: int, config: RunnableConfig) -> str:
    """Create a new invoice with a single track purchase.
    
    Uses the customer's existing billing information.
    Creates both an Invoice and InvoiceLine record.
    
    Args:
        track_id: The TrackId of the track to purchase.
        
    Returns:
        Confirmation with the new invoice ID and total.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    
    # Get track info for the invoice line
    track_info = db.run(
        f"""
        SELECT TrackId, Name, UnitPrice 
        FROM Track 
        WHERE TrackId = {track_id};
        """,
        include_columns=True
    )
    
    if not track_info or "TrackId" not in track_info:
        return f"Error: Track with ID {track_id} not found."
    
    # Get customer billing info
    customer_info = db.run(
        f"""
        SELECT Address, City, State, Country, PostalCode
        FROM Customer
        WHERE CustomerId = {customer_id};
        """
    )
    
    # Parse customer info (simple approach for demo)
    # In production, you'd use proper result parsing
    
    # Get the next invoice ID using COALESCE to handle empty table
    max_invoice_result = db.run("SELECT COALESCE(MAX(InvoiceId), 0) + 1 AS NextId FROM Invoice;")
    # Parse the result - db.run returns formatted string, extract the number
    next_invoice_match = re.search(r'(\d+)', str(max_invoice_result))
    next_invoice_id = int(next_invoice_match.group(1)) if next_invoice_match else 1
    
    # Get track price
    price_result = db.run(f"SELECT UnitPrice FROM Track WHERE TrackId = {track_id};")
    price_match = re.search(r'([\d.]+)', str(price_result))
    unit_price = float(price_match.group(1)) if price_match else 0.99
    
    # Get customer billing details
    billing = db.run(
        f"""
        SELECT Address, City, State, Country, PostalCode
        FROM Customer WHERE CustomerId = {customer_id};
        """
    )
    
    # Create the invoice
    invoice_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    db.run(
        f"""
        INSERT INTO Invoice (
            InvoiceId, CustomerId, InvoiceDate, 
            BillingAddress, BillingCity, BillingState, 
            BillingCountry, BillingPostalCode, Total
        )
        SELECT 
            {next_invoice_id},
            {customer_id},
            '{invoice_date}',
            Address,
            City,
            State,
            Country,
            PostalCode,
            {unit_price}
        FROM Customer
        WHERE CustomerId = {customer_id};
        """
    )
    
    # Get next invoice line ID using COALESCE
    max_line_result = db.run("SELECT COALESCE(MAX(InvoiceLineId), 0) + 1 AS NextId FROM InvoiceLine;")
    line_match = re.search(r'(\d+)', str(max_line_result))
    next_line_id = int(line_match.group(1)) if line_match else 1
    
    # Create the invoice line
    db.run(
        f"""
        INSERT INTO InvoiceLine (InvoiceLineId, InvoiceId, TrackId, UnitPrice, Quantity)
        VALUES ({next_line_id}, {next_invoice_id}, {track_id}, {unit_price}, 1);
        """
    )
    
    # Get track name for confirmation
    track_name_result = db.run(f"SELECT Name FROM Track WHERE TrackId = {track_id};")
    # Parse track name from SQL result (handles tuple format)
    track_name_match = re.search(r"'([^']+)'", str(track_name_result))
    track_name = track_name_match.group(1) if track_name_match else f"Track {track_id}"
    
    return f"""
Purchase complete!
- Invoice ID: {next_invoice_id}
- Track: {track_name}
- Amount: ${unit_price:.2f}
- Date: {invoice_date}

Thank you for your purchase!
"""

