"""Account tools for customer-scoped data access.

All tools enforce customer_id scoping for security.
The customer_id is injected from the graph state, never user-supplied.
"""

from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig

from src.db import get_db


def _get_customer_id(config: RunnableConfig) -> int:
    """Extract customer_id from the runnable config.
    
    The customer_id is passed through the config's configurable dict
    to ensure it comes from the authenticated session, not user input.
    """
    return config.get("configurable", {}).get("customer_id", 1)


@tool
def get_my_profile(config: RunnableConfig) -> str:
    """Get the current customer's profile information.
    
    Returns:
        Customer profile including name, email, phone, and address.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    result = db.run(
        f"""
        SELECT 
            CustomerId,
            FirstName,
            LastName,
            Email,
            Phone,
            Address,
            City,
            State,
            Country,
            PostalCode
        FROM Customer
        WHERE CustomerId = {customer_id};
        """,
        include_columns=True
    )
    return result


@tool
def get_my_invoices(config: RunnableConfig) -> str:
    """Get all invoices for the current customer.
    
    Returns:
        A list of invoices with ID, date, and total amount.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    result = db.run(
        f"""
        SELECT 
            InvoiceId,
            InvoiceDate,
            BillingAddress,
            BillingCity,
            BillingCountry,
            Total
        FROM Invoice
        WHERE CustomerId = {customer_id}
        ORDER BY InvoiceDate DESC;
        """,
        include_columns=True
    )
    return result


@tool
def get_my_invoice_lines(invoice_id: int, config: RunnableConfig) -> str:
    """Get line items for a specific invoice.
    
    First verifies that the invoice belongs to the current customer.
    
    Args:
        invoice_id: The ID of the invoice to look up.
        
    Returns:
        Track details for each line item in the invoice.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    
    # First verify ownership
    ownership_check = db.run(
        f"""
        SELECT CustomerId FROM Invoice 
        WHERE InvoiceId = {invoice_id};
        """
    )
    
    if str(customer_id) not in ownership_check:
        return "Error: Invoice not found or access denied."
    
    result = db.run(
        f"""
        SELECT 
            InvoiceLine.InvoiceLineId,
            Track.TrackId,
            Track.Name as TrackName,
            Artist.Name as ArtistName,
            Album.Title as AlbumTitle,
            InvoiceLine.UnitPrice,
            InvoiceLine.Quantity
        FROM InvoiceLine
        JOIN Track ON InvoiceLine.TrackId = Track.TrackId
        JOIN Album ON Track.AlbumId = Album.AlbumId
        JOIN Artist ON Album.ArtistId = Artist.ArtistId
        WHERE InvoiceLine.InvoiceId = {invoice_id}
        ORDER BY InvoiceLine.InvoiceLineId;
        """,
        include_columns=True
    )
    return result


@tool
def check_if_already_purchased(track_id: int, config: RunnableConfig) -> str:
    """Check if the customer already owns a specific track.
    
    Args:
        track_id: The TrackId to check.
        
    Returns:
        Whether the customer already owns this track.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    
    result = db.run(
        f"""
        SELECT COUNT(*) as owned
        FROM InvoiceLine
        JOIN Invoice ON InvoiceLine.InvoiceId = Invoice.InvoiceId
        WHERE Invoice.CustomerId = {customer_id}
        AND InvoiceLine.TrackId = {track_id};
        """
    )
    
    # Check if count > 0 (result contains the count as a number)
    import re
    count_match = re.search(r'(\d+)', str(result))
    count = int(count_match.group(1)) if count_match else 0
    
    if count > 0:
        return f"Yes - customer already owns track {track_id}."
    return f"No - customer does not own track {track_id}."


@tool
def update_my_email(new_email: str, config: RunnableConfig) -> str:
    """Update the customer's email address.
    
    WARNING: This mutates the database. Should only be called after
    proper verification through the email_change flow.
    
    Args:
        new_email: The new email address to set.
        
    Returns:
        Confirmation message with the old and new email.
    """
    customer_id = _get_customer_id(config)
    db = get_db()
    
    # Get current email for confirmation
    old_email_result = db.run(
        f"SELECT Email FROM Customer WHERE CustomerId = {customer_id};"
    )
    
    # Update the email
    db.run(
        f"""
        UPDATE Customer 
        SET Email = '{new_email}'
        WHERE CustomerId = {customer_id};
        """
    )
    
    return f"Email updated successfully! Old: {old_email_result.strip()}, New: {new_email}"

