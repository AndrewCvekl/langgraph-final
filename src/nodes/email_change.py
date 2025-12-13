"""Email change node with HITL verification.

Implements a clean multi-step flow:
1. HITL: Confirm sending verification code to phone
2. Send code via Twilio (real SMS) or mock, save to state, re-enter node
3. HITL: User enters the code (with up to 3 attempts)
4. Mark verified, re-enter node
5. HITL: User enters new email address
6. Update the database

CRITICAL DESIGN PRINCIPLE:
- Only ONE interrupt per node execution
- After each interrupt, return with state update and goto="email_change"
- State determines which step we're on when re-entering
- All terminal paths go to "__end__" for clean state transitions
"""

from typing import Literal
import re
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.types import interrupt, Command

from src.state import SupportState
from src.tools.account import update_my_email
from src.tools.services import get_twilio_service
from src.db import get_db

logger = logging.getLogger(__name__)


def _get_customer_phone(customer_id: int) -> str:
    """Get the customer's phone number from the database."""
    db = get_db()
    result = db.run(
        f"SELECT Phone FROM Customer WHERE CustomerId = {customer_id};"
    )
    # Parse the result (it comes as a string like "('phone',)")
    phone = result.strip().replace("(", "").replace(")", "").replace("'", "").replace(",", "")
    return phone if phone else ""


def _mask_phone(phone: str) -> str:
    """Mask a phone number, showing only the last 4 digits."""
    if not phone:
        return "No phone on file"
    if len(phone) >= 4:
        return "*" * (len(phone) - 4) + phone[-4:]
    return phone


def _clear_email_state() -> dict:
    """Return state update that clears all email change state."""
    return {
        "pending_email": None,
        "verification_code": None,
        "verification_id": None,
        "verification_attempts": 0,
        "verified": False,
        "masked_phone": "",
        "phone": None,
    }


def email_change_node(
    state: SupportState
) -> Command[Literal["__end__", "email_change"]]:
    """Handle email change with verification.
    
    State-driven flow with ONE interrupt per execution:
    - Step 1 (no verification_id, not verified): Ask to send code
    - Step 2 (verification_id exists, not verified): Ask for code entry
    - Step 3 (verified, no email): Ask for new email
    
    Each step ends with either goto="email_change" (continue) or goto="__end__" (done).
    """
    customer_id = state.get("customer_id", 1)
    verification_id = state.get("verification_id")
    verification_code = state.get("verification_code")  # For mock mode
    verification_attempts = state.get("verification_attempts", 0)
    verified = state.get("verified", False)
    pending_email = state.get("pending_email")
    masked_phone = state.get("masked_phone", "")
    phone = state.get("phone", "")
    
    # Get Twilio service
    twilio = get_twilio_service()
    logger.info(f"[EmailChange] Twilio is_live: {twilio.is_live}")
    
    # Get phone number if we don't have it yet
    if not phone:
        phone = _get_customer_phone(customer_id)
        masked_phone = _mask_phone(phone)
        logger.info(f"[EmailChange] Got phone from DB: {masked_phone}")
    
    # =========================================================================
    # STEP 1: Ask to send verification code
    # Condition: No verification in progress AND not verified
    # =========================================================================
    if not verification_id and not verification_code and not verified:
        confirm = interrupt({
            "type": "confirm",
            "title": "Verify Your Identity",
            "message": f"To change your email, we need to verify your identity.\n\nWe'll send a 6-digit verification code to your phone ({masked_phone}).\n\nWould you like to continue?",
            "options": ["yes", "no"]
        })
        
        if confirm.lower() != "yes":
            # User cancelled - clear state and end cleanly
            return Command(
                update={
                    "messages": [AIMessage(content="No problem! Your email remains unchanged. Is there anything else I can help you with?")],
                    **_clear_email_state()
                },
                goto="__end__"
            )
        
        # User confirmed - send code via Twilio
        logger.info(f"[EmailChange] Sending verification code to {masked_phone}...")
        new_verification_id = twilio.send_code(phone)
        logger.info(f"[EmailChange] Got verification_id: {new_verification_id[:20]}...")
        
        # Get the code for mock mode display
        mock_code = twilio.get_pending_code(new_verification_id)
        
        if twilio.is_live:
            msg = f"📱 Verification code sent to {masked_phone} via SMS!"
            logger.info("[EmailChange] Real SMS sent via Twilio")
        else:
            msg = f"📱 Sending verification code to {masked_phone}..."
            logger.info(f"[EmailChange] Mock mode - code is {mock_code}")
        
        return Command(
            update={
                "messages": [AIMessage(content=msg)],
                "verification_id": new_verification_id,
                "verification_code": mock_code,  # Only set in mock mode
                "verification_attempts": 0,
                "masked_phone": masked_phone,
                "phone": phone,
            },
            goto="email_change"  # Re-enter for code entry
        )
    
    # =========================================================================
    # STEP 2: Ask for verification code
    # Condition: Verification in progress AND not verified yet
    # =========================================================================
    if (verification_id or verification_code) and not verified:
        # Check if too many attempts already
        if verification_attempts >= 3:
            return Command(
                update={
                    "messages": [AIMessage(content="Too many incorrect attempts. For security, please try again later or contact support.")],
                    **_clear_email_state()
                },
                goto="__end__"
            )
        
        # Build the prompt - show code only in mock mode
        if twilio.is_live:
            prompt_msg = f"Please enter the 6-digit code sent to {masked_phone} (Attempt {verification_attempts + 1}/3)."
        else:
            prompt_msg = f"Please enter the 6-digit code sent to {masked_phone} (Attempt {verification_attempts + 1}/3).\n\n[Demo mode: The code is {verification_code}]"
        
        # Ask for code
        entered_code = interrupt({
            "type": "input",
            "title": "Enter Verification Code",
            "message": prompt_msg,
            "field": "code"
        })
        
        # Verify the code
        logger.info(f"[EmailChange] Checking code: {entered_code}")
        
        if verification_id:
            # Use Twilio to verify (works for both real and mock)
            is_valid = twilio.check_code(verification_id, entered_code.strip())
            logger.info(f"[EmailChange] Twilio check_code result: {is_valid}")
        else:
            # Fallback for legacy mock mode
            is_valid = entered_code.strip() == verification_code
            logger.info(f"[EmailChange] Legacy mock check result: {is_valid}")
        
        if is_valid:
            # Success! Mark as verified and re-enter for email collection
            logger.info("[EmailChange] ✅ Code verified successfully!")
            return Command(
                update={
                    "messages": [AIMessage(content="✅ Code verified successfully!")],
                    "verified": True,
                    "verification_code": None,
                    "verification_id": None,
                    "verification_attempts": 0,
                    "masked_phone": masked_phone,
                    "phone": phone,
                },
                goto="email_change"  # Re-enter for email entry
            )
        else:
            # Wrong code
            new_attempts = verification_attempts + 1
            logger.info(f"[EmailChange] ❌ Invalid code. Attempts: {new_attempts}/3")
            
            if new_attempts >= 3:
                return Command(
                    update={
                        "messages": [AIMessage(content="Too many incorrect attempts. For security, please try again later or contact support.")],
                        **_clear_email_state()
                    },
                    goto="__end__"
                )
            
            # Allow retry
            return Command(
                update={
                    "messages": [AIMessage(content=f"❌ Incorrect code. You have {3 - new_attempts} attempt(s) remaining.")],
                    "verification_attempts": new_attempts,
                    "verification_id": verification_id,
                    "verification_code": verification_code,
                    "masked_phone": masked_phone,
                    "phone": phone,
                },
                goto="email_change"  # Re-enter for another attempt
            )
    
    # =========================================================================
    # STEP 3: Collect new email
    # Condition: Verified AND no pending email
    # =========================================================================
    if verified and not pending_email:
        new_email = interrupt({
            "type": "input",
            "title": "Enter New Email",
            "message": "Your identity has been verified.\n\nPlease enter your new email address:",
            "field": "email"
        })
        
        # Basic email validation
        if not re.match(r'^[\w.+-]+@[\w-]+\.[\w.-]+$', new_email.strip()):
            return Command(
                update={
                    "messages": [AIMessage(content=f"'{new_email}' doesn't look like a valid email address. Please try the email change process again.")],
                    **_clear_email_state()
                },
                goto="__end__"
            )
        
        # Update the email in the database
        config = {"configurable": {"customer_id": customer_id}}
        result = update_my_email.invoke({"new_email": new_email.strip()}, config=config)
        
        return Command(
            update={
                "messages": [AIMessage(content=f"✅ {result}\n\nIs there anything else I can help you with?")],
                **_clear_email_state()
            },
            goto="__end__"
        )
    
    # =========================================================================
    # FALLBACK: Unexpected state - reset and end
    # =========================================================================
    return Command(
        update={
            "messages": [AIMessage(content="I'd be happy to help you change your email. Let's start fresh - just let me know when you're ready to update your email address.")],
            **_clear_email_state()
        },
        goto="__end__"
    )
