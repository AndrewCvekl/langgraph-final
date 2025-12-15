"""FastAPI server for the customer support bot.

Provides REST API endpoints for:
- Starting chat sessions
- Streaming responses via SSE
- Handling HITL interrupts
"""

import asyncio
import json
import uuid
import os
from typing import Any

from dotenv import load_dotenv

# Load environment variables FIRST (before any LangChain imports)
load_dotenv()

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.types import Command

from src.graph import compile_graph
from src.state import get_initial_state
from src.db import initialize_database, DEMO_CUSTOMER_ID


# Get the project root directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Initialize FastAPI app
app = FastAPI(
    title="Music Store Support Bot API",
    description="LangGraph-powered customer support chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for sessions and runs
# In production, use Redis or a proper database
sessions: dict[str, dict] = {}
runs: dict[str, dict] = {}

# Compile the graph once at startup
graph = None


@app.on_event("startup")
async def startup_event():
    """Initialize database and compile graph on startup."""
    global graph
    initialize_database()
    graph = compile_graph()


# Request/Response models
class ChatRequest(BaseModel):
    session_id: str | None = None
    message: str


class ChatResponse(BaseModel):
    run_id: str
    session_id: str


class InterruptRequest(BaseModel):
    resume_value: Any


class InterruptResponse(BaseModel):
    success: bool
    message: str


@app.post("/api/chat", response_model=ChatResponse)
async def start_chat(request: ChatRequest):
    """Start a new chat run.
    
    Creates or retrieves a session and starts processing the message.
    Returns a run_id that can be used to stream the response.
    """
    # Create or get session
    session_id = request.session_id or str(uuid.uuid4())
    if session_id not in sessions:
        sessions[session_id] = {
            "thread_id": str(uuid.uuid4()),
            "customer_id": DEMO_CUSTOMER_ID,
            "state": get_initial_state(customer_id=DEMO_CUSTOMER_ID),
        }
    
    session = sessions[session_id]
    
    # Create a new run
    run_id = str(uuid.uuid4())
    runs[run_id] = {
        "session_id": session_id,
        "status": "pending",
        "input": {"messages": [HumanMessage(content=request.message)]},
        "events": [],
        "interrupt": None,
        "seen_message_ids": set(),  # Track seen messages to avoid duplicates
    }
    
    return ChatResponse(run_id=run_id, session_id=session_id)


@app.get("/api/stream/{run_id}")
async def stream_response(run_id: str):
    """Stream the response for a run via SSE.
    
    Emits events for:
    - node_start: When a node begins execution
    - node_end: When a node finishes
    - tool_call: When a tool is invoked
    - tool_result: When a tool returns
    - token: Streaming tokens (if available)
    - interrupt: When HITL is required
    - done: When the run completes
    - error: If an error occurs
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = runs[run_id]
    session = sessions[run["session_id"]]
    
    async def event_generator():
        config = {
            "configurable": {
                "thread_id": session["thread_id"],
                "customer_id": session["customer_id"],
            }
        }
        
        try:
            run["status"] = "running"
            current_node = None
            # Use run's seen_message_ids to persist across stream calls (e.g., after interrupt resume)
            seen_message_ids = run.get("seen_message_ids", set())
            
            for event in graph.stream(
                run["input"],
                config=config,
                stream_mode="updates"
            ):
                # Check for interrupts
                if "__interrupt__" in event:
                    interrupts = event["__interrupt__"]
                    for interrupt_info in interrupts:
                        interrupt_value = interrupt_info.value if hasattr(interrupt_info, 'value') else interrupt_info
                        run["interrupt"] = interrupt_value
                        run["status"] = "interrupted"
                        yield f"data: {json.dumps({'type': 'interrupt', 'data': interrupt_value})}\n\n"
                    return
                
                # Process regular events
                for node_name, node_output in event.items():
                    if node_name.startswith("__"):
                        continue
                    
                    # Node start event
                    if node_name != current_node:
                        if current_node:
                            yield f"data: {json.dumps({'type': 'node_end', 'node': current_node})}\n\n"
                        current_node = node_name
                        yield f"data: {json.dumps({'type': 'node_start', 'node': node_name})}\n\n"
                    
                    if not node_output:
                        continue
                    
                    # Process messages (with deduplication)
                    if "messages" in node_output:
                        for msg in node_output["messages"]:
                            # Skip messages we've already sent
                            msg_id = getattr(msg, 'id', None)
                            if msg_id and msg_id in seen_message_ids:
                                continue
                            if msg_id:
                                seen_message_ids.add(msg_id)
                            
                            if isinstance(msg, AIMessage):
                                if msg.tool_calls:
                                    for tc in msg.tool_calls:
                                        yield f"data: {json.dumps({'type': 'tool_call', 'name': tc['name'], 'args': tc.get('args', {})})}\n\n"
                                elif msg.content:
                                    yield f"data: {json.dumps({'type': 'message', 'content': msg.content})}\n\n"
                            elif isinstance(msg, ToolMessage):
                                # Truncate long tool results
                                content = str(msg.content)[:500] + "..." if len(str(msg.content)) > 500 else str(msg.content)
                                yield f"data: {json.dumps({'type': 'tool_result', 'name': msg.name, 'content': content})}\n\n"
            
            # Final node end
            if current_node:
                yield f"data: {json.dumps({'type': 'node_end', 'node': current_node})}\n\n"
            
            run["status"] = "completed"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
            
        except Exception as e:
            run["status"] = "error"
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@app.post("/api/interrupt/{run_id}", response_model=InterruptResponse)
async def resume_interrupt(run_id: str, request: InterruptRequest):
    """Resume a run after an interrupt.
    
    Provides the user's response to the HITL interrupt and
    continues execution.
    """
    if run_id not in runs:
        raise HTTPException(status_code=404, detail="Run not found")
    
    run = runs[run_id]
    
    if run["status"] != "interrupted":
        raise HTTPException(status_code=400, detail="Run is not interrupted")
    
    # Update the run input to resume
    run["input"] = Command(resume=request.resume_value)
    run["status"] = "pending"
    run["interrupt"] = None
    
    return InterruptResponse(
        success=True,
        message="Interrupt resumed. Stream the run again to continue."
    )


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session information."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "thread_id": session["thread_id"],
        "customer_id": session["customer_id"],
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    """Delete a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del sessions[session_id]
    return {"success": True, "message": "Session deleted"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "graph_loaded": graph is not None}


# Serve static files
static_dir = os.path.join(PROJECT_ROOT, "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
async def serve_ui():
    """Serve the chat UI."""
    index_path = os.path.join(PROJECT_ROOT, "static", "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"message": "Welcome to Music Store Support Bot API. UI not found, use /docs for API."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

