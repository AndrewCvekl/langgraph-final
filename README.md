# LangGraph Customer Support Bot (Music Store Demo)

A customer support chatbot for a digital music store, built with LangGraph to demonstrate:
- **Routing** between specialized workflows
- **Tool invocation** for database queries
- **Human-in-the-Loop (HITL)** interrupts for sensitive actions
- **State management** across conversation turns
- **Observability** via LangSmith tracing

## Features

### Catalog Intelligence (Read-Only)
- Browse genres, artists, albums, and tracks
- Search for tracks by name or ID
- View track details including pricing

### Account Intelligence (Customer-Scoped)
- View customer profile
- View purchase history (invoices)
- View invoice line items (tracks purchased)

### Transactional Flows (HITL Required)
- **Email Update**: Change email with phone verification
- **Lyrics Search**: Identify songs from lyrics, listen on YouTube, purchase
- **Track Purchase**: Buy individual tracks with confirmation

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Environment Variables

```bash
export OPENAI_API_KEY="sk-your-key-here"

# Optional: Enable LangSmith tracing
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="lsv2_your_key_here"
export LANGCHAIN_PROJECT="langgraph-music-support"
```

### 3. Initialize the Database

The database will be automatically downloaded and initialized on first run, or you can initialize it manually:

```bash
python -m src.db.init_db
```

### 4. Run the CLI

```bash
python cli.py
```

### 5. Run in LangGraph Studio

```bash
langgraph dev
```

Then open the LangGraph Studio UI in your browser (typically http://localhost:8123).

### 6. Run the FastAPI Server (Optional)

```bash
uvicorn src.server:app --reload
```

Then access the API at http://localhost:8000 (see `/docs` for Swagger UI).

## Project Structure

```
langgraph-simple/
├── src/
│   ├── __init__.py
│   ├── graph.py             # Main StateGraph definition
│   ├── state.py             # SupportState TypedDict
│   ├── server.py            # FastAPI server (optional)
│   ├── nodes/               # Node implementations
│   │   ├── router.py        # Intent classification
│   │   ├── catalog_qa.py    # Music browsing
│   │   ├── account_qa.py    # Account info
│   │   ├── email_change.py  # Email update (HITL)
│   │   ├── lyrics_flow.py   # Lyrics search flow
│   │   └── purchase_flow.py # Purchase flow (HITL)
│   ├── tools/               # Tool definitions
│   │   ├── catalog.py       # Catalog queries
│   │   ├── account.py       # Account queries
│   │   ├── purchase.py      # Invoice creation
│   │   └── mocks.py         # Mock external APIs
│   └── db/
│       └── init_db.py       # Database initialization
├── cli.py                   # CLI runner (primary interface)
├── langgraph.json           # LangGraph Studio config
├── pyproject.toml           # Python project config
├── data/
│   └── chinook_demo.db      # SQLite database (auto-generated)
└── requirements.txt
```

## Demo Customer

For demo purposes, the bot assumes a logged-in customer:
- **Customer ID**: 1
- **Name**: Luís Gonçalves
- **Email**: luisg@embraer.com.br
- **Phone**: +55 (12) 3923-5555

## Architecture

```
┌─────────────┐
│   Router    │ ← Classifies user intent
└──────┬──────┘
       │
       ├──────────────┬──────────────┬──────────────┬──────────────┐
       ▼              ▼              ▼              ▼              ▼
┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐
│ Catalog   │  │ Account   │  │  Email    │  │  Lyrics   │  │ Purchase  │
│    QA     │  │    QA     │  │  Change   │  │   Flow    │  │   Flow    │
└─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘  └─────┬─────┘
      │              │              │              │              │
      └──────────────┴──────┬───────┴──────────────┴──────────────┘
                            ▼
                      ┌───────────┐
                      │ ToolNode  │ ← Executes tool calls
                      └───────────┘
```

## Human-in-the-Loop

The following actions require explicit user approval:
1. **Send Verification Code**: Before sending SMS verification
2. **Enter Verification Code**: User must input the code
3. **Confirm Purchase**: Before creating an invoice

## License

MIT

