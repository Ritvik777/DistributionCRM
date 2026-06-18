# TradeFlow Agent — From part photo to pipeline

This repository is **TradeFlow Agent** built with LangGraph — a multi-agent assistant that takes you from **part photo → catalog match → quote → outreach → CRM**, in one conversation.

Each user request is routed to the correct specialist branch, then processed through branch-specific nodes that gather context, apply business gates (pricing email verification or send intent), and return a final response with full trace visibility.

- **Supervisor Routing Agent** decides between **GTM**, **Outreach**, and **CRM**
- **GTM** branch answers product and pricing questions from your knowledge base; supports **component photo verification** via hybrid CLIP + vision matching
- **Outreach** branch creates content, finds new leads, and can send emails (with UI confirmation before delivery)
- **CRM** branch runs Salesforce operations (query/list leads, aggregates, record DML, describe objects)
- Full observability with Galileo tracing/session support (optional)

Project docs (GitHub Pages): [TradeFlow Agent walkthrough](https://ritvik777.github.io/TradeflowCRM/)

---

## How VLM agents help product distribution

Product distribution slows down at the points where a human has to look at something, recognize it, and translate it into the next business action. TradeFlow Agent uses **Vision-Language Model (VLM) agents** to remove those bottlenecks across the distribution pipeline:

- **Part identification from a photo** — A field rep, customer, or warehouse worker snaps a photo of an unlabeled component. The GTM VLM agent captions the image (Claude vision), retrieves visually similar candidates from the CLIP image catalog, and re-ranks them to return the matching SKU with a confidence score — no part number required. This collapses the slowest step in distribution (figuring out *what* the part is) from minutes of manual catalog search to one message.
- **Catalog match → quote** — Once the SKU is identified, the same conversation pulls pricing and product context from the knowledge base, so an identified part flows straight into an accurate quote instead of a separate lookup.
- **Vision-grounded outreach** — The outreach agent can inline the matched catalog image directly into B2B emails, so prospects see the exact part being offered. Visual confirmation reduces back-and-forth and wrong-part orders.
- **Closing the loop into CRM** — Identified parts and completed sends are logged as Salesforce records/tasks, so every photo-driven interaction becomes structured pipeline data (part enquiries, leads, follow-ups) that distribution teams can query and forecast against.
- **Reliability at scale** — Because matching combines CLIP retrieval with VLM re-ranking (rather than a single black-box model), results stay explainable — confidence %, caption, and candidate pool are all surfaced. Galileo tracing records every vision and routing decision, which matters when distribution decisions depend on the agent being right.

**Net effect:** a photo taken anywhere in the distribution chain — field, counter, or warehouse — becomes an identified part, a priced quote, an outbound email, and a CRM record, in one conversation.

---

## Architecture (Current)

### Graph flow

```text
START -> classify
          |- gtm      -> gtm_retrieve -> pricing_gate --not_pricing--> gtm_generate -> END
          |                                         \--pricing-------> collect_email --valid--> gtm_generate -> END
          |                                                                            \--no_email----------> END
          |
          |- outreach -> outreach_research -> outreach_generate -> send_gate --review--> END
          |                                                          \--send------------> outreach_send -> END
          |
          \- crm      -> crm_research -> crm_generate -> END
```

> **Send gate:** `send_gate` detects send intent, but actual delivery requires UI confirmation (`send_confirmed=True` via `confirm_send()`). Review-only drafts end at `END`; confirmed sends run `outreach_send`.

### Agents

- **Supervisor Routing Agent** (`agents/router_agent/nodes.py`)
  - Keyword fast-path for outreach, CRM, and image-attached queries
  - LLM structured routing (`RouteDecision`) as fallback
  - Routes to `gtm`, `outreach`, or `crm`

- **GTM Agent** (`agents/gtm_agent/nodes.py`)
  - Retrieves context from internal KB + web, or runs **component image match** when a photo is attached
  - Pricing gate requires verified email before full pricing output
  - Generates final product/pricing/catalog response

- **Outreach Agent** (`agents/outreach_agent/nodes.py`)
  - Researches context (Apollo for net-new leads; Salesforce de-dup when configured)
  - Generates formal B2B marketing content (email/post)
  - Send gate determines review-only vs send intent; **Brevo delivery** only after UI confirm
  - After a successful send, logs a completed **Task** in Salesforce (creates Lead if needed)
  - Can attach or inline catalog reference photos in outbound emails

- **CRM Agent** (`agents/crm_agent/nodes.py`)
  - Owns all Salesforce/CRM operations via TypeScript MCP server (or Python REST fallback)
  - Fetch/list/search records, SOQL + aggregate queries, record create/update/delete/upsert
  - Describe/search objects; part-enquiry search via logged Task records
  - Fast-path Markdown table for simple "latest leads" fetches; LLM tool loop for everything else

### Shared state

Defined in `agents/state.py`:

- `question` — current user message
- `chat_history` — recent turns for multi-turn context
- `agent_type` — `gtm` / `outreach` / `crm`
- `context` — retrieved KB/web/CRM/component data
- `answer` — final response text
- `kb_sources` — citation metadata for the UI
- `query_image_b64` — attached component photo (base64)
- `component_matches` — hybrid vision match results (SKU, confidence %, caption)
- `is_pricing` / `user_email` — pricing gate state
- `send_intent` / `send_requested` / `send_confirmed` — send detection vs UI-confirmed send
- `steps` — merged pipeline trace (reducer appends across nodes)

---

## Key files

```text
app.py                          # Streamlit entrypoint
ui/ui.py                        # Sidebar, chat, trace rendering, send confirmation
ui/component_vision.py          # Component match results panel
agents/__init__.py              # ask(), confirm_send(), Galileo tracing
agents/graph.py                 # LangGraph node wiring
agents/intent.py                # Regex intent classifiers (CRM/outreach fast-path)
agents/router_agent/nodes.py    # classify + route (gtm / outreach / crm)
agents/gtm_agent/nodes.py       # GTM branch nodes
agents/outreach_agent/nodes.py  # Outreach branch nodes
agents/crm_agent/nodes.py       # CRM (Salesforce) branch nodes
agents/tools/                   # KB/web/Apollo/Brevo/Salesforce tools + runner loop
services/agent_service.py       # UI → agents adapter
services/conversation_service.py # Session state helpers (history, pending drafts)
services/component_match_service.py # Shared component match formatting
services/catalog_image_host.py  # Public URLs for inline email images
services/salesforce_mcp.py      # Python MCP client → TypeScript MCP server (stdio)
services/salesforce_client.py   # CRM ops (MCP by default, Python REST fallback)
services/salesforce_repository.py # Lead fetch/format fast-paths
services/vector_db_service.py   # UI → vector_db adapter
vector_db/database.py           # Qdrant hybrid search (dense + BM25 via Cloud Inference)
vector_db/chunker.py            # Text/PDF/Excel/CSV chunking
vector_db/component_store.py    # CLIP image catalog + hybrid component matching
vector_db/vision.py             # Claude vision captioning and re-ranking
observability/galileo.py        # Tracing/session setup
evals/run_galileo_evals.py      # Baseline evaluation suite
```

### File-by-file map (detailed)

| File | What it does |
|---|---|
| `app.py` | Main Streamlit entrypoint: session init, chat loop, composer with image attach. |
| `config.py` | Global config/env loading (Qdrant, Anthropic, Brevo, Apollo, Salesforce, vision). |
| `llm.py` | Anthropic model factory (`get_llm`, vision caption/rerank variants). |
| `ui/ui.py` | UI logic: styling, sidebar uploads, chat rendering, trace display, send confirmation. |
| `ui/component_vision.py` | Renders component catalog match panel (confidence, SKU, thumbnail). |
| `services/agent_service.py` | Service adapter for `ask_agent()`, `confirm_send_email()`, graph image/ASCII. |
| `services/vector_db_service.py` | Service adapter for docs/PDFs/Excel/CSV and component image catalog. |
| `services/conversation_service.py` | Chat history, pricing-email follow-up, pending draft/send state. |
| `services/component_match_service.py` | Formats hybrid match context for GTM/outreach nodes. |
| `services/catalog_image_host.py` | Hosts catalog images for inline `<img>` in Brevo emails. |
| `services/salesforce_client.py` | Backend-agnostic CRM ops (MCP or `simple-salesforce` REST). |
| `services/salesforce_mcp.py` | Spawns TypeScript MCP server over stdio; parses query results. |
| `services/salesforce_repository.py` | Lead fetch/format fast-paths (latest, time window, part enquiry). |
| `agents/__init__.py` | Runtime `ask()` / `confirm_send()` entrypoints + Galileo top-level traces. |
| `agents/graph.py` | LangGraph wiring for nodes and conditional routing. |
| `agents/state.py` | Shared `AgentState` schema and merged `steps` reducer. |
| `agents/chat.py` | `build_turn_context()` — formats history + current message for prompts. |
| `agents/intent.py` | Regex classifiers: CRM/outreach detection, leads fast-path eligibility. |
| `agents/schemas.py` | Pydantic models for structured LLM decisions (route, pricing, send, leads gates). |
| `agents/structured.py` | `invoke_structured()` helper with fallback on parse failure. |
| `agents/constants.py` | Shared regex (`EMAIL_PATTERN`), send phrases, history limit. |
| `agents/router_agent/nodes.py` | Supervisor classification (`gtm` / `outreach` / `crm`). |
| `agents/gtm_agent/nodes.py` | GTM nodes: retrieve, pricing/email gates, answer generation. |
| `agents/outreach_agent/nodes.py` | Outreach nodes: research, draft, send gate, Brevo send + CRM log. |
| `agents/outreach_agent/email_html.py` | Formal HTML email template with optional catalog inline image. |
| `agents/crm_agent/nodes.py` | CRM nodes: fast-path lead fetch, tool loop, answer formatting. |
| `agents/tools/` | Tools by concern (`knowledge_base`, `web`, `apollo`, `salesforce`, `email`) + `runner`. |
| `vector_db/database.py` | Qdrant setup, hybrid search (dense + BM25), add/count/source management. |
| `vector_db/chunker.py` | Text chunking and PDF/Excel/CSV extraction. |
| `vector_db/component_store.py` | CLIP embeddings + Claude captions; hybrid component image matching. |
| `vector_db/vision.py` | Claude vision for query captioning and CLIP candidate re-ranking. |
| `observability/galileo.py` | Galileo SDK: spans, callbacks, traces, sessions, console links. |
| `evals/run_galileo_evals.py` | Eval runner (sessions mode + experiment mode). |
| `evals/README.md` | Evaluation guide and Galileo eval usage details. |

---

## Tech stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Anthropic (`ChatAnthropic`) |
| Embeddings | Qdrant Cloud Inference (`all-MiniLM-L6-v2` dense + BM25 sparse) |
| Component vision | CLIP (`sentence-transformers`) + Claude vision re-rank |
| Vector DB | Qdrant Cloud |
| Web Search | DuckDuckGo (`ddgs`) |
| CRM | Salesforce (TypeScript MCP server; `simple-salesforce` fallback) |
| Leads | Apollo API |
| Email | Brevo |
| UI | Streamlit |
| Observability / Evals | Galileo |

---

## How Galileo SDK is used (Tracing + Evals)

Galileo integration in this repo is centralized and explicit:

- **Core helper layer:** `observability/galileo.py`
  - `ensure_galileo_initialized()` calls `galileo_context.init(...)`
  - `get_langchain_config(...)` injects `GalileoCallback` into LLM/tool invokes
  - `merge_node_config(...)` preserves span nesting across graph nodes
  - `log_span(...)` wraps functions with Galileo span decorators
  - `start_chat_session(...)` starts per-chat Galileo sessions
  - `get_logger_instance()` returns the active logger for trace/session operations

- **Top-level request trace:** `agents/__init__.py`
  - `ask(question)` initializes Galileo when enabled
  - Starts top trace with `logger.start_trace(...)`
  - Concludes and flushes with `logger.conclude(...)` + `logger.flush()`

- **Node + tool tracing:** `agents/router_agent/nodes.py`, `agents/gtm_agent/nodes.py`, `agents/outreach_agent/nodes.py`, `agents/crm_agent/nodes.py`, `agents/tools/runner.py`
  - LLM/tool calls pass `merge_node_config(...)` so `GalileoCallback` captures spans
  - `deliver_brevo_email` uses `@log_span(...)`; `call_tools` intentionally does not (avoids duplicate spans with retrieve nodes)

- **UI session wiring:** `ui/ui.py`
  - `handle_new_prompt(...)` starts one Galileo session per fresh chat via `start_chat_session(...)`
  - Optional console links are exposed by `get_console_links()`

- **Eval integration:** `evals/run_galileo_evals.py`
  - **Sessions mode:** `logger.start_session(...)` per dataset row
  - **Experiment mode:** `run_experiment(...)` from `galileo.experiments`
  - Uses same `ask()` path, so eval and production routing logic stay aligned

Required Galileo env vars are in `.env.example`:

- `GALILEO_API_KEY`
- `GALILEO_PROJECT`
- `GALILEO_LOG_STREAM`

---

## Official links for all core services

| What we use | Link |
|---|---|
| Galileo (observability/evals) | [app.galileo.ai](https://app.galileo.ai/) |
| Anthropic API Console | [console.anthropic.com](https://console.anthropic.com/) |
| Qdrant Cloud | [cloud.qdrant.io](https://cloud.qdrant.io/) |
| LangGraph Docs | [LangGraph documentation](https://langchain-ai.github.io/langgraph/) |
| Streamlit Docs | [docs.streamlit.io](https://docs.streamlit.io/) |
| Apollo API | [apollo.io](https://www.apollo.io/) |
| Brevo | [brevo.com](https://www.brevo.com/) |
| Salesforce MCP (Cursor/Claude) | [mcp-server-salesforce](https://github.com/Ritvik777/mcp-server-salesforce) |
| DuckDuckGo Search package | [ddgs on PyPI](https://pypi.org/project/ddgs/) |

---

## Setup and run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `grandalf` (in requirements) enables ASCII graph fallback in the Streamlit sidebar when PNG rendering is unavailable. `sentence-transformers` + `torch` + `pillow` are required for component image matching (CLIP).

### 2) Configure environment

```bash
cp .env.example .env
```

Fill `.env` with your values:

- **Core:**
  - `ANTHROPIC_API_KEY`
  - `QDRANT_URL`
  - `QDRANT_API_KEY`
  - `ANTHROPIC_MODEL` (optional, default `claude-haiku-4-5`)
- **Component vision (optional):**
  - `VISION_MODEL`, `VISION_CAPTION_MODEL`, `VISION_RERANK_MODEL` — Claude models for caption/re-rank
  - `CLIP_MODEL_NAME` (default `clip-ViT-B-32`)
  - `COMPONENT_COLLECTION_NAME`, `COMPONENT_RERANK_POOL`, `COMPONENT_CLIP_CANDIDATES`
  - `COMPONENT_IMAGE_PUBLIC_BASE_URL` — CDN base for inline email images
  - `CATALOG_IMAGE_TEMP_UPLOAD` — auto-upload to catbox.moe for Brevo inline `<img>` (default `true`)
- **Outreach (optional):**
  - `APOLLO_API_KEY`
  - `BREVO_API_KEY`
  - `BREVO_FROM_EMAIL`
  - `BREVO_FROM_NAME` (optional)
- **Salesforce CRM** via [TypeScript MCP server](https://github.com/Ritvik777/mcp-server-salesforce):
  - `SALESFORCE_BACKEND=mcp` (default when Node/npx is installed) spawns `@ritvik777/mcp-server-salesforce` over stdio
  - `SALESFORCE_BACKEND=python` uses `simple-salesforce` REST (no Node required)
  - `SALESFORCE_MCP_COMMAND` / `SALESFORCE_MCP_ARGS` — same as Claude Desktop MCP config
  - Auth: `SALESFORCE_CONNECTION_TYPE` + username/password, OAuth, or `Salesforce_CLI` (MCP only)
- **Observability/evals:**
  - `GALILEO_API_KEY`
  - `GALILEO_PROJECT`
  - `GALILEO_LOG_STREAM`

### 3) Start app

```bash
streamlit run app.py
```

---

## Evaluations

Run baseline eval suite:

```bash
python evals/run_galileo_evals.py
```

Experiment mode:

```bash
GALILEO_EVAL_MODE=experiment python evals/run_galileo_evals.py
```

See full eval documentation in `evals/README.md`.

---

## GitHub Pages (branch deploy)

This repo uses root `index.html` for the docs page.

In GitHub settings:

- Pages source: **Deploy from a branch**
- Branch: `main`
- Folder: `/(root)`

Then open:

- `https://ritvik777.github.io/TradeflowCRM/`
