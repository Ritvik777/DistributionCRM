# Product Marketing — Multi-Agent RAG with Observations & Evals

This repository is a **Product Marketing** assistant built with LangGraph.

This multi-agent system is built for two production workflows in one assistant: **GTM product support** and **outreach content execution**.  
Each user request is first routed to the correct specialist branch, then processed through branch-specific nodes that gather context, apply business gates (pricing email verification or send intent), and return a final response with full trace visibility.

- Supervisor Routing Agent decides between **GTM** and **Outreach** behavior
- GTM branch answers product and pricing questions from your knowledge base
- Outreach branch creates content, finds leads, and can send emails
- Full observability with Galileo tracing/session support (optional)

Project explainer page (GitHub Pages): [Click here for Technical Understanding Blog](https://ritvik777.github.io/AI-Market/)

---

## Architecture (Current)

### Graph flow

```text
START -> classify
          |- gtm      -> gtm_retrieve -> pricing_gate --not_pricing--> gtm_generate -> END
          |                                         \--pricing-------> collect_email --valid--> gtm_generate -> END
          |                                                                            \--no_email----------> END
          |
          \- outreach -> outreach_research -> outreach_generate -> send_gate --review--> END
                                                                     \--send------------> outreach_send -> END
```

### Agents

- **Supervisor Routing Agent** (`agents/router_agent/nodes.py`)
  - Uses LLM for intent classification
  - Routes to `gtm` or `outreach`

- **GTM Agent** (`agents/gtm_agent/nodes.py`)
  - Retrieves context from internal KB + web
  - Pricing gate requires verified email before full pricing output
  - Generates final product/pricing response

- **Outreach Agent** (`agents/outreach_agent/nodes.py`)
  - Researches context (and uses Apollo for lead-intent prompts)
  - Generates marketing content (email/post)
  - Send gate determines review-only vs actual send via Brevo

### Shared state

Defined in `agents/state.py`:

- `question`
- `agent_type`
- `context`
- `answer`
- `is_pricing`
- `user_email`
- `send_requested`
- `steps` (merged pipeline trace)

---

## Key files

```text
app.py                          # Streamlit entrypoint
ui/ui.py                        # Sidebar, chat, trace rendering
agents/graph.py                 # LangGraph node wiring
agents/router_agent/nodes.py    # classify + route
agents/gtm_agent/nodes.py       # GTM branch nodes
agents/outreach_agent/nodes.py  # Outreach branch nodes
agents/tools.py                 # KB/web/Apollo/Brevo tools + tool loop
vector_db/database.py           # Qdrant hybrid search (dense + BM25 via Cloud Inference)
vector_db/chunker.py            # text/pdf chunking
observability/galileo.py        # tracing/session setup
evals/run_galileo_evals.py      # baseline evaluation suite
```

### File-by-file map (detailed)

| File | What it does |
|---|---|
| `app.py` | Main Streamlit entrypoint that initializes app shell and chat loop. |
| `ui/ui.py` | UI logic: styling, sidebar blocks, upload handlers, chat rendering, and trace display. |
| `services/agent_service.py` | Service adapter for `ask()`, `load_graph_image()` (PNG), and `load_graph_ascii()` (fallback when PNG fails). |
| `services/vector_db_service.py` | Service adapter for adding docs/PDFs and reading DB counts. |
| `agents/__init__.py` | Runtime `ask()` entrypoint, `get_graph_image()` (PNG), `get_graph_ascii()` (fallback for UI graph). |
| `agents/graph.py` | LangGraph wiring for nodes and conditional routing. |
| `agents/state.py` | Shared `AgentState` schema and merged `steps` reducer behavior. |
| `agents/router_agent/nodes.py` | Supervisor Routing Agent classification logic (`gtm` vs `outreach`) using LLM. |
| `agents/gtm_agent/nodes.py` | GTM branch nodes: retrieve, pricing/email gates, and GTM answer generation. |
| `agents/outreach_agent/nodes.py` | Outreach branch nodes: research, draft generation, send gate, send execution. |
| `agents/tools.py` | Shared tools and tool-routing loop (`search_knowledge_base`, `web_search`, `apollo_search`, `send_email`). |
| `vector_db/database.py` | Qdrant setup, hybrid search (dense + BM25), add/count operations. |
| `vector_db/chunker.py` | Text chunking and PDF/Excel/CSV extraction utilities. |
| `llm.py` | Anthropic model factory and env validation. |
| `config.py` | Global config/env variable loading. |
| `observability/galileo.py` | Galileo SDK integration for spans, callbacks, traces, sessions, and console links. |
| `evals/run_galileo_evals.py` | Eval runner (sessions mode + experiment mode + tool coverage checks). |
| `evals/README.md` | Evaluation guide and Galileo eval usage details. |

---

## Tech stack

| Component | Technology |
|---|---|
| Orchestration | LangGraph |
| LLM | Anthropic (`ChatAnthropic`) |
| Embeddings | Qdrant Cloud Inference (`all-MiniLM-L6-v2` dense + BM25 sparse) |
| Vector DB | Qdrant Cloud |
| Web Search | DuckDuckGo |
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
  - `log_span(...)` wraps functions with Galileo span decorators
  - `start_chat_session(...)` starts per-chat Galileo sessions
  - `get_logger_instance()` returns the active logger for trace/session operations

- **Top-level request trace:** `agents/__init__.py`
  - `ask(question)` initializes Galileo when enabled
  - Starts top trace with `logger.start_trace(...)`
  - Concludes and flushes with `logger.conclude(...)` + `logger.flush()`

- **Node + tool tracing:** `agents/router_agent/nodes.py`, `agents/gtm_agent/nodes.py`, `agents/outreach_agent/nodes.py`, `agents/tools.py`
  - LLM/tool calls pass `merge_node_config(...)` so `GalileoCallback` captures spans
  - `send_email` also uses `@log_span(...)`; `call_tools` intentionally does not (avoids duplicate spans with retrieve nodes)

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
| DuckDuckGo Search package | [duckduckgo-search on PyPI](https://pypi.org/project/duckduckgo-search/) |

## Setup and run

### 1) Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `grandalf` (in requirements) enables ASCII graph fallback in the Streamlit sidebar when PNG rendering (Mermaid.INK API) is unavailable.

### 2) Configure environment

```bash
cp .env.example .env
```

Fill `.env` with your values:

- Core:
  - `ANTHROPIC_API_KEY`
  - `QDRANT_URL`
  - `QDRANT_API_KEY`
- Optional outreach features:
  - `APOLLO_API_KEY`
  - `BREVO_API_KEY`
  - `BREVO_FROM_EMAIL`
  - `BREVO_FROM_NAME` (optional)
- Observability/evals:
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

This repo uses root `index.html` for docs page.

In GitHub settings:

- Pages source: **Deploy from a branch**
- Branch: `main`
- Folder: `/(root)`

Then open:

- `https://ritvik777.github.io/AI-Market/`
