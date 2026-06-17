import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from llm import get_llm
from observability import merge_node_config


# GalileoCallback (via gtm_retrieve/outreach_research nodes) logs LLM + tool calls;
# no @log_span to avoid duplicate call_tools span (same work as gtm_retrieve).
def call_tools(question, tools, system_prompt, config=None):
    """LLM picks which tools to call, runs them, returns results.
    Pass config from the graph so LLM/tool spans nest under the parent node."""
    tool_map = {t.name: t for t in tools}
    try:
        llm = get_llm().bind_tools(tools)
    except Exception as exc:
        return f"LLM_UNAVAILABLE: {exc}", []
    msgs = [SystemMessage(content=system_prompt), HumanMessage(content=question)]

    log = []
    seen_calls = set()
    invoke_config = merge_node_config(
        config,
        metadata={"component": "tools", "question": question},
        tags=["agent:shared", "phase:tool-routing"],
    )
    for _ in range(3):
        try:
            resp = llm.invoke(msgs, config=invoke_config or None)
        except Exception as exc:
            return f"LLM_ERROR: {exc}", log
        msgs.append(resp)
        if not resp.tool_calls:
            break
        for tc in resp.tool_calls:
            signature = f"{tc['name']}::{json.dumps(tc.get('args', {}), sort_keys=True, default=str)}"
            if signature in seen_calls:
                msgs.append(ToolMessage(content="Skipped duplicate tool call.", tool_call_id=tc["id"]))
                continue
            seen_calls.add(signature)
            # Per-tool config so Galileo shows which tool was used + args in span metadata
            args_str = json.dumps(tc.get("args", {}), default=str)[:200]
            tool_config = merge_node_config(
                invoke_config,
                metadata={
                    "tool_name": tc["name"],
                    "tool_args": args_str,
                },
                tags=["tool", f"tool:{tc['name']}"],
            )
            try:
                out = tool_map[tc["name"]].invoke(tc["args"], config=tool_config)
            except Exception as exc:
                out = f"TOOL_ERROR[{tc['name']}]: {exc}"
            log.append(tc["name"])
            msgs.append(ToolMessage(content=str(out), tool_call_id=tc["id"]))

    context = "\n\n".join(m.content for m in msgs if isinstance(m, ToolMessage))
    return context or "No context found.", log
