"""Call the TypeScript Salesforce MCP server (@ritvik777/mcp-server-salesforce) from Python."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
from typing import Any

_MCP_CONTENT_RE = re.compile(r"^\s*(\w+(?:\.\w+)*):\s*(.*)$", re.MULTILINE)


def salesforce_backend() -> str:
    """Return 'mcp' or 'python'. Defaults to mcp when npx/node is available."""
    explicit = (os.getenv("SALESFORCE_BACKEND") or "").strip().lower()
    if explicit in ("mcp", "python"):
        return explicit
    if shutil.which("npx") or shutil.which("node"):
        return "mcp"
    return "python"


def is_mcp_available() -> bool:
    return bool(shutil.which(_mcp_command()))


def _mcp_command() -> str:
    return (os.getenv("SALESFORCE_MCP_COMMAND") or "npx").strip()


def _mcp_args() -> list[str]:
    raw = os.getenv("SALESFORCE_MCP_ARGS") or "-y @ritvik777/mcp-server-salesforce"
    return raw.split()


def _salesforce_env() -> dict[str, str]:
    """Pass Salesforce auth vars to the MCP subprocess (same as Claude Desktop config)."""
    keys = (
        "SALESFORCE_CONNECTION_TYPE",
        "SALESFORCE_USERNAME",
        "SALESFORCE_PASSWORD",
        "SALESFORCE_TOKEN",
        "SALESFORCE_INSTANCE_URL",
        "SALESFORCE_CLIENT_ID",
        "SALESFORCE_CLIENT_SECRET",
        "PATH",
        "HOME",
        "USER",
    )
    env = {
        k: os.environ[k]
        for k in os.environ
        if k in keys or k.startswith("SALESFORCE_")
    }
    return env


def _content_to_text(result: Any) -> str:
    if getattr(result, "isError", False):
        parts = []
        for block in getattr(result, "content", []) or []:
            text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
            if text:
                parts.append(text)
        raise RuntimeError("\n".join(parts) or "MCP tool returned an error")

    parts: list[str] = []
    for block in getattr(result, "content", []) or []:
        text = getattr(block, "text", None) or (block.get("text") if isinstance(block, dict) else None)
        if text:
            parts.append(text)
    return "\n".join(parts) if parts else json.dumps(result, default=str)


async def _call_tool_async(tool_name: str, arguments: dict[str, Any]) -> str:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=_mcp_command(),
        args=_mcp_args(),
        env=_salesforce_env(),
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments=arguments)
            return _content_to_text(result)


def call_mcp_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Run one MCP tool call (spawns the Node server via stdio)."""
    if not is_mcp_available():
        raise RuntimeError(
            "Salesforce MCP requires Node.js. Install node/npx or set SALESFORCE_BACKEND=python"
        )
    return asyncio.run(_call_tool_async(tool_name, arguments))


async def _run_mcp_batch(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    server_params = StdioServerParameters(
        command=_mcp_command(),
        args=_mcp_args(),
        env=_salesforce_env(),
    )
    outputs: list[str] = []
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for tool_name, arguments in calls:
                result = await session.call_tool(tool_name, arguments=arguments)
                outputs.append(_content_to_text(result))
    return outputs


def call_mcp_tools_batch(calls: list[tuple[str, dict[str, Any]]]) -> list[str]:
    """Run multiple MCP tool calls in one subprocess (faster for post-send CRM updates)."""
    if not is_mcp_available():
        raise RuntimeError(
            "Salesforce MCP requires Node.js. Install node/npx or set SALESFORCE_BACKEND=python"
        )
    return asyncio.run(_run_mcp_batch(calls))


def parse_query_records(text: str) -> list[dict[str, str]]:
    """Parse salesforce_query_records MCP text into flat dicts per record."""
    records: list[dict[str, str]] = []
    for chunk in re.split(r"\nRecord \d+:\n", text):
        chunk = chunk.strip()
        if not chunk or chunk.startswith("Query returned 0"):
            continue
        record: dict[str, str] = {}
        for match in _MCP_CONTENT_RE.finditer(chunk):
            key = match.group(1).split(".")[-1]
            record[key] = match.group(2).strip()
        if record:
            records.append(record)
    return records


def mcp_query_records(
    object_name: str,
    fields: list[str],
    where_clause: str = "",
    order_by: str = "",
    limit: int | None = None,
) -> str:
    args: dict[str, Any] = {"objectName": object_name, "fields": fields}
    if where_clause:
        args["whereClause"] = where_clause
    if order_by:
        args["orderBy"] = order_by
    if limit is not None:
        args["limit"] = limit
    return call_mcp_tool("salesforce_query_records", args)


def mcp_dml_records(
    operation: str,
    object_name: str,
    records: list[dict[str, Any]],
    external_id_field: str = "",
) -> str:
    args: dict[str, Any] = {
        "operation": operation,
        "objectName": object_name,
        "records": records,
    }
    if external_id_field:
        args["externalIdField"] = external_id_field
    return call_mcp_tool("salesforce_dml_records", args)
