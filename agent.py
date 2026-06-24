"""
EnergyPlus MCP Agent - Minimal version for Ubuntu
Serves a chat UI on http://localhost:5000
"""

import os
import sys
import copy
import asyncio
import logging
import socket
import subprocess
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
import httpx

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
WORKSPACE_DIR = Path(__file__).parent.resolve()

# Docker command to launch the MCP server
MCP_SERVER_COMMAND = "docker"
MCP_SERVER_ARGS = [
    "run", "--rm", "-i",
    "--user", "root",
    "-v", f"{WORKSPACE_DIR / 'EnergyPlus-MCP'}:/workspace",
    "-v", "energyplus-mcp-deps:/root/.cache/uv",
    "-w", "/workspace/energyplus-mcp-server",
    "energyplus-mcp-dev",
    "uv", "run", "--no-dev", "python", "-m", "energyplus_mcp_server.server",
]

SYSTEM_PROMPT = """You are an EnergyPlus building energy simulation expert assistant.
You have access to MCP tools that let you work with EnergyPlus IDF building models.

Your capabilities include:
- Loading and inspecting EnergyPlus IDF models
- Viewing model summaries, zones, surfaces, materials, and constructions
- Checking and modifying simulation settings
- Inspecting and modifying building components
- Running EnergyPlus simulations and analyzing results
- Modifying building envelopes

CRITICAL RULES:
1. ALWAYS use your tools proactively. 
2. If the user asks you to operate on a file but doesn't provide the exact path/name (e.g. "this one" or "a sample file"), DO NOT ask them for the path. Instead, immediately use your tools (like listing available sample files or checking the directory) to find the available files, and then either proceed or ask the user which specific one from the list they meant.
3. Be helpful, precise, and concise."""

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
mcp_client = None
mcp_tools = None
agent_executors = {}
init_status = "pending"
init_error = ""


def _get_executor(model_name: str):
    """Get or create a ReAct agent for the given model."""
    name = model_name.strip().removeprefix("models/")
    if not mcp_tools:
        return None
    if name not in agent_executors:
        if name.startswith("gemini"):
            llm = ChatGoogleGenerativeAI(
                model=name, google_api_key=GOOGLE_API_KEY,
                temperature=0.1, convert_system_message_to_human=False,
            )
        else:
            llm = ChatOllama(model=name, temperature=0.1)
        agent_executors[name] = create_react_agent(llm, mcp_tools, prompt=SYSTEM_PROMPT)
    return agent_executors[name]


# ---------------------------------------------------------------------------
# Schema patch — Gemini requires 'items' on every array parameter
# ---------------------------------------------------------------------------
def _patch_array_items(node):
    if not isinstance(node, dict):
        return
    if node.get("type") == "array" and "items" not in node:
        node["items"] = {"type": "string"}
    for v in node.values():
        if isinstance(v, dict):
            _patch_array_items(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    _patch_array_items(item)


def _fix_tool_schemas(tools):
    for tool in tools:
        schema_cls = getattr(tool, "args_schema", None)
        if schema_cls is None:
            continue
        try:
            orig = (schema_cls.model_json_schema.__func__
                    if hasattr(schema_cls.model_json_schema, "__func__")
                    else schema_cls.model_json_schema)

            def _make(fn):
                @classmethod
                def patched(cls, *a, **kw):
                    try:
                        s = copy.deepcopy(fn(cls, *a, **kw))
                    except TypeError:
                        s = copy.deepcopy(fn(*a, **kw))
                    _patch_array_items(s)
                    return s
                return patched
            schema_cls.model_json_schema = _make(orig)
        except Exception:
            pass
    return tools


# ---------------------------------------------------------------------------
# Docker pre-flight
# ---------------------------------------------------------------------------
def _check_docker() -> tuple[bool, str]:
    """Return (ok, error_msg). Checks daemon + image."""
    try:
        r = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=10)
        if r.returncode != 0:
            s = r.stderr.lower()
            if "permission denied" in s:
                return False, (
                    "Docker permission denied.\n"
                    "Fix: sudo chmod 666 /var/run/docker.sock\n"
                    "  or: sudo usermod -aG docker $USER  (then log out & in)"
                )
            return False, f"Docker not accessible:\n{r.stderr.strip()}"
    except FileNotFoundError:
        return False, "Docker is not installed."
    except subprocess.TimeoutExpired:
        return False, "Docker timed out."

    r = subprocess.run(["docker", "image", "inspect", "energyplus-mcp-dev"],
                       capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        return False, (
            "Image 'energyplus-mcp-dev' not found.\n"
            "Build: docker build -t energyplus-mcp-dev "
            "-f EnergyPlus-MCP/.devcontainer/Dockerfile EnergyPlus-MCP/.devcontainer"
        )
    return True, ""


# ---------------------------------------------------------------------------
# MCP initialization (runs in background)
# ---------------------------------------------------------------------------
async def _initialize_mcp():
    global mcp_client, mcp_tools, init_status, init_error

    ok, err = _check_docker()
    if not ok:
        logger.error(f"Docker check failed: {err}")
        init_status, init_error = "error", err
        return

    init_status = "connecting"
    logger.info("Docker OK — connecting to MCP server…")

    try:
        mcp_client = MultiServerMCPClient({
            "energyplus": {
                "command": MCP_SERVER_COMMAND,
                "args": MCP_SERVER_ARGS,
                "transport": "stdio",
            }
        })
        tools = await mcp_client.get_tools()
        logger.info(f"Connected! {len(tools)} MCP tools available.")
        mcp_tools = _fix_tool_schemas(tools)

        # Pre-create the default model executor
        _get_executor(GEMINI_MODEL)
        init_status = "ready"
        logger.info(f"Agent ready  ✔  model={GEMINI_MODEL}  tools={len(tools)}")
    except Exception as e:
        logger.error(f"MCP init failed: {e}", exc_info=True)
        mcp_client, mcp_tools = None, None
        init_status, init_error = "error", str(e)


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(_app: FastAPI):
    if not GOOGLE_API_KEY:
        logger.error("GOOGLE_API_KEY not set — edit .env")
        yield
        return

    task = asyncio.create_task(_initialize_mcp())
    yield

    if not task.done():
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
    logger.info("Server stopped.")


# ---------------------------------------------------------------------------
# App + routes
# ---------------------------------------------------------------------------
app = FastAPI(title="EnergyPlus MCP Agent", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    index = static_dir / "index.html"
    if index.exists():
        return HTMLResponse(index.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EnergyPlus MCP Agent</h1><p>Static files not found.</p>")


@app.get("/api/health")
async def health():
    return {
        "status": "ok",
        "agent_ready": init_status == "ready",
        "init_status": init_status,
        "init_error": init_error if init_status == "error" else "",
        "model": GEMINI_MODEL,
        "api_key_set": bool(GOOGLE_API_KEY),
    }


@app.get("/api/models")
async def get_models():
    models = [
        {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash", "provider": "google"},
        {"id": "gemini-2.5-flash", "name": "Gemini 2.5 Flash", "provider": "google"},
        {"id": "gemini-2.0-flash", "name": "Gemini 2.0 Flash", "provider": "google"},
        {"id": "gemini-2.5-pro", "name": "Gemini 2.5 Pro", "provider": "google"},
    ]
    try:
        async with httpx.AsyncClient() as client:
            resp = await client.get("http://localhost:11434/api/tags", timeout=2.0)
            if resp.status_code == 200:
                data = resp.json()
                for m in data.get("models", []):
                    models.append({
                        "id": m["name"],
                        "name": m["name"],
                        "provider": "ollama"
                    })
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")
        
    return JSONResponse({"models": models})


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_msg = body.get("message", "").strip()
        model = body.get("model", GEMINI_MODEL).strip()

        if not user_msg:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        if not GOOGLE_API_KEY and model.startswith("gemini"):
            return JSONResponse({"response": "⚠️ Set GOOGLE_API_KEY in .env and restart to use Gemini models.", "tools_used": []})

        executor = _get_executor(model)
        if executor is None:
            msg = "Still connecting…" if init_status == "connecting" else f"Not ready. {init_error}"
            return JSONResponse({"response": msg, "tools_used": []})

        logger.info(f"Chat [{model}]: {user_msg[:100]}")
        result = await executor.ainvoke({"messages": [{"role": "user", "content": user_msg}]})

        # Extract tool calls
        tools_used, seen = [], set()
        for msg in result.get("messages", []):
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "?")
                    if name not in seen:
                        seen.add(name)
                        tools_used.append({"name": name, "args": tc.get("args", {})})

        # Extract final AI response
        response = ""
        for msg in reversed(result.get("messages", [])):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                c = msg.content
                if isinstance(c, list):
                    response = "\n".join(
                        b.get("text", "") if isinstance(b, dict) and b.get("type") == "text"
                        else (b if isinstance(b, str) else "")
                        for b in c
                    ).strip()
                else:
                    response = c
                if response:
                    break
        if not response:
            response = "Request processed (no text response generated)."

        return JSONResponse({"response": response, "tools_used": tools_used})

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        code = 429 if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) else 500
        return JSONResponse({"error": str(e)}, status_code=code)


@app.get("/api/tools")
async def list_tools():
    if not mcp_tools:
        return JSONResponse({"tools": [], "error": "Not connected"})
    return JSONResponse({"tools": [{"name": t.name, "description": t.description} for t in mcp_tools]})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    if not GOOGLE_API_KEY:
        print("\n  ⚠  GOOGLE_API_KEY not set! Edit .env first.\n")

    # Find free port
    port = 5000
    for p in range(5000, 5010):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("0.0.0.0", p))
                port = p
                break
            except OSError:
                continue

    if port != 5000:
        print(f"  ⚠  Port 5000 busy — using {port}")

    print(f"\n{'=' * 50}")
    print(f"  EnergyPlus MCP Agent")
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  URL:   http://localhost:{port}")
    print(f"{'=' * 50}\n")

    uvicorn.run("agent:app", host="0.0.0.0", port=port, log_level="info")
