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
from langgraph.checkpoint.memory import MemorySaver  # NEW: in-memory conversation state
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

ALL_FILES_DIR = WORKSPACE_DIR / "all_files"
ALL_FILES_DIR.mkdir(exist_ok=True)

OUTPUTS_DIR = WORKSPACE_DIR / "outputs"
OUTPUTS_DIR.mkdir(exist_ok=True)

# Initial population of all_files if empty
if not list(ALL_FILES_DIR.glob("*.idf")):
    import shutil
    sources = [
        WORKSPACE_DIR / "EnergyPlus-MCP" / "energyplus-mcp-server" / "sample_files",
        WORKSPACE_DIR / "epMCP" / "sample_models",
        WORKSPACE_DIR / "EnergyPlus-MCP" / "energyplus-mcp-server" / "illustrative examples",
        WORKSPACE_DIR / "EnergyPlus-MCP",
    ]
    for src in sources:
        if src.exists():
            for ext in ("*.idf", "*.epw", "*.idd", "*.csv"):
                for f in src.glob(ext):
                    dest = ALL_FILES_DIR / f.name
                    if not dest.exists():
                        try:
                            shutil.copy2(f, dest)
                        except Exception:
                            pass

# Remove legacy organize_all_files.py if present
if (WORKSPACE_DIR / "organize_all_files.py").exists():
    try:
        (WORKSPACE_DIR / "organize_all_files.py").unlink()
    except Exception:
        pass

# Copy visualization_and_converter.py to ifc_visualizer.py inside epMCP package
_vis_src = WORKSPACE_DIR / "visualization_and_converter.py"
_vis_dst = WORKSPACE_DIR / "epMCP" / "epmcp_mcp_server" / "ifc_visualizer.py"
if _vis_src.exists():
    import shutil
    try:
        shutil.copy2(_vis_src, _vis_dst)
    except Exception:
        pass

# Docker command to launch the EnergyPlus MCP server
MCP_SERVER_COMMAND = "docker"
MCP_SERVER_ARGS = [
    "run", "--rm", "-i",
    "--user", "root",
    "-v", f"{WORKSPACE_DIR / 'EnergyPlus-MCP'}:/workspace",
    "-v", f"{ALL_FILES_DIR}:/workspace/all_files",
    "-v", f"{OUTPUTS_DIR}:/workspace/outputs",
    "-e", "SAMPLE_FILES_PATH=/workspace/all_files",
    "-e", "OUTPUT_DIR=/workspace/outputs",
    "-v", "energyplus-mcp-deps:/root/.cache/uv",
    "-w", "/workspace/energyplus-mcp-server",
    "energyplus-mcp-dev",
    "uv", "run", "--no-dev", "python", "-m", "energyplus_mcp_server.server",
]

# Docker command to launch the epMCP tools server (same Docker image, different mount)
EPMCP_SERVER_COMMAND = "docker"
EPMCP_SERVER_ARGS = [
    "run", "--rm", "-i",
    "--user", "root",
    "-v", f"{WORKSPACE_DIR / 'EnergyPlus-MCP'}:/workspace",
    "-v", f"{WORKSPACE_DIR / 'epMCP'}:/workspace/epMCP",
    "-v", f"{ALL_FILES_DIR}:/workspace/all_files",
    "-v", f"{OUTPUTS_DIR}:/workspace/outputs",
    "-e", "SAMPLE_FILES_PATH=/workspace/all_files",
    "-e", "OUTPUT_DIR=/workspace/outputs",
    "-v", "epmcp-deps:/root/.cache/uv",
    "-w", "/workspace/epMCP",
    "energyplus-mcp-dev",
    "uv", "run", "--no-dev", "python", "-m", "epmcp_mcp_server.server",
]

SYSTEM_PROMPT = """You are an EnergyPlus building energy simulation expert assistant.
You have access to multiple MCP tools that let you work with EnergyPlus IDF building models,
modify occupancy parameters, run calibration loops, and analyze simulation results.

Your capabilities include:
- Loading and inspecting EnergyPlus IDF models
- Viewing model summaries, zones, surfaces, materials, and constructions
- Checking and modifying simulation settings
- Inspecting and modifying building components and occupancy (modify_people, inspect_people)
- Running EnergyPlus simulations and analyzing results (run_energyplus_simulation)
- Modifying building envelopes
- Calculating RMSE between simulation and measured data (calculate_rmse_tool)
- Automated calibration of occupancy against measured data using Nelder-Mead optimization (calibrate_occupancy_tool)
- Inspecting IFC building models, extracting storeys/spaces/elements metadata, and generating interactive 3D HTML visualizations (inspect_and_visualize_ifc_tool)

CRITICAL RULES:

1. Use tools appropriately to fulfill the user's request. For informational or listing queries (e.g. asking what files exist), use discovery tools to find the answer, but DO NOT run heavy processing tools, simulations, or visualizers unless specifically requested by the user.

2. If the user asks you to operate on a file but doesn't provide the exact path/name (e.g. "this one" or "a sample file"), DO NOT ask them for the path. Instead, use your tools to discover candidate files. If there is only one obvious match, proceed automatically. If multiple valid matches exist, present the discovered options and ask the user to choose.

3. Be helpful, precise, and concise.

4. For error calculation and automated occupancy calibration tasks, use the appropriate epMCP tools (calculate_rmse_tool, calibrate_occupancy_tool).

5. All sample EnergyPlus IDF building models, EPW weather files, IDD schema files, and target CSV data are organized and available in `/workspace/all_files/` (e.g. `/workspace/all_files/1ZoneUncontrolled.idf`, `/workspace/all_files/USA_CO_Denver.Intl.AP.725650_TMY3.epw`, `/workspace/all_files/Energy+.idd`). Both MCP servers share access to this folder.

6. Pass absolute file paths directly to tools whenever supported (for example, `/workspace/all_files/1ZoneUncontrolled.idf` or `/workspace/all_files/USA_CO_Denver.Intl.AP.725650_TMY3.epw`). DO NOT use `copy_file` before running simulations. Save all generated outputs to `/workspace/outputs`.

7. Answer user queries directly, accurately, and concisely. When asked to list files (such as IDF, EPW, or IFC files), list the discovered files cleanly and stop—do not automatically invoke visualization, simulation, or conversion tools.

9. Never fabricate information. Never invent:
   - file names
   - file paths
   - simulation outputs
   - calibration results
   - RMSE values
   - generated files
   - EnergyPlus results
   - surrogate predictions

   Only report information returned by tools.

10. Prefer complete workflows over partial workflows. When possible, autonomously complete the entire engineering task rather than stopping after a single tool call. If one tool naturally leads to another, continue automatically without waiting for additional user instructions.

11. Never ask the user for information that can be discovered using available tools. Always inspect the workspace, available files, simulation outputs, models, or metadata before requesting clarification.

12. Before modifying or simulating a model, verify that all required inputs exist (IDF, EPW, schedules, output directory, etc.). If verification fails, explain the problem and attempt automatic recovery whenever possible.

13. Use absolute file paths whenever tools accept file paths. Never construct or guess file paths. Use paths returned by discovery tools.

14. After execution completes, ALWAYS provide the following sections:

   =====================================================
   EXECUTION SUMMARY
   =====================================================
   A concise summary of what was accomplished.

   =====================================================
   TOOLS USED
   =====================================================
   List every tool that was actually invoked.

   =====================================================
   GENERATED FILES
   =====================================================
   List every file that was created or modified.

   =====================================================
   RESULTS
   =====================================================
   Present the important outputs, metrics, and observations.


15. Your primary objective is to minimize unnecessary user interaction. Discover information, validate inputs, execute appropriate tools, recover from recoverable errors, and complete engineering workflows autonomously whenever it is safe to do so."""


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
mcp_client = None
mcp_tools = None
agent_executors = {}
init_status = "pending"
init_error = ""

# NEW: single shared checkpointer for all agents/models.
# This is what actually gives the agent memory across turns.
# Note: MemorySaver is in-process only — history is lost on server restart.
# For persistence across restarts, swap this for AsyncSqliteSaver or
# AsyncPostgresSaver (from langgraph.checkpoint.sqlite / .postgres).
memory_saver = MemorySaver()


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
        agent_executors[name] = create_react_agent(
            llm,
            mcp_tools,
            prompt=SYSTEM_PROMPT,
            checkpointer=memory_saver,  # NEW: enables multi-turn memory
        )
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
    logger.info("Docker OK — connecting to MCP servers…")

    try:
        mcp_client = MultiServerMCPClient({
            "energyplus": {
                "command": MCP_SERVER_COMMAND,
                "args": MCP_SERVER_ARGS,
                "transport": "stdio",
            },
            "epmcp": {
                "command": EPMCP_SERVER_COMMAND,
                "args": EPMCP_SERVER_ARGS,
                "transport": "stdio",
            },
        })
        tools = await mcp_client.get_tools()
        logger.info(f"Connected! {len(tools)} MCP tools available (EnergyPlus-MCP + epMCP).")
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


@app.get("/api/files")
async def get_files():
    idf_files = sorted([f.name for f in ALL_FILES_DIR.glob("*.idf")])
    epw_files = sorted([f.name for f in ALL_FILES_DIR.glob("*.epw")])
    ifc_files = sorted([f.name for f in ALL_FILES_DIR.glob("*.ifc")])
    return JSONResponse({
        "idf": idf_files,
        "epw": epw_files,
        "ifc": ifc_files
    })


@app.post("/api/visualize_ifc")
async def visualize_ifc(request: Request):
    body = await request.json()
    filename = body.get("filename", "20160414office_model_CV2_fordesign.ifc").strip()
    ifc_path = f"/workspace/all_files/{filename}"
    
    # Try using connected MCP tools first
    if mcp_tools:
        for tool in mcp_tools:
            if tool.name == "inspect_and_visualize_ifc_tool":
                try:
                    res = await tool.ainvoke({
                        "ifc_path": ifc_path,
                        "output_dir": "/workspace/outputs",
                        "output_html_name": "ifc_3d_visualization.html"
                    })
                    gen_file = OUTPUTS_DIR / "ifc_3d_visualization.html"
                    if gen_file.exists():
                        import shutil
                        shutil.copy2(gen_file, static_dir / "ifc_3d_visualization.html")
                    return JSONResponse({"status": "success", "html_url": "/static/ifc_3d_visualization.html", "result": res})
                except Exception as e:
                    logger.error(f"Error invoking inspect_and_visualize_ifc_tool: {e}")
                    
    # Fallback to direct docker run
    cmd = [
        "docker", "run", "--rm", "-i", "--user", "root",
        "-v", f"{WORKSPACE_DIR / 'EnergyPlus-MCP'}:/workspace",
        "-v", f"{WORKSPACE_DIR / 'epMCP'}:/workspace/epMCP",
        "-v", f"{ALL_FILES_DIR}:/workspace/all_files",
        "-v", f"{OUTPUTS_DIR}:/workspace/outputs",
        "-w", "/workspace/epMCP",
        "energyplus-mcp-dev",
        "uv", "run", "python", "-c",
        f"from epmcp_mcp_server.tools import inspect_and_visualize_ifc; inspect_and_visualize_ifc('{ifc_path}', '/workspace/outputs', 'ifc_3d_visualization.html')"
    ]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        gen_file = OUTPUTS_DIR / "ifc_3d_visualization.html"
        if gen_file.exists():
            import shutil
            shutil.copy2(gen_file, static_dir / "ifc_3d_visualization.html")
            return JSONResponse({"status": "success", "html_url": "/static/ifc_3d_visualization.html"})
        return JSONResponse({"error": f"Failed to generate visualization: {r.stderr}"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)



def _clean_response(user_msg: str, response: str) -> str:
    if not response:
        return response

    umsg = user_msg.lower()

    # Clean raw array wrappers like [{'type': 'text', 'text': '...'}]
    if response.startswith("[{'type': 'text'") or response.startswith('[{"type": "text"'):
        try:
            arr = eval(response) if response.startswith("[{") else json.loads(response)
            if isinstance(arr, list) and len(arr) > 0 and isinstance(arr[0], dict):
                text_val = arr[0].get("text", "")
                if text_val:
                    return _clean_response(user_msg, text_val)
        except Exception:
            pass

    # Detect unformatted raw JSON dumps containing sample_files or file lists
    if "sample_files" in response or "IDF files" in response:
        try:
            start_idx = response.find("{")
            end_idx = response.rfind("}") + 1
            if start_idx >= 0 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                sf = data.get("sample_files", data)
                if isinstance(sf, dict):
                    if "ifc" in umsg:
                        ifc_files = [f["name"] for f in sf.get("Other files", []) if f.get("name", "").lower().endswith(".ifc")]
                        if ifc_files:
                            return "Available IFC files in `/workspace/all_files`:\n\n" + "\n".join(f"- 🏗️ `{f}`" for f in ifc_files)
                        else:
                            return "No IFC files found in `/workspace/all_files`."
                    elif "idf" in umsg:
                        idf_files = [f["name"] for f in sf.get("IDF files", [])]
                        return "Available IDF files in `/workspace/all_files`:\n\n" + "\n".join(f"- 📄 `{f}`" for f in idf_files)
                    elif "weather" in umsg or "epw" in umsg:
                        epw_files = [f["name"] for f in sf.get("Weather files", [])]
                        return "Available Weather files in `/workspace/all_files`:\n\n" + "\n".join(f"- 🌤️ `{f}`" for f in epw_files)
                    else:
                        items = []
                        for f in sf.get("IDF files", []):
                            items.append(f"- 📄 `{f['name']}` (IDF model)")
                        for f in sf.get("Weather files", []):
                            items.append(f"- 🌤️ `{f['name']}` (EPW weather)")
                        for f in sf.get("Other files", []):
                            fn = f.get("name", "")
                            if fn.lower().endswith(".ifc"):
                                items.append(f"- 🏗️ `{fn}` (IFC model)")
                        return "Available files in `/workspace/all_files`:\n\n" + "\n".join(items)
        except Exception:
            pass

    return response


@app.post("/api/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        user_msg = body.get("message", "").strip()
        model = body.get("model", GEMINI_MODEL).strip()
        session_id = body.get("session_id", "default").strip() or "default"

        if not user_msg:
            return JSONResponse({"error": "Empty message"}, status_code=400)

        if not GOOGLE_API_KEY and model.startswith("gemini"):
            return JSONResponse({"response": "⚠️ Set GOOGLE_API_KEY in .env and restart to use Gemini models.", "tools_used": []})

        executor = _get_executor(model)
        if executor is None:
            msg = "Still connecting…" if init_status == "connecting" else f"Not ready. {init_error}"
            return JSONResponse({"response": msg, "tools_used": []})

        logger.info(f"Chat [{model}] session=[{session_id}]: {user_msg[:100]}")

        thread_id = f"{session_id}:{model}"
        config = {"configurable": {"thread_id": thread_id}}

        payload = {"messages": [{"role": "user", "content": user_msg}]}

        try:
            result = await executor.ainvoke(payload, config=config)
        except Exception as first_error:
            error_text = str(first_error)
            if "XML syntax error" in error_text and not model.startswith("gemini"):
                logger.warning(
                    "Retrying chat after XML syntax error with fresh thread_id "
                    f"(model={model}, session={session_id})"
                )
                retry_config = {
                    "configurable": {"thread_id": f"{session_id}:{model}:fresh"}
                }
                result = await executor.ainvoke(payload, config=retry_config)
            else:
                raise

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
            # Extract content from recent ToolMessage if AI didn't format it
            for msg in reversed(result.get("messages", [])):
                if hasattr(msg, "type") and msg.type == "tool" and msg.content:
                    raw_content = msg.content
                    try:
                        parsed_json = json.loads(raw_content) if isinstance(raw_content, str) else raw_content
                        if isinstance(parsed_json, dict):
                            items = []
                            if "sample_files" in parsed_json:
                                sf = parsed_json.get("sample_files", {})
                                for f in sf.get("IDF files", []):
                                    items.append(f"📄 `{f['name']}` (IDF model)")
                                for f in sf.get("Weather files", []):
                                    items.append(f"🌤️ `{f['name']}` (EPW weather)")
                                for f in sf.get("Other files", []):
                                    fname = f.get("name", "")
                                    if fname.lower().endswith(".ifc"):
                                        items.append(f"🏗️ `{fname}` (IFC 3D model)")
                                    elif not fname.endswith(".md"):
                                        items.append(f"📁 `{fname}`")
                            if "ifc_files" in parsed_json:
                                items.extend([f"🏗️ `{f}` (IFC 3D model)" for f in parsed_json["ifc_files"]])
                            if items:
                                response = "Available files in `/workspace/all_files`:\n\n" + "\n".join(items)
                            else:
                                response = f"Tool result:\n```json\n{json.dumps(parsed_json, indent=2)}\n```"
                        else:
                            response = str(raw_content)
                    except Exception:
                        response = str(raw_content)
                    if response:
                        break

        if not response:
            response = "Request processed (no text response generated)."

        response = _clean_response(user_msg, response)

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


# NEW: lets the frontend explicitly clear a conversation's memory
@app.post("/api/reset_session")
async def reset_session(request: Request):
    body = await request.json()
    session_id = body.get("session_id", "default").strip() or "default"
    model = body.get("model", GEMINI_MODEL).strip()
    thread_id = f"{session_id}:{model}"
    try:
        # MemorySaver keeps state in a dict keyed by thread_id under the hood;
        # simplest cross-version-safe reset is to just start a new thread_id
        # client-side. If you want a hard server-side wipe, regenerate the
        # session_id on the frontend instead of calling this endpoint.
        return JSONResponse({"status": "ok", "note": "Generate a new session_id client-side to start fresh."})
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


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