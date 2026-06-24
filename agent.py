"""
EnergyPlus MCP Agent - LangChain + Google Gemini + MCP Server
Serves a beautiful chat UI on http://localhost:5000
"""

import os
import sys
import json
import asyncio
import logging
from pathlib import Path
from contextlib import asynccontextmanager
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# Resolve workspace paths
WORKSPACE_DIR = Path(__file__).parent.resolve()
MCP_SERVER_DIR = WORKSPACE_DIR / "EnergyPlus-MCP" / "energyplus-mcp-server"

# MCP server command via Docker (matches existing .vscode/settings.json)
# Uses a named volume 'energyplus-mcp-deps' for the uv cache so packages
# are only installed once and survive container restarts.
MCP_SERVER_COMMAND = "docker"
MCP_SERVER_ARGS = [
    "run",
    "--rm",
    "-i",
    "-v", f"{WORKSPACE_DIR / 'EnergyPlus-MCP'}:/workspace",
    "-v", "energyplus-mcp-deps:/root/.cache/uv",
    "-w", "/workspace/energyplus-mcp-server",
    "energyplus-mcp-dev",
    "uv", "run", "--no-dev", "python", "-m", "energyplus_mcp_server.server",
]

# Global state
mcp_client = None
agent_executor = None
mcp_tools = None
agent_executors = {}


def get_agent_executor(model_name: str):
    """Get or create the ReAct agent executor for the requested model."""
    global agent_executors, mcp_tools, GOOGLE_API_KEY
    
    clean_model = model_name.strip()
    if clean_model.startswith("models/"):
        clean_model = clean_model.replace("models/", "")
        
    if not mcp_tools:
        logger.warning("MCP tools not ready yet, cannot create agent executor")
        return None
        
    if clean_model not in agent_executors:
        logger.info(f"Creating agent executor dynamically for model: {clean_model}")
        try:
            llm = ChatGoogleGenerativeAI(
                model=clean_model,
                google_api_key=GOOGLE_API_KEY,
                temperature=0.1,
                convert_system_message_to_human=False,
            )
            agent_executors[clean_model] = create_react_agent(
                llm,
                mcp_tools,
                prompt=SYSTEM_PROMPT,
            )
        except Exception as e:
            logger.error(f"Failed to create agent executor for model {clean_model}: {e}", exc_info=True)
            return None
            
    return agent_executors[clean_model]


# System prompt for the agent
SYSTEM_PROMPT = """You are an EnergyPlus building energy simulation expert assistant. 
You have access to a suite of MCP (Model Context Protocol) tools that let you work with 
EnergyPlus IDF building models.

Your capabilities include:
- Loading and inspecting EnergyPlus IDF models
- Viewing model summaries, zones, surfaces, materials, and constructions
- Checking and modifying simulation settings (SimulationControl, RunPeriod)
- Inspecting and modifying building components (People, Lights, ElectricEquipment)
- Running EnergyPlus simulations
- Analyzing simulation results
- Modifying building envelopes (window films, coatings, infiltration)
- Copying and managing model files

When answering questions:
1. Use the available tools to gather information before answering
2. Provide detailed, technical explanations
3. Suggest energy efficiency improvements when relevant
4. Always mention which files or models you're working with
5. If a user asks about capabilities, list the tools available to you

Be helpful, precise, and proactive in using your tools."""


# ---------------------------------------------------------------------------
# Background MCP initialization
# ---------------------------------------------------------------------------
init_status = "pending"
init_error = ""


async def _initialize_mcp():
    """Background task to connect to MCP server and create the agent."""

    global mcp_client, agent_executor, mcp_tools, agent_executors, init_status, init_error

    init_status = "connecting"
    logger.info("Initializing MCP client (connecting to EnergyPlus MCP server via Docker)...")
    logger.info(f"Command: {MCP_SERVER_COMMAND} {' '.join(MCP_SERVER_ARGS)}")

    try:
        mcp_client = MultiServerMCPClient(
            {
                "energyplus": {
                    "command": MCP_SERVER_COMMAND,
                    "args": MCP_SERVER_ARGS,
                    "transport": "stdio",
                }
            }
        )

        # In langchain-mcp-adapters >= 0.1.0, get_tools() is awaitable
        tools = await mcp_client.get_tools()
        logger.info(f"Connected! Found {len(tools)} MCP tools:")
        for tool in tools:
            logger.info(f"  - {tool.name}: {tool.description[:80]}...")

        # Fix tool schemas for Gemini compatibility:
        # Gemini strictly requires array parameters to have 'items' defined.
        tools = _fix_tool_schemas(tools)

        # Save to global mcp_tools
        mcp_tools = tools

        # Create the LLM
        llm = ChatGoogleGenerativeAI(
            model=GEMINI_MODEL,
            google_api_key=GOOGLE_API_KEY,
            temperature=0.1,
            convert_system_message_to_human=False,
        )

        # Create the ReAct agent
        agent_executor = create_react_agent(
            llm,
            tools,
            prompt=SYSTEM_PROMPT,
        )

        # Cache the initial model
        clean_model = GEMINI_MODEL.strip()
        if clean_model.startswith("models/"):
            clean_model = clean_model.replace("models/", "")
        agent_executors[clean_model] = agent_executor

        init_status = "ready"
        logger.info(f"Agent ready with {GEMINI_MODEL} + {len(tools)} tools")

    except Exception as e:
        logger.error(f"Failed to initialize MCP client: {e}", exc_info=True)
        mcp_client = None
        agent_executor = None
        mcp_tools = None
        init_status = "error"
        init_error = str(e)

def _fix_tool_schemas(tools):
    """
    Patch tool input schemas for Gemini API compatibility.
    Gemini requires every 'array' type parameter to have an 'items' field.

    LangChain calls `args_schema.model_json_schema()` as a classmethod when
    converting tools for the Gemini API. We patch that classmethod directly.
    """
    import copy

    def _patch_node(node):
        """Recursively add items:{type:string} to any array node missing items."""
        if not isinstance(node, dict):
            return
        if node.get("type") == "array" and "items" not in node:
            node["items"] = {"type": "string"}
        for val in list(node.values()):
            if isinstance(val, dict):
                _patch_node(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, dict):
                        _patch_node(item)

    for tool in tools:
        try:
            schema_cls = getattr(tool, "args_schema", None)
            if schema_cls is None:
                continue

            # Capture original classmethod
            orig_fn = schema_cls.model_json_schema.__func__ if hasattr(
                schema_cls.model_json_schema, "__func__"
            ) else schema_cls.model_json_schema

            def _make_patched(original):
                @classmethod
                def patched(cls, *args, **kwargs):
                    try:
                        s = copy.deepcopy(original(cls, *args, **kwargs))
                    except TypeError:
                        s = copy.deepcopy(original(*args, **kwargs))
                    _patch_node(s)
                    return s
                return patched

            schema_cls.model_json_schema = _make_patched(orig_fn)
            logger.debug(f"Patched classmethod schema for: {tool.name}")

        except Exception as e:
            logger.warning(f"Could not patch schema for {getattr(tool, 'name', '?')}: {e}")

    return tools


# ---------------------------------------------------------------------------
# Lifespan – launch background init, don't block server start
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the server immediately; MCP connects in the background."""
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        logger.error(
            "GOOGLE_API_KEY is not set! "
            "Edit .env and paste your key on the GOOGLE_API_KEY line."
        )
        yield
        return

    # Launch MCP init as a background task so the server starts immediately
    init_task = asyncio.create_task(_initialize_mcp())

    yield

    # Shutdown: cancel init if still running, then clean up
    if not init_task.done():
        init_task.cancel()
        try:
            await init_task
        except asyncio.CancelledError:
            pass

    if mcp_client:
        logger.info("Shutting down MCP client...")
        try:
            await mcp_client.close()
        except Exception as e:
            logger.warning(f"Error during MCP client shutdown: {e}")
    logger.info("Agent server stopped.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
app = FastAPI(
    title="EnergyPlus MCP Agent",
    description="LangChain Agent powered by Google Gemini with EnergyPlus MCP tools",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files (HTML/CSS/JS)
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the chat UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>EnergyPlus MCP Agent</h1><p>Static files not found.</p>")


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "agent_ready": agent_executor is not None,
        "init_status": init_status,
        "init_error": init_error if init_status == "error" else "",
        "model": GEMINI_MODEL,
        "api_key_set": bool(GOOGLE_API_KEY) and GOOGLE_API_KEY != "PASTE_YOUR_API_KEY_HERE",
    }


@app.post("/api/chat")
async def chat(request: Request):
    """Handle chat messages from the browser UI."""
    global GEMINI_MODEL

    try:
        body = await request.json()
        user_message = body.get("message", "").strip()
        selected_model = body.get("model", GEMINI_MODEL).strip()

        if not user_message:
            return JSONResponse(
                {"error": "Empty message"}, status_code=400
            )

        # Check if agent is ready
        if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
            return JSONResponse({
                "response": "⚠️ **Google API Key not configured!**\n\n"
                           "Please edit the `.env` file in your workspace and paste your "
                           "Google Gemini API key on the `GOOGLE_API_KEY=` line.\n\n"
                           "Get a key from: https://aistudio.google.com/apikey\n\n"
                           "Then restart the server.",
                "tools_used": [],
            })

        # Get executor dynamically for the selected model
        executor = get_agent_executor(selected_model)

        if executor is None:
            if init_status == "connecting":
                return JSONResponse({
                    "response": "**Still connecting to EnergyPlus MCP server...**\n\n"
                               "The Docker container is starting up (first run may take a few minutes "
                               "while dependencies install). Please try again in a moment.",
                    "tools_used": [],
                })
            return JSONResponse({
                "response": f"**Agent not initialized for model '{selected_model}'.**\n\n"
                           "The MCP server connection may have failed, or the model initialization failed. "
                           "Check that Docker Desktop is running and the "
                           "`energyplus-mcp-dev` image is built.\n\n"
                           f"Error: {init_error}" if init_error else ""
                           "Check the terminal for error logs.",
                "tools_used": [],
            })

        logger.info(f"User message: {user_message[:100]}... [Model: {selected_model}]")

        # Invoke the agent
        tools_used = []
        final_response = ""

        result = await executor.ainvoke(
            {"messages": [{"role": "user", "content": user_message}]},
        )

        # Extract the response and tool calls from the result
        messages = result.get("messages", [])
        for msg in messages:
            # Check for tool calls (AIMessage with tool_calls)
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    tools_used.append({
                        "name": tc.get("name", "unknown"),
                        "args": tc.get("args", {}),
                    })
            # Check for tool responses
            if hasattr(msg, "type") and msg.type == "tool":
                tools_used.append({
                    "name": getattr(msg, "name", "tool"),
                    "result_preview": str(getattr(msg, "content", ""))[:200],
                })

        # The final AI message is the response
        for msg in reversed(messages):
            if hasattr(msg, "type") and msg.type == "ai" and msg.content:
                content = msg.content
                # Gemini 2.5 returns content as a list of blocks: [{'type':'text','text':'...'}]
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            parts.append(block.get("text", ""))
                        elif isinstance(block, str):
                            parts.append(block)
                    final_response = "\n".join(parts).strip()
                elif isinstance(content, str):
                    final_response = content
                if final_response:
                    break

        if not final_response:
            final_response = "I processed your request but didn't generate a text response. The tool operations may have completed successfully."

        # Deduplicate tools_used - keep only tool calls (not results)
        seen = set()
        unique_tools = []
        for t in tools_used:
            if "args" in t and t["name"] not in seen:
                seen.add(t["name"])
                unique_tools.append({"name": t["name"], "args": t["args"]})

        logger.info(f"Response generated. Tools used: {[t['name'] for t in unique_tools]}")

        return JSONResponse({
            "response": final_response,
            "tools_used": unique_tools,
        })

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error processing chat: {e}", exc_info=True)
        if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
            return JSONResponse(
                {
                    "error": f"Quota or rate limit exceeded for model '{selected_model}'.",
                    "error_code": "RESOURCE_EXHAUSTED",
                    "model": selected_model
                },
                status_code=429
            )
        return JSONResponse(
            {"error": f"Internal error: {str(e)}"}, status_code=500
        )


@app.get("/api/tools")
async def list_tools():
    """List all available MCP tools."""
    if mcp_client is None:
        return JSONResponse({"tools": [], "error": "MCP client not connected"})

    try:
        tools = mcp_client.get_tools()
        tool_list = []
        for tool in tools:
            tool_list.append({
                "name": tool.name,
                "description": tool.description,
            })
        return JSONResponse({"tools": tool_list})
    except Exception as e:
        return JSONResponse({"tools": [], "error": str(e)})


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    import sys
    
    # Fix Windows console encoding
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    # Validate environment
    if not GOOGLE_API_KEY or GOOGLE_API_KEY == "PASTE_YOUR_API_KEY_HERE":
        print("\n" + "=" * 60)
        print("  [!] GOOGLE_API_KEY is not set!")
        print("  Edit .env and paste your key on the GOOGLE_API_KEY line")
        print("  Get a key: https://aistudio.google.com/apikey")
        print("=" * 60)
        print("\nStarting server anyway (UI will show the error)...\n")

    print("\n" + "=" * 60)
    print("  EnergyPlus MCP Agent")
    print(f"  Model: {GEMINI_MODEL}")
    print(f"  Open: http://localhost:5000")
    print("=" * 60 + "\n")

    uvicorn.run(
        "agent:app",
        host="0.0.0.0",
        port=5000,
        reload=False,
        log_level="info",
    )
