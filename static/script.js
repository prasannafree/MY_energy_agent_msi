/**
 * EnergyPlus MCP Agent - Chat UI Logic
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isProcessing = false;
let messageHistory = [];
let currentModel = localStorage.getItem("selected_model") || "gemini-3.5-flash";

const chatArea = document.getElementById("chat-area");
const welcomeScreen = document.getElementById("welcome-screen");
const messageInput = document.getElementById("message-input");
const sendBtn = document.getElementById("send-btn");
const statusDot = document.getElementById("status-dot");
const statusText = document.getElementById("status-text");
const toolsModal = document.getElementById("tools-modal");
const toolsList = document.getElementById("tools-list");

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------
document.addEventListener("DOMContentLoaded", () => {
  fetchModels();
  
  checkHealth();
  messageInput.addEventListener("input", () => {
    sendBtn.disabled = !messageInput.value.trim() || isProcessing;
  });
  // Focus input on load
  messageInput.focus();
});

async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const select = document.getElementById("model-select");
    if (!select || !data.models) return;
    
    const groups = { google: [], ollama: [] };
    data.models.forEach(m => {
      if (!groups[m.provider]) groups[m.provider] = [];
      groups[m.provider].push(m);
    });
    
    let html = "";
    if (groups.google.length > 0) {
      html += `<optgroup label="Google Gemini">`;
      groups.google.forEach(m => html += `<option value="${m.id}">${m.name}</option>`);
      html += `</optgroup>`;
    }
    
    if (groups.ollama && groups.ollama.length > 0) {
      html += `<optgroup label="Local Models (Ollama)">`;
      groups.ollama.forEach(m => html += `<option value="${m.id}">${m.name}</option>`);
      html += `</optgroup>`;
    }
    
    select.innerHTML = html;
    
    const isModelValid = data.models.some(m => m.id === currentModel);
    if (!isModelValid && data.models.length > 0) {
      currentModel = data.models[0].id;
      localStorage.setItem("selected_model", currentModel);
    }
    select.value = currentModel;
    
  } catch (err) {
    console.error("Could not fetch models:", err);
  }
}

// ---------------------------------------------------------------------------
// Model Switching Logic
// ---------------------------------------------------------------------------
function changeModel() {
  const modelSelect = document.getElementById("model-select");
  if (modelSelect) {
    currentModel = modelSelect.value;
    localStorage.setItem("selected_model", currentModel);
    dismissQuotaBanner();
    checkHealth();
  }
}

function showQuotaBanner(modelName) {
  const container = document.getElementById("alert-banner-container");
  if (!container) return;
  container.innerHTML = `
    <div class="alert-banner" id="quota-alert">
      <span>⚠️ <strong>Quota Exceeded:</strong> Model <code>${escapeHtml(modelName)}</code> has exceeded its rate limit or daily quota. Please select another model from the dropdown to continue.</span>
      <button class="alert-banner-close" onclick="dismissQuotaBanner()">✕</button>
    </div>
  `;
}

function dismissQuotaBanner() {
  const container = document.getElementById("alert-banner-container");
  if (container) {
    container.innerHTML = "";
  }
}

// ---------------------------------------------------------------------------
// Health Check
// ---------------------------------------------------------------------------
async function checkHealth() {
  try {
    const res = await fetch("/api/health");
    const data = await res.json();

    if (data.agent_ready) {
      statusDot.classList.remove("disconnected");
      statusDot.style.background = "";
      statusDot.style.boxShadow = "";
      statusText.textContent = currentModel;
    } else if (!data.api_key_set) {
      statusDot.classList.add("disconnected");
      statusDot.style.background = "";
      statusDot.style.boxShadow = "";
      statusText.textContent = "API Key Missing";
    } else if (data.init_status === "connecting") {
      statusDot.classList.remove("disconnected");
      statusDot.style.background = "#f59e0b";
      statusDot.style.boxShadow = "0 0 8px rgba(245,158,11,0.4)";
      statusText.textContent = "Connecting to MCP...";
    } else {
      statusDot.classList.add("disconnected");
      statusDot.style.background = "";
      statusDot.style.boxShadow = "";
      statusText.textContent = "MCP Error";
    }
  } catch {
    statusDot.classList.add("disconnected");
    statusDot.style.background = "";
    statusDot.style.boxShadow = "";
    statusText.textContent = "Offline";
  }
}

// Poll health every 5 seconds
setInterval(checkHealth, 5000);

// ---------------------------------------------------------------------------
// Send Message
// ---------------------------------------------------------------------------
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isProcessing) return;

  // Hide welcome screen
  if (welcomeScreen) {
    welcomeScreen.style.display = "none";
  }

  // Add user message
  appendMessage("user", text);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendBtn.disabled = true;
  isProcessing = true;

  // Show thinking indicator
  const thinkingEl = showThinking();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, model: currentModel }),
    });

    const data = await res.json();

    // Remove thinking
    thinkingEl.remove();

    if (res.status === 429) {
      showQuotaBanner(data.model || currentModel);
      appendMessage(
        "assistant",
        `⚠️ **Quota Exceeded:** The request failed because the rate limit or daily quota for \`${data.model || currentModel}\` was exceeded. You can switch to another model using the dropdown at the top to continue.`,
        []
      );
    } else if (res.ok) {
      appendMessage("assistant", data.response, data.tools_used || []);
    } else {
      appendMessage(
        "assistant",
        `⚠️ **Error:** ${data.error || "Something went wrong"}`,
        []
      );
    }
  } catch (err) {
    thinkingEl.remove();
    appendMessage(
      "assistant",
      `⚠️ **Connection Error:** Could not reach the server. Make sure it's running on localhost:5000.`,
      []
    );
  }

  isProcessing = false;
  sendBtn.disabled = !messageInput.value.trim();
  messageInput.focus();
}

// ---------------------------------------------------------------------------
// Render Messages
// ---------------------------------------------------------------------------
function appendMessage(role, content, toolsUsed = []) {
  const msg = document.createElement("div");
  msg.className = `message ${role}`;

  const avatar = role === "user" ? "👤" : "⚡";
  const name = role === "user" ? "You" : "EnergyPlus Agent";
  const time = new Date().toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });

  let toolBadgesHtml = "";
  if (toolsUsed.length > 0) {
    const badges = toolsUsed
      .map(
        (t) =>
          `<span class="tool-badge" title="${escapeHtml(JSON.stringify(t.args || {}, null, 2))}">
            <span class="tool-icon">🔧</span>${escapeHtml(t.name)}
          </span>`
      )
      .join("");
    toolBadgesHtml = `<div class="tools-container">${badges}</div>`;
  }

  msg.innerHTML = `
    <div class="message-avatar">${avatar}</div>
    <div class="message-content">
      <div class="message-header">
        <span class="message-name">${name}</span>
        <span class="message-time">${time}</span>
      </div>
      <div class="message-body">${renderMarkdown(content)}</div>
      ${toolBadgesHtml}
    </div>
  `;

  chatArea.appendChild(msg);
  scrollToBottom();
}

// ---------------------------------------------------------------------------
// Thinking Indicator
// ---------------------------------------------------------------------------
function showThinking() {
  const el = document.createElement("div");
  el.className = "thinking-indicator";
  el.innerHTML = `
    <div class="message-avatar" style="background: linear-gradient(135deg, #10b981, #059669); box-shadow: 0 0 30px rgba(139,92,246,0.15);">⚡</div>
    <div class="thinking-dots">
      <span></span><span></span><span></span>
    </div>
    <span class="thinking-label">Thinking & using tools...</span>
  `;
  chatArea.appendChild(el);
  scrollToBottom();
  return el;
}

// ---------------------------------------------------------------------------
// Tools Modal
// ---------------------------------------------------------------------------
function toggleToolsModal() {
  const isActive = toolsModal.classList.contains("active");
  if (isActive) {
    toolsModal.classList.remove("active");
  } else {
    toolsModal.classList.add("active");
    loadTools();
  }
}

// Close modal on overlay click
toolsModal.addEventListener("click", (e) => {
  if (e.target === toolsModal) {
    toolsModal.classList.remove("active");
  }
});

// Close modal on Escape
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape" && toolsModal.classList.contains("active")) {
    toolsModal.classList.remove("active");
  }
});

async function loadTools() {
  toolsList.innerHTML = '<div class="tools-loading">Loading tools...</div>';

  try {
    const res = await fetch("/api/tools");
    const data = await res.json();

    if (data.tools && data.tools.length > 0) {
      toolsList.innerHTML = data.tools
        .map(
          (t) => `
          <div class="tool-item">
            <div class="tool-name">🔧 ${escapeHtml(t.name)}</div>
            <div class="tool-desc">${escapeHtml(truncate(t.description, 150))}</div>
          </div>
        `
        )
        .join("");
    } else {
      toolsList.innerHTML = `
        <div class="tools-loading">
          ${data.error ? "⚠️ " + escapeHtml(data.error) : "No tools available. MCP server may not be connected."}
        </div>
      `;
    }
  } catch {
    toolsList.innerHTML =
      '<div class="tools-loading">⚠️ Could not load tools</div>';
  }
}

// ---------------------------------------------------------------------------
// Suggestions
// ---------------------------------------------------------------------------
function useSuggestion(text) {
  messageInput.value = text;
  sendBtn.disabled = false;
  messageInput.focus();
  sendMessage();
}

// ---------------------------------------------------------------------------
// Keyboard Handling
// ---------------------------------------------------------------------------
function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) {
      sendMessage();
    }
  }
}

// ---------------------------------------------------------------------------
// Textarea Auto-Resize
// ---------------------------------------------------------------------------
function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

// ---------------------------------------------------------------------------
// Scroll
// ---------------------------------------------------------------------------
function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

// ---------------------------------------------------------------------------
// Markdown Renderer (lightweight)
// ---------------------------------------------------------------------------
function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(text);

  // Code blocks (```)
  html = html.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_, lang, code) => `<pre><code>${code.trim()}</code></pre>`
  );

  // Inline code
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");

  // Bold
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

  // Italic
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");

  // Headers (## → h3, ### → h4)
  html = html.replace(/^### (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^## (.+)$/gm, "<strong>$1</strong>");

  // Unordered lists
  html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);

  // Ordered lists
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  // Links
  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener" style="color: var(--accent-blue);">$1</a>'
  );

  // Line breaks → paragraphs
  html = html
    .split(/\n\n+/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");

  return html;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------
function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, len) {
  if (!str) return "";
  return str.length > len ? str.slice(0, len) + "…" : str;
}
