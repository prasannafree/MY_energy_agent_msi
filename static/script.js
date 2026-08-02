/**
 * EnergyPlus MCP Agent - Chat & Dashboard Logic
 */

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
let isProcessing = false;
let messageHistory = [];
let currentModel = localStorage.getItem("selected_model") || "gemini-3.5-flash";
let delhiChartInstance = null;
let currentSessionId = "session_" + Date.now();

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
  loadFiles();
  loadToolsInventory();
  loadDelhiComparisonData();

  messageInput.addEventListener("input", () => {
    sendBtn.disabled = !messageInput.value.trim() || isProcessing;
  });
  
  messageInput.focus();
});

function showWelcomeScreen() {
  currentSessionId = "session_" + Date.now();
  if (chatArea) {
    chatArea.innerHTML = "";
    if (welcomeScreen) {
      welcomeScreen.style.display = "flex";
      chatArea.appendChild(welcomeScreen);
    }
  }
}

// ---------------------------------------------------------------------------
// Tabs Navigation
// ---------------------------------------------------------------------------
function switchTab(tabId) {
  const tabs = document.querySelectorAll(".tab-btn");
  const contents = document.querySelectorAll(".tab-content");

  tabs.forEach((t) => t.classList.remove("active"));
  contents.forEach((c) => c.classList.remove("active"));

  const targetTab = Array.from(tabs).find((t) =>
    t.getAttribute("onclick").includes(tabId)
  );
  if (targetTab) targetTab.classList.add("active");

  const targetContent = document.getElementById(tabId);
  if (targetContent) targetContent.classList.add("active");

  if (tabId === "tab-delhi" && delhiChartInstance) {
    delhiChartInstance.resize();
  }
}

// ---------------------------------------------------------------------------
// Fetch Models & Health
// ---------------------------------------------------------------------------
async function fetchModels() {
  try {
    const res = await fetch("/api/models");
    const data = await res.json();
    const select = document.getElementById("model-select");
    if (!select || !data.models) return;

    const groups = { google: [], ollama: [] };
    data.models.forEach((m) => {
      if (!groups[m.provider]) groups[m.provider] = [];
      groups[m.provider].push(m);
    });

    let html = "";
    if (groups.google.length > 0) {
      html += `<optgroup label="Google Gemini">`;
      groups.google.forEach(
        (m) => (html += `<option value="${m.id}">${m.name}</option>`)
      );
      html += `</optgroup>`;
    }

    if (groups.ollama && groups.ollama.length > 0) {
      html += `<optgroup label="Local Models (Ollama)">`;
      groups.ollama.forEach(
        (m) => (html += `<option value="${m.id}">${m.name}</option>`)
      );
      html += `</optgroup>`;
    }

    select.innerHTML = html;

    const isModelValid = data.models.some((m) => m.id === currentModel);
    if (!isModelValid && data.models.length > 0) {
      currentModel = data.models[0].id;
      localStorage.setItem("selected_model", currentModel);
    }
    select.value = currentModel;
  } catch (err) {
    console.error("Could not fetch models:", err);
  }
}

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
      <span>⚠️ <strong>Quota Exceeded:</strong> Model <code>${escapeHtml(
        modelName
      )}</code> has exceeded its rate limit or daily quota. Please select another model from the dropdown to continue.</span>
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

setInterval(checkHealth, 5000);

// ---------------------------------------------------------------------------
// Load Files & Tools
// ---------------------------------------------------------------------------
async function loadFiles() {
  try {
    const res = await fetch("/api/files");
    const data = await res.json();

    const idfContainer = document.getElementById("list-idf-files");
    if (idfContainer && data.idf) {
      idfContainer.innerHTML = data.idf
        .map(
          (f) =>
            `<div class="file-item" onclick="useSuggestion('Load and inspect IDF model ${f}')">📄 ${escapeHtml(
              f
            )}</div>`
        )
        .join("");
    }

    const epwContainer = document.getElementById("list-epw-files");
    if (epwContainer && data.epw) {
      epwContainer.innerHTML = data.epw
        .map(
          (f) =>
            `<div class="file-item">🌤️ ${escapeHtml(f)}</div>`
        )
        .join("");
    }

    const ifcContainer = document.getElementById("list-ifc-files");
    const ifcSelect = document.getElementById("ifc-file-select");

    if (ifcContainer && data.ifc) {
      ifcContainer.innerHTML = data.ifc
        .map(
          (f) =>
            `<div class="file-item" onclick="useSuggestion('Inspect IFC model ${f}')">🏗️ ${escapeHtml(
              f
            )}</div>`
        )
        .join("");

      if (ifcSelect) {
        ifcSelect.innerHTML = data.ifc
          .map((f) => `<option value="${f}">${f}</option>`)
          .join("");
      }
    }
  } catch (err) {
    console.error("Could not load workspace files:", err);
  }
}

async function loadToolsInventory() {
  try {
    const res = await fetch("/api/tools");
    const data = await res.json();
    const inventory = document.getElementById("tools-inventory-list");

    if (data.tools && data.tools.length > 0) {
      if (inventory) {
        inventory.innerHTML = data.tools
          .map(
            (t) => `
          <div class="tool-inv-card">
            <div class="tool-inv-name">🔧 ${escapeHtml(t.name)}</div>
            <div class="tool-inv-desc">${escapeHtml(t.description)}</div>
          </div>
        `
          )
          .join("");
      }
    }
  } catch (err) {
    console.error("Could not load tools inventory:", err);
  }
}

// ---------------------------------------------------------------------------
// Load New Delhi Comparison Data & Render Chart
// ---------------------------------------------------------------------------
async function loadDelhiComparisonData() {
  try {
    const res = await fetch("/static/delhi_comparison.json");
    if (!res.ok) return;
    const data = await res.json();

    document.getElementById(
      "val-base-kwh"
    ).innerHTML = `${data.base_case.total_kwh.toLocaleString()} <span class="metric-unit">kWh</span>`;
    document.getElementById("val-base-peak").textContent = `${data.base_case.peak_kw.toLocaleString()} kW`;

    document.getElementById(
      "val-new-kwh"
    ).innerHTML = `${data.new_case.total_kwh.toLocaleString()} <span class="metric-unit">kWh</span>`;
    document.getElementById("val-new-peak").textContent = `${data.new_case.peak_kw.toLocaleString()} kW`;

    document.getElementById(
      "val-diff-kwh"
    ).innerHTML = `+${data.difference.kwh.toLocaleString()} <span class="metric-unit">kWh</span>`;
    document.getElementById(
      "val-diff-pct"
    ).textContent = `+${data.difference.pct}%`;

    document.getElementById(
      "val-peak-diff"
    ).innerHTML = `+${data.difference.peak_diff_kw.toLocaleString()} <span class="metric-unit">kW</span>`;
    document.getElementById(
      "val-peak-pct"
    ).textContent = `+${data.difference.peak_pct}%`;

    renderChart(data.daily);
  } catch (err) {
    console.error("Could not load Delhi comparison JSON:", err);
  }
}

function renderChart(dailyData) {
  const ctx = document.getElementById("delhiChart");
  if (!ctx) return;

  if (delhiChartInstance) {
    delhiChartInstance.destroy();
  }

  delhiChartInstance = new Chart(ctx, {
    type: "line",
    data: {
      labels: dailyData.labels,
      datasets: [
        {
          label: "Base Case (kWh)",
          data: dailyData.base_kwh,
          borderColor: "#3b82f6",
          backgroundColor: "rgba(59, 130, 246, 0.1)",
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
        },
        {
          label: "+50% Occupancy Case (kWh)",
          data: dailyData.new_kwh,
          borderColor: "#f59e0b",
          backgroundColor: "rgba(245, 158, 11, 0.1)",
          fill: true,
          tension: 0.3,
          borderWidth: 2,
          pointRadius: 3,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: "index",
          intersect: false,
          backgroundColor: "#1e293b",
          titleColor: "#f8fafc",
          bodyColor: "#cbd5e1",
          borderColor: "#334155",
          borderWidth: 1,
        },
      },
      scales: {
        x: {
          grid: { color: "rgba(148, 163, 184, 0.1)" },
          ticks: { color: "#94a3b8", font: { size: 10 } },
        },
        y: {
          grid: { color: "rgba(148, 163, 184, 0.1)" },
          ticks: {
            color: "#94a3b8",
            font: { size: 11 },
            callback: (val) => `${val.toLocaleString()} kWh`,
          },
        },
      },
    },
  });
}

// ---------------------------------------------------------------------------
// 3D IFC Visualizer Trigger
// ---------------------------------------------------------------------------
async function generateAndLoadIFC() {
  const select = document.getElementById("ifc-file-select");
  const filename = select ? select.value : "";
  const placeholder = document.getElementById("ifc-placeholder");
  const iframe = document.getElementById("ifc-iframe");
  const visBtn = document.getElementById("vis-btn");

  if (!filename) return;

  visBtn.disabled = true;
  visBtn.textContent = "⏳ Generating 3D Model Visualizer...";
  placeholder.style.display = "flex";
  placeholder.innerHTML = `
    <div class="thinking-dots"><span></span><span></span><span></span></div>
    <p style="margin-top: 12px; color: var(--text-secondary);">Parsing IFC geometry & building 3D Plotly scene...</p>
  `;
  iframe.style.display = "none";

  try {
    const res = await fetch("/api/visualize_ifc", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });

    const data = await res.json();
    if (res.ok && data.html_url) {
      iframe.src = data.html_url + "?t=" + Date.now();
      iframe.onload = () => {
        placeholder.style.display = "none";
        iframe.style.display = "block";
      };
    } else {
      placeholder.innerHTML = `⚠️ <p style="color: var(--accent-red); margin-top: 8px;">Error generating 3D view: ${
        data.error || "Unknown error"
      }</p>`;
    }
  } catch (err) {
    placeholder.innerHTML = `⚠️ <p style="color: var(--accent-red); margin-top: 8px;">Error: ${err.message}</p>`;
  }

  visBtn.disabled = false;
  visBtn.textContent = "✨ Generate 3D Interactive Visualization";
}

// ---------------------------------------------------------------------------
// Send Message (LLM Chat)
// ---------------------------------------------------------------------------
async function sendMessage() {
  const text = messageInput.value.trim();
  if (!text || isProcessing) return;

  if (welcomeScreen) {
    welcomeScreen.style.display = "none";
  }

  appendMessage("user", text);
  messageInput.value = "";
  messageInput.style.height = "auto";
  sendBtn.disabled = true;
  isProcessing = true;

  const thinkingEl = showThinking();

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, model: currentModel, session_id: currentSessionId }),
    });

    const data = await res.json();
    thinkingEl.remove();

    if (res.status === 429) {
      showQuotaBanner(data.model || currentModel);
      appendMessage(
        "assistant",
        `⚠️ **Quota Exceeded:** The request failed because the rate limit or daily quota for \`${
          data.model || currentModel
        }\` was exceeded. You can switch to another model using the dropdown at the top to continue.`,
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
          `<span class="tool-badge" title="${escapeHtml(
            JSON.stringify(t.args || {}, null, 2)
          )}">
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

function showThinking() {
  const el = document.createElement("div");
  el.className = "thinking-indicator";
  el.innerHTML = `
    <div class="message-avatar" style="background: linear-gradient(135deg, #10b981, #059669); box-shadow: 0 0 30px rgba(139,92,246,0.15);">⚡</div>
    <div class="thinking-dots">
      <span></span><span></span><span></span>
    </div>
    <span class="thinking-label">Thinking & executing tools...</span>
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

toolsModal.addEventListener("click", (e) => {
  if (e.target === toolsModal) {
    toolsModal.classList.remove("active");
  }
});

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
          ${
            data.error
              ? "⚠️ " + escapeHtml(data.error)
              : "No tools available. MCP server may not be connected."
          }
        </div>
      `;
    }
  } catch {
    toolsList.innerHTML =
      '<div class="tools-loading">⚠️ Could not load tools</div>';
  }
}

// ---------------------------------------------------------------------------
// Suggestions / Quick Prompt Cards
// ---------------------------------------------------------------------------
function useSuggestion(text) {
  messageInput.value = text;
  sendBtn.disabled = false;
  messageInput.focus();
  sendMessage();
}

function handleKeyDown(e) {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!sendBtn.disabled) {
      sendMessage();
    }
  }
}

function autoResize(el) {
  el.style.height = "auto";
  el.style.height = Math.min(el.scrollHeight, 150) + "px";
}

function scrollToBottom() {
  requestAnimationFrame(() => {
    chatArea.scrollTop = chatArea.scrollHeight;
  });
}

function renderMarkdown(text) {
  if (!text) return "";

  let html = escapeHtml(text);

  html = html.replace(
    /```(\w*)\n([\s\S]*?)```/g,
    (_, lang, code) => `<pre><code>${code.trim()}</code></pre>`
  );

  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/\*(.+?)\*/g, "<em>$1</em>");
  html = html.replace(/^### (.+)$/gm, "<strong>$1</strong>");
  html = html.replace(/^## (.+)$/gm, "<strong>$1</strong>");

  html = html.replace(/^[\-\*] (.+)$/gm, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>\n?)+/g, (match) => `<ul>${match}</ul>`);
  html = html.replace(/^\d+\. (.+)$/gm, "<li>$1</li>");

  html = html.replace(
    /\[([^\]]+)\]\(([^)]+)\)/g,
    '<a href="$2" target="_blank" rel="noopener" style="color: var(--accent-blue);">$1</a>'
  );

  html = html
    .split(/\n\n+/)
    .map((p) => `<p>${p.replace(/\n/g, "<br>")}</p>`)
    .join("");

  return html;
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

function truncate(str, len) {
  if (!str) return "";
  return str.length > len ? str.slice(0, len) + "…" : str;
}
