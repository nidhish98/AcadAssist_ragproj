const API_BASE = "http://localhost:8000";

const chat   = document.getElementById("chat");
const form   = document.getElementById("form");
const input  = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const badge  = document.getElementById("status-badge");

let isAsking = false;

// ── Check API health on load ──────────────────────────────────────

async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    badge.textContent = `${data.total_chunks} chunks · ${data.llm_backend}`;
    badge.className = "ok";
  } catch {
    badge.textContent = "offline";
    badge.className = "error";
  }
}

checkStatus();

// ── Send query ────────────────────────────────────────────────────

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || isAsking) return;

  appendMessage("user", question);
  input.value = "";
  isAsking = true;
  sendBtn.disabled = true;

  const typingId = showTyping();

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.text();
      appendMessage("bot", `Error: ${err}`);
      return;
    }

    const data = await res.json();
    replaceTyping(typingId, formatBotMessage(data));
  } catch (err) {
    replaceTyping(typingId, formatBotMessage({
      answer: `Network error — is the API running at ${API_BASE}?`,
      sources: [],
      backend: "-",
      model: "-",
      response_time_ms: 0,
    }));
  } finally {
    isAsking = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

// ── Helpers ───────────────────────────────────────────────────────

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === "user" ? "U" : "R"}</div>
    <div class="bubble"><p>${escapeHtml(text)}</p></div>
  `;
  chat.appendChild(div);
  scrollBottom();
}

function showTyping() {
  const id = "typing-" + Date.now();
  const div = document.createElement("div");
  div.id = id;
  div.className = "message bot typing";
  div.innerHTML = `
    <div class="avatar">R</div>
    <div class="bubble">
      <span class="dot"></span>
      <span class="dot"></span>
      <span class="dot"></span>
    </div>
  `;
  chat.appendChild(div);
  scrollBottom();
  return id;
}

function replaceTyping(id, html) {
  const el = document.getElementById(id);
  if (el) {
    el.className = "message bot";
    el.querySelector(".bubble").innerHTML = html;
    el.classList.remove("typing");
  }
  scrollBottom();
}

function formatBotMessage(data) {
  let html = `<p>${escapeHtml(data.answer)}</p>`;

  if (data.sources && data.sources.length > 0) {
    html += `<div class="sources">
      <details>
        <summary>${data.sources.length} source${data.sources.length > 1 ? "s" : ""}</summary>
        <ol>`;
    for (const s of data.sources) {
      html += `<li>${escapeHtml(s.source)} pp.${s.page_start}–${s.page_end} &middot; score ${s.score.toFixed(3)}</li>`;
    }
    html += `</ol></details></div>`;
  }

  html += `<div class="meta">${data.backend} · ${data.model} · ${data.response_time_ms}ms</div>`;
  return html;
}

function scrollBottom() {
  chat.scrollTop = chat.scrollHeight;
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}
