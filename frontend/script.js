const chat = document.getElementById("chat");
const form = document.getElementById("form");
const input = document.getElementById("input");
const sendBtn = document.getElementById("send-btn");
const badge = document.getElementById("status-badge");

let isAsking = false;

/* ── API base (relative in dev, absolute in prod) ── */

const API_BASE = window.location.origin.startsWith("http://localhost")
  ? "http://localhost:8000"
  : window.location.origin;

/* ── Theme ── */

const themeToggle = document.getElementById("theme-toggle");
const stored = localStorage.getItem("raka-theme") || "light";
document.documentElement.setAttribute("data-theme", stored);

themeToggle.addEventListener("click", () => {
  const cur = document.documentElement.getAttribute("data-theme");
  const next = cur === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("raka-theme", next);
});

/* ── Auto-resize textarea ── */

input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 150) + "px";
  sendBtn.disabled = !input.value.trim();
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    form.dispatchEvent(new Event("submit"));
  }
});

/* ── Status check ── */

async function checkStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(res.statusText);
    const data = await res.json();
    badge.textContent = `${data.total_chunks} chunks · ${data.llm_backend}`;
    badge.className = "badge online";
  } catch {
    badge.textContent = "Offline";
    badge.className = "badge offline";
  }
}

checkStatus();

/* ── Suggestion clicks ── */

chat.addEventListener("click", (e) => {
  const btn = e.target.closest(".suggestion");
  if (btn) {
    input.value = btn.dataset.q;
    sendBtn.disabled = false;
    form.dispatchEvent(new Event("submit"));
  }
});

/* ── Submit ── */

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  const question = input.value.trim();
  if (!question || isAsking) return;

  appendMessage("user", question);
  input.value = "";
  input.style.height = "auto";
  sendBtn.disabled = true;
  isAsking = true;

  const typingId = showTyping();

  try {
    const res = await fetch(`${API_BASE}/api/query`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });

    if (!res.ok) {
      const err = await res.text();
      replaceTyping(typingId, formatError(err));
      return;
    }

    const data = await res.json();
    replaceTyping(typingId, formatBotMessage(data));
  } catch (err) {
    replaceTyping(typingId, formatError(`Cannot reach the API at ${API_BASE}. Is the server running?`));
  } finally {
    isAsking = false;
    sendBtn.disabled = false;
    input.focus();
  }
});

/* ── Helpers ── */

function appendMessage(role, text) {
  const div = document.createElement("div");
  div.className = `message ${role}`;
  div.innerHTML = `
    <div class="avatar">${role === "user" ? "U" : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="2" width="20" height="20" rx="4"/><path d="M9 12h6M12 9v6"/></svg>`}</div>
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
    <div class="avatar">
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><rect x="2" y="2" width="20" height="20" rx="4"/><path d="M9 12h6M12 9v6"/></svg>
    </div>
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
  if (!el) return;
  el.className = "message bot";
  el.querySelector(".bubble").innerHTML = html;
  el.classList.remove("typing");
  scrollBottom();
}

function formatBotMessage(data) {
  const answer = escapeHtml(data.answer).replace(/\n/g, "<br>");
  let html = `<p>${answer}</p>`;

  if (data.sources && data.sources.length > 0) {
    html += `<div class="sources">`;
    html += `<button class="source-toggle" onclick="toggleSources(this)"><span class="arrow">▶</span> ${data.sources.length} source${data.sources.length > 1 ? "s" : ""}</button>`;
    html += `<div class="source-list">`;
    for (const s of data.sources) {
      html += `<div class="source-item">
        <svg class="source-icon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        ${escapeHtml(s.source)} <span class="source-score">p.${s.page_start}–${s.page_end} · ${s.score.toFixed(3)}</span>
      </div>`;
    }
    html += `</div></div>`;
  }

  html += `<div class="meta">
    <span class="meta-tag">⚡ ${escapeHtml(data.backend)}</span>
    <span class="meta-tag">🧠 ${escapeHtml(data.model)}</span>
    <span class="meta-tag">⏱ ${data.response_time_ms}ms</span>
    <button class="copy-btn" onclick="copyAnswer(this)">Copy</button>
  </div>`;

  return html;
}

function formatError(msg) {
  return `<p>${escapeHtml(msg)}</p>`;
}

/* ── Toggle Sources ── */

function toggleSources(btn) {
  btn.classList.toggle("open");
  btn.nextElementSibling.classList.toggle("open");
}

/* ── Copy Answer ── */

function copyAnswer(btn) {
  const bubble = btn.closest(".bubble");
  const text = bubble.querySelector("p")?.textContent || "";
  navigator.clipboard.writeText(text).then(() => {
    btn.textContent = "Copied!";
    setTimeout(() => { btn.textContent = "Copy"; }, 2000);
  });
}

function scrollBottom() {
  requestAnimationFrame(() => {
    chat.scrollTop = chat.scrollHeight;
  });
}

function escapeHtml(text) {
  const d = document.createElement("div");
  d.textContent = text;
  return d.innerHTML;
}
