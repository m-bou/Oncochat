// OncoChat UI — vanilla JS, no build step. Talks to the FastAPI backend (SSE over fetch for streaming).
"use strict";

const $ = (sel) => document.querySelector(sel);
const state = { conversationId: null, busy: false };

const els = {
  app: $("#app"), main: document.querySelector(".main"), thread: $("#thread"), input: $("#input"),
  composer: $("#composer"), send: $("#send"), conversations: $("#conversations"), drawer: $("#drawer"),
  drawerTitle: $("#drawerTitle"), drawerBody: $("#drawerBody"), health: $("#health"),
};

// ------------------------------------------------------------------ helpers
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => (
  { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

function markdown(src) {
  // Minimal, safe subset: paragraphs, "- " lists, **bold**, *italic*, `code`.
  const inline = (t) => esc(t)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*(?!\s)(.+?)\*(?!\*)/g, "$1<em>$2</em>")
    .replace(/`([^`]+)`/g, "<code>$1</code>");
  const out = [];
  let list = null;
  for (const line of String(src || "").split("\n")) {
    const m = line.match(/^\s*(?:[-*•]|\d+\.)\s+(.*)/);
    if (m) { (list ??= []).push(`<li>${inline(m[1])}</li>`); continue; }
    if (list) { out.push(`<ul>${list.join("")}</ul>`); list = null; }
    if (line.trim()) out.push(`<p>${inline(line)}</p>`);
  }
  if (list) out.push(`<ul>${list.join("")}</ul>`);
  return out.join("");
}

function fmtMs(ms) { return ms == null ? "" : ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`; }

function relTime(ts) {
  const d = (Date.now() / 1000 - ts);
  if (d < 60) return "just now";
  if (d < 3600) return `${Math.floor(d / 60)} min ago`;
  if (d < 86400) return `${Math.floor(d / 3600)} h ago`;
  return new Date(ts * 1000).toLocaleDateString();
}

const fmtArgs = (args) => Object.values(args || {}).map((v) => Array.isArray(v) ? v.join(", ") : v).join(", ");

// Human-readable progress labels for graph nodes (shown while the answer is being built).
function stepLabel(ev) {
  const d = ev.detail || {};
  switch (ev.node) {
    case "cache_lookup": return d.hit ? "Found in cache" : null;
    case "router": return d.intent === "help" || d.intent === "greeting" ? "Help request" : "Read the question";
    case "select_model": return `${d.tier === "smart" ? "Smart" : "Fast"} model (${d.model})`;
    case "agent":
      if (d.error) return { text: "Model unavailable", fail: true };
      return d.tool_calls?.length ? "Chose a lookup" : "Wrote the answer";
    case "tools": return (Array.isArray(d) ? d : []).map((c) => ({
      text: `${c.name}(${fmtArgs(c.args)})${c.error ? ` — ${c.error.replace("_", " ")}` : ""}`, fail: !!c.error && c.error !== "unknown_cancer",
    }));
    case "guard": return d.ok ? "Checked against the data" : { text: "Answer failed the data check", fail: true };
    case "escalate": return { text: "Retrying with the smart model", fail: true };
    case "fallback": return { text: "No reliable answer", fail: true };
    default: return null;
  }
}

// ------------------------------------------------------------------ rendering
function renderTable(t) {
  const box = document.createElement("div");
  box.className = "datatable";
  let body;
  if (t.columns.length === 1) {
    body = `<div class="genes">${t.rows.map((r) => `<span>${esc(r[0])}</span>`).join("")}</div>`;
  } else {
    const symIsGene = t.columns[0] === "gene";
    body = `<table>${t.rows.map(([k, v]) => `<tr>
      <td class="sym"${symIsGene ? "" : ' style="font-family:var(--sans)"'}>${esc(k)}</td>
      <td class="bar"><div class="band" role="img" aria-label="${esc(k)} ${Number(v)}"><span style="width:${Math.max(0, Math.min(1, v)) * 100}%"></span></div></td>
      <td class="num">${Number(v).toFixed(3)}</td></tr>`).join("")}</table>`;
  }
  box.innerHTML = `<h3>${esc(t.title)}</h3>${body}${t.footer ? `<div class="foot">${esc(t.footer)}</div>` : ""}`;
  return box;
}

function addUser(text) {
  const el = document.createElement("div");
  el.className = "msg user";
  el.innerHTML = `<div class="bubble">${esc(text)}</div>`;
  els.thread.appendChild(el);
}

function addAssistant() {
  const el = document.createElement("div");
  el.className = "msg assistant";
  el.innerHTML = `<div class="progress"></div><div class="text cursor"></div><div class="tables"></div><div class="meta"></div>`;
  els.thread.appendChild(el);
  return {
    el, draft: "",
    progress: el.querySelector(".progress"), text: el.querySelector(".text"),
    tables: el.querySelector(".tables"), meta: el.querySelector(".meta"),
  };
}

function pushStep(view, ev) {
  let labels = stepLabel(ev);
  if (!labels) return;
  labels = (Array.isArray(labels) ? labels : [labels]).map((l) => typeof l === "string" ? { text: l } : l);
  view.progress.querySelectorAll(".live").forEach((s) => s.classList.remove("live"));
  for (const l of labels) {
    const chip = document.createElement("span");
    chip.className = `step${l.fail ? " fail" : ""}`;
    chip.textContent = l.text;
    view.progress.appendChild(chip);
  }
  view.progress.lastElementChild?.classList.add("live");
}

function finish(view, f) {
  view.text.classList.remove("cursor");
  view.text.innerHTML = markdown(f.answer);
  view.tables.replaceChildren(...(f.data || []).map(renderTable));
  view.progress.querySelectorAll(".live").forEach((s) => s.classList.remove("live"));
  const tier = f.tier ? `<span class="tier ${f.tier}">${f.tier}</span><span>${esc(f.model || "")}</span>`
    : `<span class="tier none">no model needed</span>`;
  const flags = [f.cache_hit && "from cache", f.escalated && "escalated", f.status === "fallback" && "not answered"]
    .filter(Boolean).map((x) => `<span class="${x === "not answered" ? "notice" : ""}">${x}</span>`).join("");
  view.meta.innerHTML = `${tier}<span>${fmtMs(f.latency_ms)}</span>${flags}
    <button class="linkish" type="button" data-trace="${esc(f.request_id)}">View trace</button>`;
  if (f.status === "help" || f.cache_hit) view.progress.replaceChildren();
}

function renderHistoricTurn(t) {
  addUser(t.question);
  const view = addAssistant();
  finish(view, { ...t, data: t.data, cache_hit: !!t.cache_hit, escalated: !!t.escalated, request_id: t.id });
}

const scrollDown = () => { els.thread.scrollTop = els.thread.scrollHeight; };

function setChatting(on) {
  els.main.classList.toggle("chatting", on);
  if (!on) els.thread.replaceChildren();
}

// ------------------------------------------------------------------ chat
async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  els.send.disabled = true;
  setChatting(true);
  addUser(question);
  const view = addAssistant();
  scrollDown();

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, conversation_id: state.conversationId }),
    });
    if (!res.ok) throw new Error(`Server returned ${res.status}`);
    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buf.indexOf("\n\n")) >= 0) {
        const block = buf.slice(0, idx);
        buf = buf.slice(idx + 2);
        const event = block.match(/^event: (.*)$/m)?.[1];
        const data = JSON.parse(block.match(/^data: (.*)$/m)?.[1] || "{}");
        handleEvent(view, event, data);
      }
      scrollDown();
    }
  } catch (err) {
    view.text.classList.remove("cursor");
    view.text.innerHTML = `<p class="notice">The request failed: ${esc(err.message)}. Check that the server is running and try again.</p>`;
  } finally {
    state.busy = false;
    els.send.disabled = false;
    loadConversations();
    els.input.focus();
  }
}

function handleEvent(view, event, data) {
  if (event === "start") { state.conversationId = data.conversation_id; history.replaceState(null, "", `#c=${data.conversation_id}`); }
  else if (event === "step") pushStep(view, data);
  else if (event === "token") { view.draft += data.text; view.text.innerHTML = markdown(view.draft); }
  else if (event === "reset") { view.draft = ""; view.text.innerHTML = ""; }
  else if (event === "final") finish(view, data);
  else if (event === "error") {
    view.text.classList.remove("cursor");
    view.text.innerHTML = `<p class="notice">${esc(data.message)}</p>`;
  }
}

// ------------------------------------------------------------------ sidebar & drawer
async function loadConversations() {
  try {
    const convs = await (await fetch("/api/conversations")).json();
    els.conversations.replaceChildren(...convs.map((c) => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = `conv${c.id === state.conversationId ? " active" : ""}`;
      b.innerHTML = `<span class="conv-title">${esc(c.title)}</span>
        <span class="conv-meta">${c.n_messages} ${c.n_messages === 1 ? "question" : "questions"}, ${relTime(c.updated_at || c.created_at)}</span>`;
      b.onclick = () => openConversation(c.id);
      return b;
    }));
  } catch { /* sidebar is optional */ }
}

async function openConversation(id) {
  state.conversationId = id;
  history.replaceState(null, "", `#c=${id}`);
  const turns = await (await fetch(`/api/conversations/${id}`)).json();
  setChatting(false);
  setChatting(true);
  turns.forEach(renderHistoricTurn);
  scrollDown();
  loadConversations();
  closeSidebarOnMobile();
}

function newChat() {
  state.conversationId = null;
  history.replaceState(null, "", location.pathname);
  setChatting(false);
  loadConversations();
  els.input.focus();
  closeSidebarOnMobile();
}

function openDrawer(title, html) {
  els.drawerTitle.textContent = title;
  els.drawerBody.innerHTML = html;
  els.drawer.classList.add("open");
  els.drawer.setAttribute("aria-hidden", "false");
}

function closeDrawer() {
  els.drawer.classList.remove("open");
  els.drawer.setAttribute("aria-hidden", "true");
}

async function showTrace(id) {
  const t = await (await fetch(`/api/traces/${id}`)).json();
  const kv = [
    ["Question", t.question], ["Status", t.status], ["Intent", `${t.intent ?? "-"} (${t.lang ?? "-"})`],
    ["Model", t.tier ? `${t.tier} (${t.model})` : "none"], ["Escalated", t.escalated ? "yes" : "no"],
    ["From cache", t.cache_hit ? "yes" : "no"], ["Complexity", t.complexity_score ?? "-"],
    ["Total time", fmtMs(t.latency_ms)], ["Tokens", t.prompt_tokens != null ? `${t.prompt_tokens} in, ${t.completion_tokens} out` : "-"],
    ["Guard", t.guard ? (t.guard.ok ? "passed" : t.guard.reasons.join("; ")) : "-"],
    ["Data version", t.data_version], ["Prompt version", t.prompt_version],
  ];
  const steps = (t.steps || []).map((s) => `<div class="trace-step"><span class="name">${esc(s.node)}</span>
    <span class="ms">${fmtMs(s.latency_ms)}</span>${s.detail ? `<pre>${esc(JSON.stringify(s.detail, null, 2))}</pre>` : ""}</div>`).join("");
  openDrawer("Trace", `<dl class="kv">${kv.map(([k, v]) => `<dt>${k}</dt><dd>${esc(v)}</dd>`).join("")}</dl>${steps}`);
}

async function showActivity() {
  const { stats, requests } = await (await fetch("/api/history?limit=100")).json();
  const n = stats.n || 0;
  const cards = [
    ["Questions", n], ["Avg time", fmtMs(stats.avg_ms)], ["From cache", stats.cache_hits || 0],
    ["Fast / smart", `${stats.fast || 0} / ${stats.smart || 0}`], ["Escalations", stats.escalations || 0],
    ["Not answered", (stats.fallbacks || 0) + (stats.errors || 0)],
  ];
  const rows = requests.map((r) => `<div class="activity-row" data-trace="${esc(r.id)}">
      <div class="q">${esc(r.question)}</div>
      <div class="s">${esc(r.status)}, ${r.tier ? esc(r.tier) + " model" : "no model"}, ${fmtMs(r.latency_ms)}, ${relTime(r.created_at)}</div>
    </div>`).join("");
  openDrawer("Activity", `<div class="stats">${cards.map(([k, v]) => `<div><b>${esc(v)}</b><span>${k}</span></div>`).join("")}</div>
    ${rows || "<p>No questions yet. Ask one to see it here.</p>"}`);
}

async function checkHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    const ok = h.ollama === "ok";
    els.health.className = `health ${ok ? "ok" : "bad"}`;
    els.health.textContent = ok ? `${h.models.fast} / ${h.models.smart}` : `Models: ${h.ollama}`;
  } catch {
    els.health.className = "health bad";
    els.health.textContent = "Server unreachable";
  }
}

function closeSidebarOnMobile() { els.app.classList.remove("sidebar-open"); }

// ------------------------------------------------------------------ wiring
function autoGrow() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 220)}px`;
}

els.composer.addEventListener("submit", (e) => {
  e.preventDefault();
  const q = els.input.value;
  els.input.value = "";
  autoGrow();
  ask(q);
});
els.input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); els.composer.requestSubmit(); }
});
els.input.addEventListener("input", autoGrow);
document.querySelectorAll("#suggestions button").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));
$("#newChat").onclick = newChat;
$("#openActivity").onclick = showActivity;
$("#closeDrawer").onclick = closeDrawer;
document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeDrawer(); });
document.addEventListener("click", (e) => {
  const id = e.target.closest("[data-trace]")?.dataset.trace;
  if (id) showTrace(id);
});
$("#toggleSidebar").onclick = () => {
  const mobile = window.matchMedia("(max-width: 760px)").matches;
  els.app.classList.toggle(mobile ? "sidebar-open" : "sidebar-hidden");
};

const themeBtn = $("#toggleTheme");
function applyTheme(theme) {
  if (theme) document.documentElement.dataset.theme = theme; else delete document.documentElement.dataset.theme;
  const dark = theme ? theme === "dark" : window.matchMedia("(prefers-color-scheme: dark)").matches;
  themeBtn.textContent = dark ? "Light mode" : "Dark mode";
}
themeBtn.onclick = () => {
  const dark = themeBtn.textContent === "Dark mode";
  const theme = dark ? "dark" : "light";
  try { localStorage.setItem("theme", theme); } catch { /* private mode */ }
  applyTheme(theme);
};
let saved = null;
try { saved = localStorage.getItem("theme"); } catch { /* private mode */ }
applyTheme(saved);

const linked = location.hash.match(/^#c=([0-9a-f]+)$/)?.[1];
if (linked) openConversation(linked); else loadConversations();
checkHealth();
setInterval(checkHealth, 30000);
els.input.focus();
