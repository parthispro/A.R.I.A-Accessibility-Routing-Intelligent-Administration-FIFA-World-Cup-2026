/*
A.R.I.A front-end logic.

  Security posture:
    - ALL dynamic text (user input and assistant replies) is rendered with
      document.createElement + textContent / createTextNode. Dynamic content is
      never assigned to an element's markup property, so untrusted model/user
      text can never be parsed as markup (no XSS sink).
    - No inline event handlers; every listener is attached with
      addEventListener. This keeps a strict CSP (default-src 'self') working.

  Accessibility posture:
    - New messages are appended into a role="log" aria-live="polite" region,
      so screen readers announce each new reply.
    - Author is labelled in text ("You:" / "A.R.I.A:").
*/

"use strict";

// ---- Constants ----
const HISTORY_LIMIT = 20;     // keep the last N turns (per API contract)
const MAX_MESSAGE_LEN = 2000; // matches the backend's 1..2000 bound

// Human-readable author labels; used as text prefixes, not colour cues.
const AUTHOR_LABEL = {
  user: "You",
  assistant: "A.R.I.A",
  status: "A.R.I.A",
};

// ---- State ----
/** @type {{role: "user"|"assistant", text: string}[]} */
let history = [];

// ---- Element references ----
const els = {
  venue: document.getElementById("venue-select"),
  language: document.getElementById("language-select"),
  transcript: document.getElementById("transcript"),
  form: document.getElementById("chat-form"),
  input: document.getElementById("message-input"),
  send: document.getElementById("send-button"),
  banner: document.getElementById("offline-banner"),
  fanTab: document.getElementById("fan-tab"),
  adminTab: document.getElementById("admin-tab"),
  fanView: document.getElementById("fan-view"),
  adminView: document.getElementById("admin-view"),
  adminVenue: document.getElementById("admin-venue-select"),
  adminRefresh: document.getElementById("admin-refresh"),
  adminStatus: document.getElementById("admin-status"),
  adminAlerts: document.getElementById("admin-alerts"),
  adminKpis: document.getElementById("admin-kpis"),
  adminTraffic: document.getElementById("admin-traffic-body"),
  adminResources: document.getElementById("admin-resources"),
  adminLoginForm: document.getElementById("admin-login-form"),
  adminUsername: document.getElementById("admin-username"),
  adminPassword: document.getElementById("admin-password"),
  adminLoginMessage: document.getElementById("admin-login-message"),
  adminDashboard: document.getElementById("admin-dashboard"),
  adminSignout: document.getElementById("admin-signout"),
};

// =====================================================================
// Rendering helpers — every one builds DOM via createElement/textContent.
// =====================================================================

/**
 * Append a message bubble to the transcript.
 * @param {"user"|"assistant"|"status"} role
 * @param {string} text
 * @returns {HTMLElement} the created message element
 */
function appendMessage(role, text) {
  const wrap = document.createElement("div");
  wrap.className = "msg msg--" + role;

  // Errors ("status" bubbles) are announced immediately by assistive tech,
  // not queued behind the polite log.
  if (role === "status") {
    wrap.setAttribute("role", "alert");
  }

  const author = document.createElement("span");
  author.className = "msg__author";
  // textContent keeps this a plain string, never parsed as HTML.
  author.textContent = AUTHOR_LABEL[role] + ":";

  const body = document.createElement("span");
  body.className = "msg__text";
  body.textContent = text;

  wrap.appendChild(author);
  wrap.appendChild(body);

  // Stamp lang/dir on this bubble at creation time (not just the transcript
  // container), so a later language switch can't retroactively change how a
  // screen reader pronounces messages that were already sent.
  const lang = els.language.value;
  wrap.setAttribute("lang", lang);
  wrap.setAttribute("dir", lang === "ar" ? "rtl" : "ltr");

  els.transcript.appendChild(wrap);
  els.transcript.scrollTop = els.transcript.scrollHeight;
  return wrap;
}

/** Add a "typing" placeholder bubble; returns it so it can be updated later. */
function appendPending() {
  const wrap = appendMessage("assistant", "");
  wrap.classList.add("msg--pending");
  const body = wrap.querySelector(".msg__text");
  if (body) {
    body.textContent = ""; // CSS renders an ellipsis for the pending state
  }
  return wrap;
}

// =====================================================================
// History management
// =====================================================================

/**
 * Push a turn and cap the array at the last HISTORY_LIMIT entries.
 * @param {"user"|"assistant"} role
 * @param {string} text
 */
function pushHistory(role, text) {
  history.push({ role, text });
  if (history.length > HISTORY_LIMIT) {
    history = history.slice(history.length - HISTORY_LIMIT);
  }
}

// =====================================================================
// Offline-mode banner
// =====================================================================

/** Show/hide the non-alarming offline banner based on the reply mode. */
function setOfflineBanner(isOffline) {
  els.banner.hidden = !isOffline;
}

// =====================================================================
// Language / direction
// =====================================================================

/** Reflect the chosen language on the transcript and the composer input. */
function applyLanguage() {
  const lang = els.language.value;
  const dir = lang === "ar" ? "rtl" : "ltr";
  els.transcript.setAttribute("lang", lang);
  els.transcript.setAttribute("dir", dir);
  // The composer holds text typed in the chosen language, so screen readers
  // and the caret/text direction must follow it too (RTL for Arabic).
  els.input.setAttribute("lang", lang);
  els.input.setAttribute("dir", dir);
}

// =====================================================================
// Venue loading
// =====================================================================

/** Fetch the venue list and populate the select; fail gracefully. */
async function loadVenues() {
  try {
    const res = await fetch("/api/venues", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }
    const data = await res.json();
    const venues = Array.isArray(data.venues) ? data.venues : [];
    for (const v of venues) {
      const opt = document.createElement("option");
      opt.value = String(v.id);
      // Build a readable label from name + city/country (text only).
      const parts = [v.name];
      if (v.city) {
        parts.push(v.city);
      } else if (v.country) {
        parts.push(v.country);
      }
      opt.textContent = parts.join(" — ");
      els.venue.appendChild(opt);
      const adminOpt = document.createElement("option");
      adminOpt.value = String(v.id);
      adminOpt.textContent = parts.join(" — ");
      els.adminVenue.appendChild(adminOpt);
    }
  } catch (err) {
    // Non-fatal: the app still works without a venue. Leave a hint in the list.
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = "Stadium list unavailable — you can still ask questions";
    opt.disabled = true;
    els.venue.appendChild(opt);
  }
}

// =====================================================================
// Health check (optional): pre-set the offline banner if the LLM is down.
// =====================================================================

async function checkHealth() {
  try {
    const res = await fetch("/api/healthz", { headers: { Accept: "application/json" } });
    if (!res.ok) {
      return;
    }
    const data = await res.json();
    if (data && data.llm === "offline") {
      setOfflineBanner(true);
    }
  } catch (err) {
    // Ignore — the banner will settle correctly on the first chat reply.
  }
}

// =====================================================================
// Operations dashboard (read-only simulated snapshot)
// =====================================================================

// Credentials are kept only in memory for the current browser tab. They are
// never written to localStorage, the URL, or the chat history.
let adminCredentials = null;

/** Encode credentials as an HTTP Basic Authorization header value. */
function adminAuthorization(username, password) {
  const bytes = new TextEncoder().encode(username + ":" + password);
  let binary = "";
  bytes.forEach((byte) => {
    binary += String.fromCharCode(byte);
  });
  return "Basic " + btoa(binary);
}

/** Clear all child nodes from an element without parsing any markup. */
function clearChildren(element) {
  while (element.firstChild) {
    element.removeChild(element.firstChild);
  }
}

/** Create a text-only status pill for a dashboard item. */
function statusPill(status) {
  const pill = document.createElement("span");
  pill.className = "status-pill status-pill--" + status;
  pill.textContent = status.replace("_", " ");
  return pill;
}

/** Render one operations overview received from the same-origin API. */
function renderAdminOverview(data) {
  clearChildren(els.adminAlerts);
  clearChildren(els.adminKpis);
  clearChildren(els.adminTraffic);
  clearChildren(els.adminResources);

  const number = new Intl.NumberFormat("en-US");
  const metrics = [
    [
      "Estimated occupancy",
      data.summary.estimated_occupancy_percent + "%",
      data.summary.estimate_note,
    ],
    [
      "Estimated occupied seats",
      number.format(data.summary.estimated_occupied_seats),
      "of " + number.format(data.summary.capacity) + " total seats",
    ],
    ["Available seats", number.format(data.summary.estimated_available_seats), "planning estimate"],
    [
      "Accessible seats available",
      number.format(data.seating.estimated_accessible_available),
      data.seating.estimate_note,
    ],
  ];
  metrics.forEach((metric) => {
    const card = document.createElement("article");
    card.className = "admin-kpi";
    const label = document.createElement("p");
    label.className = "admin-kpi__label";
    label.textContent = metric[0];
    const value = document.createElement("strong");
    value.className = "admin-kpi__value";
    value.textContent = metric[1];
    const note = document.createElement("p");
    note.className = "admin-kpi__note";
    note.textContent = metric[2];
    card.append(label, value, note);
    els.adminKpis.appendChild(card);
  });

  data.alerts.forEach((alert) => {
    const item = document.createElement("article");
    item.className = "admin-alert admin-alert--" + alert.severity;
    const title = document.createElement("strong");
    title.textContent = alert.title;
    const detail = document.createElement("p");
    detail.textContent = alert.detail;
    item.append(title, detail);
    els.adminAlerts.appendChild(item);
  });

  data.gate_traffic.forEach((gate) => {
    const row = document.createElement("tr");
    const values = [
      gate.gate,
      gate.accessible ? "Accessible" : "Standard",
      gate.congestion,
      number.format(gate.estimated_entrants_per_15_min),
      gate.estimated_queue_minutes + " min",
    ];
    values.forEach((value, index) => {
      const cell = document.createElement(index === 0 ? "th" : "td");
      if (index === 0) {
        cell.scope = "row";
      }
      cell.textContent = value;
      row.appendChild(cell);
    });
    els.adminTraffic.appendChild(row);
  });

  data.resources.forEach((resource) => {
    const item = document.createElement("li");
    const head = document.createElement("div");
    head.className = "resource-list__head";
    const name = document.createElement("strong");
    name.textContent = resource.resource;
    head.append(name, statusPill(resource.status));
    const detail = document.createElement("p");
    detail.textContent = resource.detail;
    item.append(head, detail);
    els.adminResources.appendChild(item);
  });

  els.adminStatus.textContent = (
    data.venue_name + " snapshot updated for UTC hour " + data.hour_utc + "."
  );
}

/** Fetch the selected venue's protected operations snapshot. */
async function loadAdminOverview() {
  if (!adminCredentials) {
    els.adminLoginMessage.textContent = "Sign in to access operations data.";
    return false;
  }
  const venueId = els.adminVenue.value;
  if (!venueId) {
    els.adminStatus.textContent = "Choose a venue to load its operations snapshot.";
    return false;
  }
  els.adminStatus.textContent = "Loading operations snapshot…";
  els.adminRefresh.disabled = true;
  try {
    const res = await fetch("/api/admin/venues/" + encodeURIComponent(venueId) + "/overview", {
      headers: {
        Accept: "application/json",
        Authorization: adminAuthorization(adminCredentials.username, adminCredentials.password),
      },
    });
    if (res.status === 401) {
      adminCredentials = null;
      els.adminDashboard.hidden = true;
      els.adminLoginForm.hidden = false;
      els.adminLoginMessage.textContent = "The administrator ID or password is incorrect.";
      els.adminPassword.value = "";
      els.adminPassword.focus();
      return false;
    }
    if (res.status === 503) {
      els.adminLoginMessage.textContent = "Administrator access has not been configured on this deployment.";
      return false;
    }
    if (!res.ok) {
      throw new Error("HTTP " + res.status);
    }
    renderAdminOverview(await res.json());
    return true;
  } catch (err) {
    els.adminStatus.textContent = "The operations snapshot could not be loaded. Please try again.";
    return false;
  } finally {
    els.adminRefresh.disabled = false;
  }
}

/** Verify the supplied administrator credentials and open the dashboard. */
async function submitAdminLogin(event) {
  event.preventDefault();
  adminCredentials = {
    username: els.adminUsername.value.trim(),
    password: els.adminPassword.value,
  };
  els.adminLoginMessage.textContent = "Checking credentials…";

  // Auto-select a venue if needed to allow the API call
  if (!els.adminVenue.value && els.adminVenue.options.length > 1) {
    els.adminVenue.value = els.adminVenue.options[1].value;
  }

  const loaded = await loadAdminOverview();
  if (loaded) {
    els.adminLoginForm.hidden = true;
    els.adminDashboard.hidden = false;
    els.adminLoginMessage.textContent = "";
    els.adminPassword.value = "";
  } else {
    if (els.adminLoginMessage.textContent === "Checking credentials…") {
      els.adminLoginMessage.textContent = "Failed to verify credentials or load snapshot.";
    }
  }
}

/** Drop the in-memory credentials and hide operational information. */
function signOutAdmin() {
  adminCredentials = null;
  els.adminDashboard.hidden = true;
  els.adminLoginForm.hidden = false;
  els.adminUsername.value = "";
  els.adminPassword.value = "";
  els.adminLoginMessage.textContent = "You have signed out.";
  els.adminUsername.focus();
}
/** Switch between the fan assistant and operations dashboard tabs. */
function setActiveView(view) {
  const showingAdmin = view === "admin";
  els.fanView.hidden = showingAdmin;
  els.adminView.hidden = !showingAdmin;
  els.fanTab.setAttribute("aria-selected", String(!showingAdmin));
  els.adminTab.setAttribute("aria-selected", String(showingAdmin));
  els.fanTab.tabIndex = showingAdmin ? -1 : 0;
  els.adminTab.tabIndex = showingAdmin ? 0 : -1;
  if (showingAdmin) {
    if (!els.adminVenue.value && els.venue.value) {
      els.adminVenue.value = els.venue.value;
    }
    els.adminView.focus();
    if (adminCredentials) {
      loadAdminOverview();
    } else {
      els.adminUsername.focus();
    }
  }
}

// =====================================================================
// Sending a message
// =====================================================================

/** Build the profile object from the current form controls. */
function readProfile() {
  const needs = Array.from(
    document.querySelectorAll('input[name="needs"]:checked')
  ).map((el) => el.value);

  return {
    language: els.language.value,
    needs: needs,
    venue_id: els.venue.value ? els.venue.value : null,
  };
}

let inFlight = false;

/**
 * Send a message to the backend and render the reply as it streams in.
 *
 * The reply arrives as newline-delimited JSON frames from /api/chat/stream:
 *   {"type":"meta","mode":...}  then  {"type":"delta","text":...} pieces.
 * Deltas are typed into a bubble that is aria-hidden during streaming, so a
 * screen reader is NOT spammed with each fragment; on completion the streaming
 * bubble is swapped for a single, final bubble that is announced exactly once.
 * @param {string} rawText
 */
async function sendMessage(rawText) {
  const text = rawText.trim();
  if (!text || inFlight) {
    return;
  }
  if (text.length > MAX_MESSAGE_LEN) {
    appendMessage("status", "Message is too long. Please shorten it to 2000 characters or fewer.");
    return;
  }

  // Optimistically render the user's message and record it in history.
  appendMessage("user", text);
  pushHistory("user", text);

  // Lock the UI while the request is in flight. aria-busy tells assistive
  // tech the log is updating; queued changes are announced when it clears.
  inFlight = true;
  els.send.disabled = true;
  els.input.value = "";
  els.transcript.setAttribute("aria-busy", "true");

  // The streaming bubble is hidden from assistive tech: partial tokens must not
  // be announced one-by-one in the aria-live log. The completed reply is added
  // as a fresh (announced) bubble once the stream finishes.
  const pending = appendPending();
  pending.setAttribute("aria-hidden", "true");
  const pendingBody = pending.querySelector(".msg__text");

  const payload = {
    message: text,
    profile: readProfile(),
    history: history.slice(0, HISTORY_LIMIT),
  };

  let replyText = "";
  let started = false; // has the first delta arrived (bubble left pending state)?
  let errored = false;

  const handleFrame = (line) => {
    const trimmed = line.trim();
    if (!trimmed) {
      return;
    }
    let frame;
    try {
      frame = JSON.parse(trimmed);
    } catch (e) {
      return; // ignore a malformed frame rather than break the whole stream
    }
    if (frame.type === "meta") {
      setOfflineBanner(frame.mode === "offline");
    } else if (frame.type === "delta" && typeof frame.text === "string") {
      if (!started) {
        started = true;
        pending.classList.remove("msg--pending");
      }
      replyText += frame.text;
      if (pendingBody) {
        pendingBody.textContent = replyText;
      }
      els.transcript.scrollTop = els.transcript.scrollHeight;
    } else if (frame.type === "error") {
      errored = true;
    }
  };

  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Accept: "application/x-ndjson",
      },
      body: JSON.stringify(payload),
    });

    if (!res.ok || !res.body) {
      throw new Error("HTTP " + res.status);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) {
        break;
      }
      buffer += decoder.decode(value, { stream: true });
      let nl;
      while ((nl = buffer.indexOf("\n")) >= 0) {
        handleFrame(buffer.slice(0, nl));
        buffer = buffer.slice(nl + 1);
      }
    }
    handleFrame(buffer); // final line, if not newline-terminated

    if (errored && !replyText) {
      throw new Error("stream error");
    }

    // Swap the (aria-hidden) streaming bubble for a final, announced bubble so
    // screen readers hear the complete reply exactly once.
    pending.remove();
    const finalText = replyText || "(No reply received.)";
    appendMessage("assistant", finalText);
    if (replyText) {
      pushHistory("assistant", replyText);
    }
  } catch (err) {
    // Polite, non-technical inline error — never a raw stack trace.
    pending.remove();
    appendMessage(
      "status",
      "Sorry, I could not reach the assistant just now. Please check your connection and try again."
    );
  } finally {
    inFlight = false;
    els.send.disabled = false;
    els.transcript.setAttribute("aria-busy", "false");
    els.input.focus(); // return focus to the input after sending
  }
}

// =====================================================================
// Event wiring — all via addEventListener (no inline handlers).
// =====================================================================

function wireEvents() {
  // Submit (covers Enter key and the Send button).
  els.form.addEventListener("submit", (event) => {
    event.preventDefault();
    sendMessage(els.input.value);
  });

  // Language change updates transcript lang/dir.
  els.language.addEventListener("change", applyLanguage);

  els.fanTab.addEventListener("click", () => setActiveView("fan"));
  els.adminTab.addEventListener("click", () => setActiveView("admin"));
  els.adminRefresh.addEventListener("click", loadAdminOverview);
  els.adminLoginForm.addEventListener("submit", submitAdminLogin);
  els.adminSignout.addEventListener("click", signOutAdmin);
  els.venue.addEventListener("change", () => {
    if (els.venue.value) {
      els.adminVenue.value = els.venue.value;
    }
  });
  els.adminVenue.addEventListener("change", () => {
    if (els.adminVenue.value) {
      els.venue.value = els.adminVenue.value;
      loadAdminOverview();
    }
  });

  // Quick-action chips: fill and send their prompt.
  const chips = document.querySelectorAll(".chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const prompt = chip.getAttribute("data-prompt") || chip.textContent || "";
      sendMessage(prompt);
    });
  });
}

// =====================================================================
// Welcome message
// =====================================================================

// Static, first-load greeting. Rendered via the same textContent-only path as
// every other bubble (no markup), and deliberately NOT pushed into `history`,
// so the API contract (only real user/assistant turns) is unchanged.
const WELCOME_TEXT =
  "Hello! I'm A.R.I.A, your accessibility copilot for the FIFA World Cup 2026. " +
  "Pick a stadium and your access needs, or just ask me anything — wheelchair " +
  "routes, sensory rooms, assistive listening, and live gate status.";

function renderWelcome() {
  appendMessage("assistant", WELCOME_TEXT);
}

// =====================================================================
// Init
// =====================================================================

function init() {
  applyLanguage();
  renderWelcome();
  wireEvents();
  loadVenues();
  checkHealth();
  window.setInterval(() => {
    if (!els.adminView.hidden && adminCredentials) {
      loadAdminOverview();
    }
  }, 60000);
}

// `defer` guarantees the DOM is parsed before this runs, but guard anyway.
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
