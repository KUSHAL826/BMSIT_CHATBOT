/* ==========================================================================
   Ask BMSIT - public chatbot client
   Plain ES2019, no build step, no dependencies.
   All rendering escapes HTML before applying a small markdown subset, so
   answers and citations from the knowledge base can never inject markup.
   ========================================================================== */
(function () {
  "use strict";

  var MAX_HISTORY_TURNS = 8;

  var conversation = document.getElementById("conversation");
  var welcome = document.getElementById("welcome");
  var form = document.getElementById("composer");
  var input = document.getElementById("composer-input");
  var sendButton = document.getElementById("send-button");
  var clearButton = document.getElementById("clear-chat");
  var statusPill = document.getElementById("kb-status");

  var history = [];
  var busy = false;

  /* ---------------- helpers ---------------- */

  function escapeHtml(value) {
    return String(value == null ? "" : value)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;")
      .replace(/'/g, "&#39;");
  }

  // Minimal markdown: bold, italics, inline code, links, bullet lists.
  // Runs strictly after escaping.
  function renderMarkdown(text) {
    var safe = escapeHtml(text);

    safe = safe.replace(/`([^`\n]+)`/g, "<code>$1</code>");
    safe = safe.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");
    safe = safe.replace(
      /\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );
    safe = safe.replace(
      /(^|\s)(https?:\/\/[^\s<]+)/g,
      '$1<a href="$2" target="_blank" rel="noopener noreferrer">$2</a>'
    );

    var blocks = safe.split(/\n{2,}/);
    return blocks.map(function (block) {
      var lines = block.split("\n").filter(function (line) { return line.trim() !== ""; });
      if (!lines.length) { return ""; }

      var isList = lines.every(function (line) {
        return /^\s*([-*\u2022]|\d+[.)])\s+/.test(line);
      });
      if (isList) {
        var items = lines.map(function (line) {
          return "<li>" + line.replace(/^\s*([-*\u2022]|\d+[.)])\s+/, "") + "</li>";
        }).join("");
        return "<ul>" + items + "</ul>";
      }
      return "<p>" + lines.join("<br>") + "</p>";
    }).join("");
  }

  function timeLabel() {
    return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  }

  function hideWelcome() {
    if (welcome && welcome.parentNode) {
      welcome.parentNode.removeChild(welcome);
      welcome = null;
    }
  }

  function scrollToEnd() {
    conversation.scrollTop = conversation.scrollHeight;
  }

  /* ---------------- rendering ---------------- */

  function addMessage(role, html, metaText) {
    hideWelcome();

    var wrapper = document.createElement("div");
    wrapper.className = "message " + role;

    var avatar = document.createElement("div");
    avatar.className = "avatar";
    avatar.setAttribute("aria-hidden", "true");
    avatar.textContent = role === "user" ? "You" : (role === "system" ? "!" : "AI");

    var column = document.createElement("div");
    column.className = "bubble-wrap";

    var bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.innerHTML = html;
    column.appendChild(bubble);

    if (metaText) {
      var meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = metaText;
      column.appendChild(meta);
    }

    wrapper.appendChild(avatar);
    wrapper.appendChild(column);
    conversation.appendChild(wrapper);
    scrollToEnd();
    return { wrapper: wrapper, bubble: bubble, column: column };
  }

  function addCitations(column, sources) {
    if (!sources || !sources.length) { return; }

    var details = document.createElement("details");
    details.className = "citations";

    var summary = document.createElement("summary");
    summary.textContent = sources.length === 1
      ? "1 source used"
      : sources.length + " sources used";
    details.appendChild(summary);

    var list = document.createElement("ul");
    list.className = "citation-list";

    sources.forEach(function (source) {
      var item = document.createElement("li");
      item.className = "citation";

      var name = document.createElement("span");
      name.className = "citation-name";
      if (source.url) {
        var link = document.createElement("a");
        link.href = source.url;
        link.target = "_blank";
        link.rel = "noopener noreferrer";
        link.textContent = source.name || "BMSIT source";
        name.appendChild(link);
      } else {
        name.textContent = source.name || "BMSIT source";
      }
      item.appendChild(name);

      if (source.section) {
        var detail = document.createElement("span");
        detail.className = "citation-detail";
        detail.textContent = source.section;
        item.appendChild(detail);
      }

      if (typeof source.score === "number") {
        var score = document.createElement("span");
        score.className = "citation-score";
        score.textContent = source.score.toFixed(2);
        score.title = "Relevance score";
        item.appendChild(score);
      }

      list.appendChild(item);
    });

    details.appendChild(list);
    column.appendChild(details);
    scrollToEnd();
  }

  function addThinking() {
    var placeholder = addMessage(
      "bot",
      '<span class="thinking" role="status" aria-label="Searching the BMSIT knowledge base">' +
      "<span></span><span></span><span></span></span>"
    );
    return placeholder;
  }

  /* ---------------- status ---------------- */

  function setStatus(state, label, title) {
    if (!statusPill) { return; }
    statusPill.className = "status-pill " + state;
    var text = statusPill.querySelector(".status-label");
    if (text) { text.textContent = label; }
    statusPill.title = title || label;
  }

  function loadStatus() {
    fetch("/health", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) { throw new Error("health " + response.status); }
        return response.json();
      })
      .then(function (data) {
        if (!data.chunks) {
          setStatus("is-empty", "No data yet",
            "The knowledge base is empty. An administrator needs to run a crawl or upload documents.");
          addMessage("system", renderMarkdown(
            "The knowledge base is currently empty, so I have nothing verified to answer from. " +
            "An administrator can populate it from the admin dashboard."
          ));
          return;
        }
        setStatus("is-ready", "Ready",
          data.chunks + " indexed passages from " + data.sources + " source(s)");
      })
      .catch(function () {
        setStatus("is-down", "Offline", "Could not reach the assistant service.");
      });
  }

  /* ---------------- sending ---------------- */

  function setBusy(value) {
    busy = value;
    sendButton.disabled = value;
    input.disabled = value;
    if (!value) { input.focus(); }
  }

  function autoGrow() {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 168) + "px";
  }

  function send(message) {
    if (busy) { return; }
    var text = (message || input.value || "").trim();
    if (!text) { return; }

    addMessage("user", renderMarkdown(text), timeLabel());
    input.value = "";
    autoGrow();
    setBusy(true);

    var placeholder = addThinking();

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json", Accept: "application/json" },
      body: JSON.stringify({ message: text, history: history.slice(-MAX_HISTORY_TURNS) })
    })
      .then(function (response) {
        return response.json().catch(function () {
          throw new Error("The assistant returned an unreadable response.");
        });
      })
      .then(function (data) {
        var reply = data.reply || data.error || "I could not produce an answer for that.";
        placeholder.bubble.innerHTML = renderMarkdown(reply);

        var meta = document.createElement("div");
        meta.className = "meta";
        meta.textContent = timeLabel() + (data.guardrail_triggered ? " · policy filter" : "");
        placeholder.column.appendChild(meta);

        addCitations(placeholder.column, data.sources);

        history.push({ role: "user", text: text });
        history.push({ role: "assistant", text: reply });
        if (history.length > MAX_HISTORY_TURNS * 2) {
          history = history.slice(-MAX_HISTORY_TURNS * 2);
        }
      })
      .catch(function (error) {
        placeholder.wrapper.className = "message system";
        placeholder.bubble.innerHTML = renderMarkdown(
          "I could not reach the assistant just now. " + (error.message || "") +
          " Please try again, or contact the college at `admissions@bmsit.in`."
        );
      })
      .then(function () {
        setBusy(false);
        scrollToEnd();
      });
  }

  /* ---------------- events ---------------- */

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    send();
  });

  input.addEventListener("input", autoGrow);

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      send();
    }
  });

  document.addEventListener("click", function (event) {
    var suggestion = event.target.closest ? event.target.closest(".suggestion") : null;
    if (suggestion && suggestion.dataset.q) {
      send(suggestion.dataset.q);
    }
  });

  if (clearButton) {
    clearButton.addEventListener("click", function () {
      history = [];
      conversation.innerHTML = "";
      var fresh = document.createElement("section");
      fresh.className = "welcome";
      fresh.innerHTML =
        "<h2>New chat</h2><p>Ask me anything about BMSIT admissions, courses, " +
        "placements, hostels or campus facilities.</p>";
      conversation.appendChild(fresh);
      welcome = fresh;
      input.focus();
    });
  }

  loadStatus();
  autoGrow();
  input.focus();
})();
