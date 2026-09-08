/*
 * Operator console.
 *
 * Two things this deliberately does not do:
 *
 *   It never shows a preset without its warning. The warnings are the content -
 *   "low latency" measured better on the fixture and dropped 225 seconds of a
 *   live speaker, and a console that hides that is worse than no console.
 *
 *   It never builds a QR code from a session id. Room URLs survive a restart;
 *   a session id does not, and a printed code built on one is a wall of paper
 *   pointing at a 404 the first time a worker restarts.
 */
(function () {
  "use strict";

  var el = {
    room: document.getElementById("room"),
    status: document.getElementById("status"),
    note: document.getElementById("status-note"),
    pill: document.getElementById("pill"),
    presets: document.getElementById("presets"),
    advanced: document.getElementById("advanced"),
    toggle: document.getElementById("toggle-advanced"),
    restart: document.getElementById("restart"),
    stop: document.getElementById("stop"),
    toast: document.getElementById("toast"),
    qrSpeak: document.getElementById("qr-speak"),
    qrListen: document.getElementById("qr-listen"),
    urlSpeak: document.getElementById("url-speak"),
    urlListen: document.getElementById("url-listen")
  };

  var chosenPreset = null;
  var editable = [];
  var publicBase = "";
  var timer = null;

  function toast(message, bad) {
    el.toast.textContent = message;
    el.toast.className = bad ? "bad" : "";
    el.toast.hidden = false;
    setTimeout(function () { el.toast.hidden = true; }, bad ? 8000 : 3500);
  }

  function room() { return el.room.value.trim(); }

  function api(path, options) {
    return fetch(path, options).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (b) {
          throw new Error(b.detail || ("request failed: " + r.status));
        });
      }
      return r.json();
    });
  }

  // --- status ---------------------------------------------------------------

  function row(term, value, cls) {
    var dt = document.createElement("dt"); dt.textContent = term;
    var dd = document.createElement("dd"); dd.textContent = value;
    if (cls) dd.className = cls;
    el.status.appendChild(dt); el.status.appendChild(dd);
  }

  function refreshStatus() {
    if (!room()) return;
    api("/api/status?room=" + encodeURIComponent(room())).then(function (s) {
      el.status.innerHTML = "";
      el.pill.textContent = s.active ? "running" : "stopped";
      el.pill.className = "pill " + (s.active ? "on" : "off");

      row("Backend", s.backend || "—");
      row("Model", s.model || "—");
      row("Session", s.session_id ? s.session_id.slice(0, 8) : "—");
      row("Chunks", s.chunks);
      // Drops are the number that decides whether a session is usable, so they
      // are coloured even at zero rather than only when something is wrong.
      row("Audio dropped", s.dropped_s + "s", s.dropped_s > 0 ? "bad" : "ok");
      row("Drift skips", s.skips, s.skips > 0 ? "warn" : "ok");
      row("Errors", s.errors, s.errors > 0 ? "bad" : "ok");
      row("Latency p50", s.latency_p50 === null ? "—" : s.latency_p50 + "s");
      row("Latency max", s.latency_max === null ? "—" : s.latency_max + "s",
          s.latency_max > 5 ? "warn" : null);

      if (!s.active) {
        el.note.textContent = "Not running. Apply a preset to start it.";
      } else if (s.waiting_for_speaker) {
        el.note.textContent =
          "Waiting for a speaker. Sessions give up after LAD_TRANSLATE_WAIT " +
          "seconds with nobody publishing.";
      } else if (s.dropped_s > 0) {
        el.note.textContent =
          s.dropped_s + "s of speech never reached the transcript. Listeners " +
          "hear fragments, which sounds like mistranslation and is not.";
      } else {
        el.note.textContent = "";
      }
    }).catch(function (err) { toast(err.message, true); });
  }

  // --- presets --------------------------------------------------------------

  function renderPresets(presets) {
    el.presets.innerHTML = "";
    presets.forEach(function (p) {
      var card = document.createElement("div");
      card.className = "preset";
      card.tabIndex = 0;

      var h = document.createElement("h3");
      h.textContent = p.label;
      if (p.warning) {
        var flag = document.createElement("span");
        flag.className = "pill"; flag.textContent = "caution";
        h.appendChild(flag);
      }
      card.appendChild(h);

      var summary = document.createElement("p");
      summary.textContent = p.summary;
      card.appendChild(summary);

      var measured = document.createElement("p");
      measured.className = "measured";
      measured.textContent = p.measured;
      card.appendChild(measured);

      if (p.warning) {
        var warn = document.createElement("p");
        warn.className = "warning";
        warn.textContent = p.warning;
        card.appendChild(warn);
      }

      function pick() {
        chosenPreset = p.key;
        Array.prototype.forEach.call(
          el.presets.children, function (c) { c.classList.remove("selected"); });
        card.classList.add("selected");
      }
      card.addEventListener("click", pick);
      card.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); pick(); }
      });
      el.presets.appendChild(card);
    });
  }

  // --- advanced -------------------------------------------------------------

  function renderAdvanced(settings) {
    el.advanced.innerHTML = "";
    editable.forEach(function (key) {
      var wrap = document.createElement("div");
      wrap.className = "field";
      var label = document.createElement("label");
      label.textContent = key.replace(/^LAD_TRANSLATE_/, "");
      label.htmlFor = "adv-" + key;
      var input = document.createElement("input");
      input.id = "adv-" + key;
      input.dataset.key = key;
      input.value = settings[key] === undefined ? "" : settings[key];
      wrap.appendChild(label); wrap.appendChild(input);
      el.advanced.appendChild(wrap);
    });
  }

  function advancedChanges(original) {
    var out = {};
    Array.prototype.forEach.call(el.advanced.querySelectorAll("input"), function (i) {
      var was = original[i.dataset.key];
      if (i.value !== "" && i.value !== (was === undefined ? "" : was)) {
        out[i.dataset.key] = i.value;
      }
    });
    return out;
  }

  // --- QR -------------------------------------------------------------------

  function refreshQr() {
    if (!room() || !publicBase) return;
    var stamp = Date.now();   // defeat the cache when the room changes
    el.qrSpeak.src = "/api/qr?kind=speak&room=" + encodeURIComponent(room()) + "&t=" + stamp;
    el.qrListen.src = "/api/qr?kind=listen&room=" + encodeURIComponent(room()) + "&t=" + stamp;
    el.urlSpeak.textContent = publicBase + "/room/" + room() + "/speak";
    el.urlListen.textContent = publicBase + "/room/" + room();
  }

  // --- actions --------------------------------------------------------------

  var currentSettings = {};

  function apply() {
    var body = {
      room: room(),
      preset: chosenPreset,
      settings: advancedChanges(currentSettings),
      restart: true
    };
    el.restart.disabled = true;
    api("/api/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      toast(r.changed.length
        ? "Applied " + r.changed.length + " setting(s) and restarted."
        : "Nothing changed; restarted anyway.");
      return load();
    }).catch(function (err) {
      toast(err.message, true);
    }).then(function () {
      el.restart.disabled = false;
      refreshStatus();
    });
  }

  function stop() {
    api("/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: room() })
    }).then(function () {
      toast("Stopped.");
      refreshStatus();
    }).catch(function (err) { toast(err.message, true); });
  }

  // --- boot -----------------------------------------------------------------

  function load() {
    return Promise.all([api("/api/presets"), api("/api/settings")])
      .then(function (both) {
        renderPresets(both[0].presets);
        editable = both[1].editable;
        currentSettings = both[1].settings;
        publicBase = both[1].public_base;
        renderAdvanced(currentSettings);
        refreshQr();
      });
  }

  el.toggle.addEventListener("click", function () {
    var open = el.advanced.hidden;
    el.advanced.hidden = !open;
    el.toggle.textContent = open ? "hide" : "show";
    el.toggle.setAttribute("aria-expanded", String(open));
  });
  el.restart.addEventListener("click", apply);
  el.stop.addEventListener("click", stop);
  el.room.addEventListener("change", function () { refreshQr(); refreshStatus(); });

  load().then(refreshStatus).catch(function (err) { toast(err.message, true); });
  timer = setInterval(refreshStatus, 5000);
  window.addEventListener("beforeunload", function () { clearInterval(timer); });
})();
