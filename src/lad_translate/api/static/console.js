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

  var $ = function (id) { return document.getElementById(id); };

  var el = {
    room: $("room"),
    pill: $("pill"),
    refreshed: $("refreshed"),
    pageTitle: $("page-title"),
    publicHost: $("public-host"),
    who: $("who"),
    avatar: $("avatar"),
    build: $("build"),
    facts: $("facts"),
    kpis: $("kpis"),
    note: $("status-note"),
    restart: $("restart"),
    stop: $("stop"),
    presetCards: $("preset-cards"),
    presetChosen: $("preset-chosen"),
    lcd: $("lcd"),
    lcdName: $("lcd-name"),
    lcdTag: $("lcd-tag"),
    lcdSpec: $("lcd-spec"),
    lcdSummary: $("lcd-summary"),
    lcdMeasured: $("lcd-measured"),
    lcdWarning: $("lcd-warning"),
    faderEmit: $("fader-emit"),
    faderWindow: $("fader-window"),
    faderEmitValue: $("fader-emit-value"),
    faderWindowValue: $("fader-window-value"),
    faderReset: $("fader-reset"),
    ledRun: $("led-run"),
    ledWait: $("led-wait"),
    ledDrop: $("led-drop"),
    deckApply: $("deck-apply"),
    deckStop: $("deck-stop"),
    advanced: $("advanced-fields"),
    advancedChanges: $("advanced-changes"),
    toggle: $("toggle-advanced"),
    toasts: $("toasts"),
    qrSpeak: $("qr-speak"),
    qrListen: $("qr-listen"),
    urlSpeak: $("url-speak"),
    urlListen: $("url-listen"),
    pngSpeak: $("png-speak"),
    pngListen: $("png-listen"),
    outPill: $("outputs-pill"),
    outState: $("outputs-state"),
    devices: $("devices"),
    addDevice: $("add-device"),
    editor: $("device-editor"),
    editorTitle: $("editor-title"),
    devName: $("dev-name"),
    devDeviceName: $("dev-device-name"),
    devKind: $("dev-kind"),
    devChannelCount: $("dev-channel-count"),
    devSampleRate: $("dev-sample-rate"),
    devEnabled: $("dev-enabled"),
    matrix: $("matrix"),
    matrixSummary: $("matrix-summary"),
    saveDevice: $("save-device"),
    cancelDevice: $("cancel-device"),
    cancelDeviceX: $("cancel-device-x")
  };

  // Absolute and prefixed, matching where the app actually serves. Relative
  // would depend on whether the browser landed on /console or /console/.
  var BASE = "/console";

  var chosenPreset = null;
  var presetsByKey = {};
  var editable = [];
  var publicBase = "";
  var currentSettings = {};
  var timer = null;

  // --- plumbing -------------------------------------------------------------

  function toast(message, kind) {
    var t = document.createElement("div");
    t.className = "toast " + (kind === true ? "bad" : (kind || ""));
    var icon = document.createElement("span");
    icon.className = "t-icon";
    icon.textContent = kind === true ? "✕" : (kind === "ok" ? "✓" : "•");
    var text = document.createElement("span");
    text.textContent = message;
    t.appendChild(icon); t.appendChild(text);
    el.toasts.appendChild(t);
    setTimeout(function () { if (t.parentNode) t.parentNode.removeChild(t); }, kind === true ? 8000 : 3500);
  }

  function room() { return el.room.value.trim(); }

  function api(path, options) {
    return fetch(path, options).then(function (r) {
      if (!r.ok) {
        return r.json().catch(function () { return {}; }).then(function (b) {
          throw new Error(b.detail || ("request failed: " + r.status));
        });
      }
      return r.status === 204 ? null : r.json();
    });
  }

  function h(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function clock() {
    var d = new Date();
    return [d.getHours(), d.getMinutes(), d.getSeconds()].map(function (n) {
      return (n < 10 ? "0" : "") + n;
    }).join(":");
  }

  // --- who is here ------------------------------------------------------------

  function loadIdentity() {
    api(BASE + "/api/whoami").then(function (w) {
      var email = w.email || "";
      el.who.textContent = email || "not signed in";
      el.who.title = email;
      el.avatar.textContent = (email.charAt(0) || "·").toUpperCase();
    }).catch(function () { el.who.textContent = "—"; });
  }

  // --- status -----------------------------------------------------------------

  function fact(k, v) {
    var f = h("span", "fact");
    f.appendChild(h("span", "k", k));
    f.appendChild(h("span", "v", v));
    el.facts.appendChild(f);
  }

  function kpi(label, value, unit, cls, sub) {
    var k = h("div", "kpi " + (cls || ""));
    k.appendChild(h("div", "label", label));
    var v = h("div", "value", value);
    if (unit) v.appendChild(h("small", null, unit));
    k.appendChild(v);
    if (sub) k.appendChild(h("div", "sub", sub));
    el.kpis.appendChild(k);
  }

  function refreshStatus() {
    if (!room()) return;
    api(BASE + "/api/status?room=" + encodeURIComponent(room())).then(function (s) {
      el.facts.innerHTML = "";
      el.kpis.innerHTML = "";

      var state = !s.active ? "stopped" : (s.waiting_for_speaker ? "waiting for speaker" : "running");
      el.pill.textContent = state;
      el.pill.className = "pill " + (!s.active ? "off" : (s.waiting_for_speaker ? "warn" : "on"));
      el.refreshed.textContent = "refreshed " + clock();

      fact("room", s.room || room());
      fact("session", s.session_id ? s.session_id.slice(0, 8) : "—");
      fact("backend", s.backend || "—");
      fact("model", s.model || "—");
      if (s.since) fact("since", s.since);

      // Drops are the number that decides whether a session is usable, so they
      // are coloured even at zero rather than only when something is wrong.
      kpi("Audio dropped", s.dropped_s, "s", s.dropped_s > 0 ? "bad" : "ok",
          s.dropped_s > 0 ? "speech that never reached the transcript" : "nothing shed");
      kpi("Drift skips", s.skips, "", s.skips > 0 ? "warn" : "ok",
          "phrases skipped to catch up");
      kpi("Errors", s.errors, "", s.errors > 0 ? "bad" : "ok", "in the session log");
      kpi("Chunks", s.chunks, "", "", "phrases translated");
      kpi("Latency p50", s.latency_p50 === null ? "—" : s.latency_p50, s.latency_p50 === null ? "" : "s",
          "", "glass to glass");
      kpi("Latency max", s.latency_max === null ? "—" : s.latency_max, s.latency_max === null ? "" : "s",
          s.latency_max > 5 ? "warn" : "", "worst phrase");

      var note = "", cls = "alert";
      if (!s.active) {
        note = "Not running. Pick a preset and Apply & restart to start it.";
      } else if (s.waiting_for_speaker) {
        note = "Waiting for a speaker. Sessions give up after LAD_TRANSLATE_WAIT seconds with nobody publishing.";
        cls = "alert warn";
      } else if (s.dropped_s > 0) {
        note = s.dropped_s + "s of speech never reached the transcript. Listeners hear fragments, " +
               "which sounds like mistranslation and is not.";
        cls = "alert bad";
      }
      el.note.textContent = note;
      el.note.className = cls;
      el.note.hidden = !note;
      setLeds(s);
    }).catch(function (err) {
      el.pill.textContent = "unreachable";
      el.pill.className = "pill off";
      el.refreshed.textContent = err.message;
      setLeds({ active: false, waiting_for_speaker: false, dropped_s: 0 });
    });
  }

  // --- preset deck ------------------------------------------------------------
  //
  // Pads arm a preset; the LCD shows what the armed (or hovered) preset
  // measured, warning included; the faders are the chunker pair, linked
  // because moving one alone is how audio gets shed; APPLY commits. The pad
  // that is lit green is what the box is running now, worked out by matching
  // the live settings against each preset's numbers, so an operator can see
  // at a glance whether the deck and the box agree.

  var EMIT_KEY = "LAD_TRANSLATE_EMIT_INTERVAL";
  var WINDOW_KEY = "LAD_TRANSLATE_WINDOW";
  var faderOverride = false;   // the operator moved a fader since arming a preset

  function livePresetKey() {
    var keys = Object.keys(presetsByKey);
    for (var i = 0; i < keys.length; i++) {
      var p = presetsByKey[keys[i]];
      if (currentSettings.STT_BACKEND === p.stt_backend &&
          currentSettings.LAD_TRANSLATE_STT_MODEL === p.model &&
          parseFloat(currentSettings[EMIT_KEY]) === p.emit_interval &&
          parseFloat(currentSettings[WINDOW_KEY]) === p.window) {
        return p.key;
      }
    }
    return null;
  }

  function showOnLcd(preset, tag) {
    if (!preset) {
      el.lcd.classList.add("blank");
      el.lcdName.textContent = "NO PRESET";
      el.lcdTag.textContent = ""; el.lcdTag.className = "lcd-tag";
      el.lcdSpec.textContent = currentSettings.STT_BACKEND
        ? (currentSettings.STT_BACKEND + " " + (currentSettings.LAD_TRANSLATE_STT_MODEL || "") +
           "  " + (currentSettings[EMIT_KEY] || "?") + "/" + (currentSettings[WINDOW_KEY] || "?"))
        : "";
      el.lcdSummary.textContent = "The box is on raw settings that match no preset.";
      el.lcdMeasured.textContent = "";
      el.lcdWarning.hidden = true;
      return;
    }
    el.lcd.classList.remove("blank");
    el.lcdName.textContent = preset.label;
    el.lcdTag.textContent = tag || "";
    el.lcdTag.className = "lcd-tag" + (tag === "LIVE" ? " live" : "");
    el.lcdSpec.textContent = preset.stt_backend + " " + preset.model + "  " +
      preset.emit_interval.toFixed(1) + "/" + preset.window.toFixed(1) + "  " + preset.lookahead;
    el.lcdSummary.textContent = preset.summary;
    el.lcdMeasured.textContent = preset.measured;
    el.lcdWarning.textContent = preset.warning || "";
    el.lcdWarning.hidden = !preset.warning;
  }

  function lcdDefault() {
    if (chosenPreset) { showOnLcd(presetsByKey[chosenPreset], "ARMED"); return; }
    var live = livePresetKey();
    showOnLcd(live ? presetsByKey[live] : null, live ? "LIVE" : "");
  }

  function setFaders(emit, win, override) {
    el.faderEmit.value = emit;
    el.faderWindow.value = win;
    el.faderEmitValue.textContent = parseFloat(emit).toFixed(1);
    el.faderWindowValue.textContent = parseFloat(win).toFixed(1);
    el.faderEmitValue.classList.toggle("override", !!override);
    el.faderWindowValue.classList.toggle("override", !!override);
  }

  function syncAdvancedFromFaders() {
    // The faders and the Advanced fields are the same two settings. One
    // path to the server: apply() reads the Advanced fields, so the faders
    // write there and the badge counts them like any other edit.
    var emitInput = $("adv-" + EMIT_KEY), winInput = $("adv-" + WINDOW_KEY);
    if (emitInput) emitInput.value = parseFloat(el.faderEmit.value).toFixed(1);
    if (winInput) winInput.value = parseFloat(el.faderWindow.value).toFixed(1);
    updateAdvancedBadge();
  }

  function fadersFollowArmed() {
    faderOverride = false;
    var p = chosenPreset ? presetsByKey[chosenPreset] : null;
    if (p) {
      setFaders(p.emit_interval, p.window, false);
      // Arming a preset means the preset's pair, not a stale override: put
      // the Advanced fields back to the live values so apply() sends the
      // preset alone and the server's "raw values win" cannot bite.
      var emitInput = $("adv-" + EMIT_KEY), winInput = $("adv-" + WINDOW_KEY);
      if (emitInput) emitInput.value = currentSettings[EMIT_KEY] || "";
      if (winInput) winInput.value = currentSettings[WINDOW_KEY] || "";
      updateAdvancedBadge();
    } else {
      setFaders(parseFloat(currentSettings[EMIT_KEY]) || 3.0, parseFloat(currentSettings[WINDOW_KEY]) || 6.0, false);
    }
  }

  function onFaderInput() {
    faderOverride = true;
    setFaders(el.faderEmit.value, el.faderWindow.value, true);
    syncAdvancedFromFaders();
  }

  function selectPreset(key) {
    chosenPreset = key;
    Array.prototype.forEach.call(el.presetCards.children, function (c) {
      var armed = c.dataset.key === key;
      c.classList.toggle("armed", armed);
      c.setAttribute("aria-pressed", String(armed));
    });
    var p = presetsByKey[key];
    el.presetChosen.textContent = p ? "armed: " + p.label : "nothing armed";
    el.presetChosen.className = "chip" + (p ? " accent" : "");
    fadersFollowArmed();
    lcdDefault();
  }

  function markLivePad() {
    var live = livePresetKey();
    Array.prototype.forEach.call(el.presetCards.children, function (c) {
      var isLive = c.dataset.key === live;
      c.classList.toggle("live", isLive);
      var led = c.querySelector(".pad-led");
      if (led) led.className = "led pad-led" + (isLive ? " on" : "");
      var sub = c.querySelector(".pad-sub");
      if (sub) sub.textContent = presetSub(presetsByKey[c.dataset.key]) + (isLive ? "  · LIVE" : "");
    });
  }

  function presetSub(p) {
    return p.stt_backend.replace("faster-whisper", "whisper") + " " + p.model + " · " +
      p.emit_interval.toFixed(1) + "/" + p.window.toFixed(1);
  }

  function renderPresets(presets) {
    el.presetCards.innerHTML = "";
    presetsByKey = {};
    presets.forEach(function (p) {
      presetsByKey[p.key] = p;
      var pad = h("button", "pad");
      pad.type = "button";
      pad.dataset.key = p.key;
      pad.setAttribute("aria-pressed", "false");
      pad.title = p.summary;
      pad.appendChild(h("span", "led pad-led"));
      if (p.warning) {
        var caution = h("span", "led pad-caution warn");
        caution.title = "Carries a warning - read the display";
        pad.appendChild(caution);
      }
      pad.appendChild(h("span", "pad-name", p.label));
      pad.appendChild(h("span", "pad-sub", presetSub(p)));
      pad.addEventListener("click", function () { selectPreset(p.key); });
      pad.addEventListener("mouseenter", function () { showOnLcd(p, chosenPreset === p.key ? "ARMED" : (livePresetKey() === p.key ? "LIVE" : "")); });
      pad.addEventListener("mouseleave", lcdDefault);
      pad.addEventListener("focus", function () { showOnLcd(p, chosenPreset === p.key ? "ARMED" : ""); });
      pad.addEventListener("blur", lcdDefault);
      el.presetCards.appendChild(pad);
    });
    markLivePad();
    lcdDefault();
  }

  function setLeds(s) {
    el.ledRun.className = "led" + (s.active && !s.waiting_for_speaker ? " on" : "");
    el.ledWait.className = "led" + (s.active && s.waiting_for_speaker ? " warn" : "");
    el.ledDrop.className = "led" + (s.dropped_s > 0 ? " bad" : "");
  }

  // --- advanced ----------------------------------------------------------------

  function renderAdvanced(settings) {
    el.advanced.innerHTML = "";
    editable.forEach(function (key) {
      var label = h("label");
      label.appendChild(document.createTextNode(key.replace(/^LAD_TRANSLATE_/, "")));
      var input = document.createElement("input");
      input.id = "adv-" + key;
      input.dataset.key = key;
      input.spellcheck = false;
      input.value = settings[key] === undefined ? "" : settings[key];
      input.addEventListener("input", function () {
        updateAdvancedBadge();
        if (key === EMIT_KEY || key === WINDOW_KEY) {
          faderOverride = true;
          setFaders(
            key === EMIT_KEY ? (parseFloat(input.value) || el.faderEmit.value) : el.faderEmit.value,
            key === WINDOW_KEY ? (parseFloat(input.value) || el.faderWindow.value) : el.faderWindow.value,
            true
          );
        }
      });
      label.appendChild(input);
      el.advanced.appendChild(label);
    });
    updateAdvancedBadge();
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

  function updateAdvancedBadge() {
    var n = Object.keys(advancedChanges(currentSettings)).length;
    el.advancedChanges.textContent = n + " changed";
    el.advancedChanges.className = "chip" + (n ? " accent" : "");
    el.advancedChanges.hidden = n === 0;
  }

  // --- QR ---------------------------------------------------------------------

  function refreshQr() {
    if (!room()) return;
    // One authenticated fetch, then data URIs. Pointing <img> at the API meant
    // two more network requests carrying no-store, which the browser answered
    // with a second sign-in dialog over an already-loaded page - a login that
    // appeared not to stay logged in. A data URI is not a request.
    api(BASE + "/api/qr.json?room=" + encodeURIComponent(room())).then(function (r) {
      el.qrSpeak.src = r.images.speak;
      el.qrListen.src = r.images.listen;
      el.urlSpeak.textContent = r.urls.speak;
      el.urlListen.textContent = r.urls.listen;
      // Printing wants a file. A link is a navigation, which the cookie rides
      // on; only an <img> at this endpoint was ever the problem.
      var q = "?room=" + encodeURIComponent(room());
      el.pngSpeak.href = BASE + "/api/qr" + q + "&kind=speak";
      el.pngListen.href = BASE + "/api/qr" + q + "&kind=listen";
    }).catch(function (err) { toast(err.message, true); });
  }

  function copyFrom(id) {
    var text = $(id).textContent;
    if (!text) return;
    var done = function () { toast("Copied.", "ok"); };
    if (navigator.clipboard && navigator.clipboard.writeText) {
      navigator.clipboard.writeText(text).then(done, function () { toast(text); });
    } else {
      toast(text);
    }
  }

  // --- actions ------------------------------------------------------------------

  function apply() {
    var body = {
      room: room(),
      preset: chosenPreset,
      settings: advancedChanges(currentSettings),
      restart: true
    };
    el.restart.disabled = true;
    el.deckApply.disabled = true;
    api(BASE + "/api/apply", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (r) {
      toast(r.changed.length
        ? "Applied " + r.changed.length + " setting(s) and restarted."
        : "Nothing changed; restarted anyway.", "ok");
      return load();
    }).catch(function (err) {
      toast(err.message, true);
    }).then(function () {
      el.restart.disabled = false;
      el.deckApply.disabled = false;
      refreshStatus();
    });
  }

  function stop() {
    if (!window.confirm("Stop the session in " + room() + "? Listeners will hear nothing until it is restarted.")) return;
    api(BASE + "/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: room() })
    }).then(function () {
      toast("Stopped.", "ok");
      refreshStatus();
    }).catch(function (err) { toast(err.message, true); });
  }

  // --- hardware output ------------------------------------------------------
  //
  // The channel map is the venue's, not the session's, so it is loaded once
  // and on demand rather than on the 5s status timer. Three states are shown
  // as states, not as an empty list: no database, migration pending, and
  // genuinely no devices. The first two used to be indistinguishable from
  // the third, which is the kind of empty that costs a morning at a venue.

  var outputs = { languages: [], kinds: [], sampleRates: [], devices: [] };
  var editing = null;      // device being edited, or null for a new one
  var rowState = {};       // channel -> {language, ir_channel, label, gain_db, enabled}

  function languageName(code) {
    for (var i = 0; i < outputs.languages.length; i++) {
      if (outputs.languages[i].code === code) return outputs.languages[i].native;
    }
    return code.toUpperCase();
  }

  function kindInfo(key) {
    for (var k = 0; k < outputs.kinds.length; k++) {
      if (outputs.kinds[k].key === key) return outputs.kinds[k];
    }
    return null;
  }

  function setOutputsState(pillText, pillClass, message) {
    el.outPill.textContent = pillText;
    el.outPill.className = "pill " + (pillClass || "");
    if (message) { el.outState.textContent = message; el.outState.hidden = false; }
    else { el.outState.hidden = true; }
  }

  function renderDevices() {
    el.devices.innerHTML = "";
    if (!outputs.devices.length && !el.addDevice.disabled) {
      var empty = h("p", "muted", "No devices yet. Add the venue's rig once; it is kept across sessions.");
      el.devices.appendChild(empty);
    }
    outputs.devices.forEach(function (d) {
      var card = h("div", "device");

      var head = h("h3", null, d.name);
      head.appendChild(h("span", "pill " + (d.enabled ? "on" : "off"), d.enabled ? "enabled" : "disabled"));
      card.appendChild(head);

      var ki = kindInfo(d.kind);
      card.appendChild(h("p", "meta",
        d.kind + (ki && ki.built === false ? " (not built)" : "") + " · " + d.device_name +
        " · " + d.channel_count + " channels · " + d.sample_rate + " Hz"));

      var map = h("ul", "map");
      if (!d.channels.length) map.appendChild(h("li", null, "no channels assigned"));
      d.channels.forEach(function (c) {
        var li = h("li");
        var text = "ch " + c.channel + " → " + languageName(c.language);
        if (c.gain_db) text += " " + (c.gain_db > 0 ? "+" : "") + c.gain_db + " dB";
        if (c.label) text += " “" + c.label + "”";
        li.appendChild(h("span", c.enabled ? null : "off", text));
        if (c.ir_channel !== null) li.appendChild(h("span", "ir", "  IR " + c.ir_channel));
        map.appendChild(li);
      });
      card.appendChild(map);

      var actions = h("div", "actions");
      var edit = h("button", "small", "Edit");
      edit.addEventListener("click", function () { openEditor(d); });
      var signage = h("button", "small", "Signage");
      signage.title = "The IR channel list, as text, for printing";
      signage.addEventListener("click", function () {
        window.open(BASE + "/api/outputs/devices/" + d.device_id + "/signage", "_blank");
      });
      var profile = h("button", "small", "Profile JSON");
      profile.title = "The device as saved, for tools/output_agent.py at the venue";
      profile.addEventListener("click", function () {
        // A Blob, not a data: URI, so a 64-channel map does not become a
        // 20 KB href; and a download, not a new tab, because the agent
        // wants a file.
        var blob = new Blob([JSON.stringify(d, null, 2)], { type: "application/json" });
        var a = document.createElement("a");
        a.href = URL.createObjectURL(blob);
        a.download = d.name.replace(/[^\w.-]+/g, "-").toLowerCase() + ".json";
        document.body.appendChild(a); a.click(); document.body.removeChild(a);
        setTimeout(function () { URL.revokeObjectURL(a.href); }, 1000);
      });
      var del = h("button", "small danger", "Delete");
      del.addEventListener("click", function () { deleteDevice(d); });
      actions.appendChild(edit); actions.appendChild(signage);
      actions.appendChild(profile); actions.appendChild(del);
      card.appendChild(actions);

      el.devices.appendChild(card);
    });
  }

  function loadOutputs() {
    return api(BASE + "/api/outputs").then(function (o) {
      outputs.languages = o.languages;
      outputs.kinds = o.kinds;
      outputs.sampleRates = o.sample_rates;
      outputs.devices = o.devices;

      if (!o.configured) {
        setOutputsState("not configured", "warn", o.reason);
        el.addDevice.disabled = true;
      } else if (o.tenant === null) {
        setOutputsState("no tenant", "warn", o.reason);
        el.addDevice.disabled = true;
      } else if (o.migrated === false) {
        setOutputsState("migration pending", "warn", o.reason);
        el.addDevice.disabled = true;
      } else {
        setOutputsState(o.devices.length ? o.devices.length + " device" +
          (o.devices.length === 1 ? "" : "s") : "no devices yet", "on", "");
        el.addDevice.disabled = false;
      }
      renderDevices();
    }).catch(function (err) {
      setOutputsState("unavailable", "off", err.message);
    });
  }

  function fillSelect(select, values, current) {
    select.innerHTML = "";
    values.forEach(function (v) {
      var opt = document.createElement("option");
      // Kinds arrive as {key, built, engine}; sample rates as numbers. A kind
      // without an engine says so in its own label, because a bare value in
      // a dropdown reads as something that works.
      var key = (v !== null && typeof v === "object") ? v.key : v;
      opt.value = String(key);
      opt.textContent = (v !== null && typeof v === "object" && v.built === false)
        ? key + " — not built"
        : String(key);
      if (v !== null && typeof v === "object" && v.engine) opt.title = v.engine;
      if (String(key) === String(current)) opt.selected = true;
      select.appendChild(opt);
    });
  }

  function defaultKind() {
    // The first kind with an engine, so a new device starts on something
    // that routes audio rather than on the schema's first value.
    for (var i = 0; i < outputs.kinds.length; i++) {
      if (outputs.kinds[i].built) return outputs.kinds[i].key;
    }
    return outputs.kinds.length ? outputs.kinds[0].key : "aes67";
  }

  function blankRow() { return { language: "", ir_channel: "", label: "", gain_db: 0, enabled: true }; }

  function updateMatrixSummary() {
    var assigned = 0, ir = 0;
    Object.keys(rowState).forEach(function (ch) {
      if (rowState[ch].language) {
        assigned++;
        if (rowState[ch].ir_channel !== "" && rowState[ch].ir_channel !== null) ir++;
      }
    });
    el.matrixSummary.textContent = assigned
      ? assigned + " channel" + (assigned === 1 ? "" : "s") + " patched, " + ir + " on IR handsets"
      : "Nothing patched yet — pick a language on a channel.";
  }

  function renderMatrix() {
    var count = Math.max(1, Math.min(64, parseInt(el.devChannelCount.value, 10) || 16));
    el.matrix.innerHTML = "";
    for (var ch = 1; ch <= count; ch++) {
      (function (ch) {
        var st = rowState[ch] || blankRow();
        var tr = document.createElement("tr");
        if (st.language) tr.className = "assigned";

        var tdCh = h("td", "ch", String(ch));
        tr.appendChild(tdCh);

        var tdLang = document.createElement("td");
        var sel = document.createElement("select");
        var blank = document.createElement("option");
        blank.value = ""; blank.textContent = "—";
        sel.appendChild(blank);
        outputs.languages.forEach(function (l) {
          var opt = document.createElement("option");
          opt.value = l.code;
          opt.textContent = l.native + (l.english !== l.native ? " (" + l.english + ")" : "");
          if (l.code === st.language) opt.selected = true;
          sel.appendChild(opt);
        });
        sel.addEventListener("change", function () {
          rowState[ch] = rowState[ch] || blankRow();
          rowState[ch].language = sel.value;
          tr.className = sel.value ? "assigned" : "";
          updateMatrixSummary();
        });
        tdLang.appendChild(sel); tr.appendChild(tdLang);

        function numberCell(key, min, max, step, cls) {
          var td = document.createElement("td");
          var input = document.createElement("input");
          input.type = "number"; input.min = min; input.max = max; input.step = step;
          input.className = cls;
          input.value = st[key] === null || st[key] === undefined ? "" : st[key];
          input.addEventListener("input", function () {
            rowState[ch] = rowState[ch] || blankRow();
            rowState[ch][key] = input.value;
            updateMatrixSummary();
          });
          td.appendChild(input);
          return td;
        }
        tr.appendChild(numberCell("ir_channel", 1, 99, 1, "narrow"));

        var tdLabel = document.createElement("td");
        var label = document.createElement("input");
        label.value = st.label || "";
        label.placeholder = "e.g. Recorder";
        label.addEventListener("input", function () {
          rowState[ch] = rowState[ch] || blankRow();
          rowState[ch].label = label.value;
        });
        tdLabel.appendChild(label); tr.appendChild(tdLabel);

        tr.appendChild(numberCell("gain_db", -60, 12, 0.5, "narrow"));

        var tdOn = document.createElement("td");
        var on = document.createElement("input");
        on.type = "checkbox"; on.checked = st.enabled !== false;
        on.addEventListener("change", function () {
          rowState[ch] = rowState[ch] || blankRow();
          rowState[ch].enabled = on.checked;
        });
        tdOn.appendChild(on); tr.appendChild(tdOn);

        el.matrix.appendChild(tr);
      })(ch);
    }
    updateMatrixSummary();
  }

  function openEditor(device) {
    editing = device || null;
    rowState = {};
    el.editorTitle.textContent = device ? "Edit “" + device.name + "”" : "New device";
    el.devName.value = device ? device.name : "";
    el.devDeviceName.value = device ? device.device_name : "";
    fillSelect(el.devKind, outputs.kinds, device ? device.kind : defaultKind());
    el.devChannelCount.value = device ? device.channel_count : 16;
    fillSelect(el.devSampleRate, outputs.sampleRates, device ? device.sample_rate : 48000);
    el.devEnabled.checked = device ? device.enabled : true;
    if (device) {
      device.channels.forEach(function (c) {
        rowState[c.channel] = {
          language: c.language,
          ir_channel: c.ir_channel === null ? "" : c.ir_channel,
          label: c.label || "",
          gain_db: c.gain_db || 0,
          enabled: c.enabled
        };
      });
    }
    renderMatrix();
    el.editor.hidden = false;
    el.editor.scrollIntoView({ behavior: "smooth", block: "start" });
    el.devName.focus();
  }

  function closeEditor() {
    el.editor.hidden = true;
    editing = null;
    rowState = {};
  }

  function collectDevice() {
    var channels = [];
    Object.keys(rowState).forEach(function (ch) {
      var st = rowState[ch];
      if (!st.language) return;
      channels.push({
        language: st.language,
        channel: parseInt(ch, 10),
        ir_channel: st.ir_channel === "" || st.ir_channel === null ? null : parseInt(st.ir_channel, 10),
        label: st.label || "",
        gain_db: parseFloat(st.gain_db) || 0,
        enabled: st.enabled !== false
      });
    });
    return {
      name: el.devName.value.trim(),
      device_name: el.devDeviceName.value.trim(),
      kind: el.devKind.value,
      channel_count: parseInt(el.devChannelCount.value, 10),
      sample_rate: parseInt(el.devSampleRate.value, 10),
      enabled: el.devEnabled.checked,
      channels: channels
    };
  }

  function saveDevice() {
    var body = collectDevice();
    var url = BASE + "/api/outputs/devices" + (editing ? "/" + editing.device_id : "");
    el.saveDevice.disabled = true;
    // Whole map, every time. The server replaces it in one transaction, so
    // a dropped request leaves the previous patch intact rather than half
    // of this one.
    api(url, {
      method: editing ? "PUT" : "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body)
    }).then(function (d) {
      toast("Saved “" + d.name + "” with " + d.channels.length + " channel" +
        (d.channels.length === 1 ? "" : "s") + ".", "ok");
      closeEditor();
      return loadOutputs();
    }).catch(function (err) {
      // The 422s are the domain's own words - "two languages share a
      // channel on 'Main hall DVS': [2, 2]" - and are shown as such.
      toast(err.message, true);
    }).then(function () { el.saveDevice.disabled = false; });
  }

  function deleteDevice(device) {
    if (!window.confirm("Delete “" + device.name + "” and its whole channel map?")) return;
    api(BASE + "/api/outputs/devices/" + device.device_id, { method: "DELETE" })
      .then(function () {
        toast("Deleted “" + device.name + "”.", "ok");
        if (editing && editing.device_id === device.device_id) closeEditor();
        return loadOutputs();
      }).catch(function (err) { toast(err.message, true); });
  }

  // --- navigation ---------------------------------------------------------------

  function setActiveNav(id) {
    Array.prototype.forEach.call(document.querySelectorAll(".nav-link"), function (a) {
      a.classList.toggle("active", a.dataset.section === id);
    });
    var section = $(id);
    if (section) el.pageTitle.textContent = section.dataset.title || "Overview";
  }

  function watchSections() {
    if (!("IntersectionObserver" in window)) return;
    var visible = {};
    var observer = new IntersectionObserver(function (entries) {
      entries.forEach(function (e) { visible[e.target.id] = e.isIntersecting ? e.intersectionRatio : 0; });
      var best = null, ratio = 0;
      Object.keys(visible).forEach(function (id) { if (visible[id] > ratio) { ratio = visible[id]; best = id; } });
      if (best) setActiveNav(best);
    }, { rootMargin: "-40% 0px -50% 0px", threshold: [0, 0.25, 0.5, 1] });
    Array.prototype.forEach.call(document.querySelectorAll("main > section"), function (s) { observer.observe(s); });
  }

  // --- boot -------------------------------------------------------------------

  function load() {
    return Promise.all([api(BASE + "/api/presets"), api(BASE + "/api/settings")])
      .then(function (both) {
        renderPresets(both[0].presets);
        if (chosenPreset) selectPreset(chosenPreset);
        editable = both[1].editable;
        currentSettings = both[1].settings;
        publicBase = both[1].public_base;
        el.publicHost.textContent = publicBase.replace(/^https?:\/\//, "");
        el.publicHost.title = "Where phones reach the join service";
        el.build.textContent = "build " + (both[1].build || "—");
        renderAdvanced(currentSettings);
        markLivePad();
        fadersFollowArmed();
        lcdDefault();
        refreshQr();
      });
  }

  el.toggle.addEventListener("click", function () {
    var open = el.advanced.hidden;
    el.advanced.hidden = !open;
    el.toggle.textContent = open ? "Hide" : "Show";
    el.toggle.setAttribute("aria-expanded", String(open));
  });
  el.restart.addEventListener("click", apply);
  el.stop.addEventListener("click", stop);
  el.deckApply.addEventListener("click", apply);
  el.deckStop.addEventListener("click", stop);
  el.faderEmit.addEventListener("input", onFaderInput);
  el.faderWindow.addEventListener("input", onFaderInput);
  el.faderReset.addEventListener("click", fadersFollowArmed);
  el.room.addEventListener("change", function () { refreshQr(); refreshStatus(); });
  el.addDevice.addEventListener("click", function () { openEditor(null); });
  el.cancelDevice.addEventListener("click", closeEditor);
  el.cancelDeviceX.addEventListener("click", closeEditor);
  el.saveDevice.addEventListener("click", saveDevice);
  el.devChannelCount.addEventListener("change", renderMatrix);
  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape" && !el.editor.hidden) closeEditor();
  });
  Array.prototype.forEach.call(document.querySelectorAll("[data-copy]"), function (b) {
    b.addEventListener("click", function () { copyFrom(b.dataset.copy); });
  });
  Array.prototype.forEach.call(document.querySelectorAll(".nav-link"), function (a) {
    a.addEventListener("click", function () { setActiveNav(a.dataset.section); });
  });

  loadIdentity();
  load().then(refreshStatus).catch(function (err) { toast(err.message, true); });
  loadOutputs();
  watchSections();
  timer = setInterval(refreshStatus, 5000);
  window.addEventListener("beforeunload", function () { clearInterval(timer); });
})();
