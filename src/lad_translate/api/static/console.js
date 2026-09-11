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
    urlListen: document.getElementById("url-listen"),
    outPill: document.getElementById("outputs-pill"),
    outState: document.getElementById("outputs-state"),
    devices: document.getElementById("devices"),
    addDevice: document.getElementById("add-device"),
    editor: document.getElementById("device-editor"),
    editorTitle: document.getElementById("editor-title"),
    devName: document.getElementById("dev-name"),
    devDeviceName: document.getElementById("dev-device-name"),
    devKind: document.getElementById("dev-kind"),
    devChannelCount: document.getElementById("dev-channel-count"),
    devSampleRate: document.getElementById("dev-sample-rate"),
    devEnabled: document.getElementById("dev-enabled"),
    matrix: document.getElementById("matrix"),
    saveDevice: document.getElementById("save-device"),
    cancelDevice: document.getElementById("cancel-device")
  };

  // Absolute and prefixed, matching where the app actually serves. Relative
  // would depend on whether the browser landed on /console or /console/.
  var BASE = "/console";

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
    api(BASE + "/api/status?room=" + encodeURIComponent(room())).then(function (s) {
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
    }).catch(function (err) { toast(err.message, true); });
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
    api(BASE + "/api/apply", {
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
    api(BASE + "/api/stop", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ room: room() })
    }).then(function () {
      toast("Stopped.");
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

  function setOutputsState(pillText, pillClass, message) {
    el.outPill.textContent = pillText;
    el.outPill.className = "pill " + (pillClass || "");
    if (message) { el.outState.textContent = message; el.outState.hidden = false; }
    else { el.outState.hidden = true; }
  }

  function renderDevices() {
    el.devices.innerHTML = "";
    outputs.devices.forEach(function (d) {
      var card = document.createElement("div");
      card.className = "device";

      var h = document.createElement("h3");
      h.textContent = d.name;
      var pill = document.createElement("span");
      pill.className = "pill " + (d.enabled ? "on" : "off");
      pill.textContent = d.enabled ? "enabled" : "disabled";
      h.appendChild(pill);
      card.appendChild(h);

      var meta = document.createElement("p");
      meta.className = "meta";
      meta.textContent = d.kind + " · " + d.device_name + " · " + d.channel_count +
        " channels · " + d.sample_rate + " Hz";
      card.appendChild(meta);

      var map = document.createElement("ul");
      map.className = "map";
      if (!d.channels.length) {
        var none = document.createElement("li");
        none.textContent = "no channels assigned";
        map.appendChild(none);
      }
      d.channels.forEach(function (c) {
        var li = document.createElement("li");
        var text = "ch " + c.channel + " → " + languageName(c.language);
        if (c.gain_db) text += " " + (c.gain_db > 0 ? "+" : "") + c.gain_db + " dB";
        if (c.label) text += " “" + c.label + "”";
        var span = document.createElement("span");
        if (!c.enabled) span.className = "off";
        span.textContent = text;
        li.appendChild(span);
        if (c.ir_channel !== null) {
          var ir = document.createElement("span");
          ir.className = "ir";
          ir.textContent = "  IR " + c.ir_channel;
          li.appendChild(ir);
        }
        map.appendChild(li);
      });
      card.appendChild(map);

      var actions = document.createElement("div");
      actions.className = "actions";
      var edit = document.createElement("button");
      edit.textContent = "Edit";
      edit.addEventListener("click", function () { openEditor(d); });
      var signage = document.createElement("button");
      signage.textContent = "Signage";
      signage.title = "The IR channel list, as text, for printing";
      signage.addEventListener("click", function () {
        window.open(BASE + "/api/outputs/devices/" + d.device_id + "/signage", "_blank");
      });
      var del = document.createElement("button");
      del.textContent = "Delete";
      del.addEventListener("click", function () { deleteDevice(d); });
      actions.appendChild(edit); actions.appendChild(signage); actions.appendChild(del);
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
      opt.value = String(v); opt.textContent = String(v);
      if (String(v) === String(current)) opt.selected = true;
      select.appendChild(opt);
    });
  }

  function renderMatrix() {
    var count = Math.max(1, Math.min(64, parseInt(el.devChannelCount.value, 10) || 16));
    el.matrix.innerHTML = "";
    for (var ch = 1; ch <= count; ch++) {
      (function (ch) {
        var st = rowState[ch] || { language: "", ir_channel: "", label: "", gain_db: 0, enabled: true };
        var tr = document.createElement("tr");
        if (st.language) tr.className = "assigned";

        var tdCh = document.createElement("td"); tdCh.className = "ch"; tdCh.textContent = ch;
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
          rowState[ch] = rowState[ch] || { ir_channel: "", label: "", gain_db: 0, enabled: true };
          rowState[ch].language = sel.value;
          tr.className = sel.value ? "assigned" : "";
        });
        tdLang.appendChild(sel); tr.appendChild(tdLang);

        function numberCell(key, min, max, step, cls) {
          var td = document.createElement("td");
          var input = document.createElement("input");
          input.type = "number"; input.min = min; input.max = max; input.step = step;
          input.className = cls;
          input.value = st[key] === null || st[key] === undefined ? "" : st[key];
          input.addEventListener("input", function () {
            rowState[ch] = rowState[ch] || { language: "", ir_channel: "", label: "", gain_db: 0, enabled: true };
            rowState[ch][key] = input.value;
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
          rowState[ch] = rowState[ch] || { language: "", ir_channel: "", label: "", gain_db: 0, enabled: true };
          rowState[ch].label = label.value;
        });
        tdLabel.appendChild(label); tr.appendChild(tdLabel);

        tr.appendChild(numberCell("gain_db", -60, 12, 0.5, "narrow"));

        var tdOn = document.createElement("td");
        var on = document.createElement("input");
        on.type = "checkbox"; on.checked = st.enabled !== false;
        on.addEventListener("change", function () {
          rowState[ch] = rowState[ch] || { language: "", ir_channel: "", label: "", gain_db: 0, enabled: true };
          rowState[ch].enabled = on.checked;
        });
        tdOn.appendChild(on); tr.appendChild(tdOn);

        el.matrix.appendChild(tr);
      })(ch);
    }
  }

  function openEditor(device) {
    editing = device || null;
    rowState = {};
    el.editorTitle.textContent = device ? "Edit “" + device.name + "”" : "New device";
    el.devName.value = device ? device.name : "";
    el.devDeviceName.value = device ? device.device_name : "";
    fillSelect(el.devKind, outputs.kinds, device ? device.kind : "dante-vsc");
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
      var row = {
        language: st.language,
        channel: parseInt(ch, 10),
        ir_channel: st.ir_channel === "" || st.ir_channel === null ? null : parseInt(st.ir_channel, 10),
        label: st.label || "",
        gain_db: parseFloat(st.gain_db) || 0,
        enabled: st.enabled !== false
      };
      channels.push(row);
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
        (d.channels.length === 1 ? "" : "s") + ".");
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
    fetch(BASE + "/api/outputs/devices/" + device.device_id, { method: "DELETE" })
      .then(function (r) {
        if (!r.ok) {
          return r.json().catch(function () { return {}; }).then(function (b) {
            throw new Error(b.detail || ("delete failed: " + r.status));
          });
        }
        toast("Deleted “" + device.name + "”.");
        if (editing && editing.device_id === device.device_id) closeEditor();
        return loadOutputs();
      }).catch(function (err) { toast(err.message, true); });
  }

  el.addDevice.addEventListener("click", function () { openEditor(null); });
  el.cancelDevice.addEventListener("click", closeEditor);
  el.saveDevice.addEventListener("click", saveDevice);
  el.devChannelCount.addEventListener("change", renderMatrix);

  // --- boot -----------------------------------------------------------------

  function load() {
    return Promise.all([api(BASE + "/api/presets"), api(BASE + "/api/settings")])
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
  loadOutputs();
  timer = setInterval(refreshStatus, 5000);
  window.addEventListener("beforeunload", function () { clearInterval(timer); });
})();
