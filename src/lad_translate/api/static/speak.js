/* Speaker page: publishes this phone's microphone as the session's source.
 *
 * Two rules decide whether this works at all.
 *
 * 1. SECURE CONTEXT. navigator.mediaDevices is undefined on plain http from
 *    anything except localhost. A phone on http://192.168.x.x has no mic API
 *    to call, so this page must be served over https. That is why there is a
 *    TLS proxy in front of the service at all.
 *
 * 2. USER GESTURE. getUserMedia must be called from inside a tap handler.
 *    Move it after an await and the permission prompt is suppressed on iOS.
 */

(function () {
  "use strict";

  var LK = window.LivekitClient;

  // Which input to open. A phone has one and never sees this; a laptop at a
  // venue has several, and its default is the built-in microphone - the one
  // pointing at the room rather than the desk send patched into its sound
  // card. Remembered per device so the rig comes back the same tomorrow.
  var INPUT_KEY = "lad.speaker.input";
  var chosenInput = null;
  try { chosenInput = window.localStorage.getItem(INPUT_KEY); } catch (e) { chosenInput = null; }
  // The page is served at two URL shapes and must talk to the matching API:
  //
  //   /s/<session-id>      -> /api/sessions/<session-id>
  //   /room/<room-name>    -> /api/rooms/<room-name>
  //
  // Room URLs are the ones that go on printed material: a session id changes
  // on every restart, a room name does not.
  function apiBase() {
    var parts = location.pathname.split("/").filter(Boolean);
    if (parts[0] === "room") return "/api/rooms/" + encodeURIComponent(parts[1]);
    return "/api/sessions/" + encodeURIComponent(parts[parts.length - 1]);
  }


  var el = {
    inputPick: document.getElementById("input-pick"),
    inputDevice: document.getElementById("input-device"),
    inputList: document.getElementById("input-list"),
    inputHint: document.getElementById("input-hint"),
    goSub: document.getElementById("go-sub"),
    event: document.getElementById("event"),
    subtitle: document.getElementById("subtitle"),
    start: document.getElementById("start"),
    go: document.getElementById("go"),
    live: document.getElementById("live"),
    dot: document.getElementById("dot"),
    statusText: document.getElementById("status-text"),
    bar: document.getElementById("bar"),
    levelNote: document.getElementById("level-note"),
    listeners: document.getElementById("listeners"),
    sent: document.getElementById("sent"),
    stop: document.getElementById("stop"),
    error: document.getElementById("error"),
    errorText: document.getElementById("error-text"),
    retry: document.getElementById("retry")
  };

  var room = null;
  var stream = null;
  var meter = null;
  var startedAt = 0;
  var ticker = null;
  var sawSound = false;

  function show(which) {
    el.start.hidden = which !== "start";
    el.live.hidden = which !== "live";
    el.error.hidden = which !== "error";
  }

  function status(text, state) {
    el.statusText.textContent = text;
    el.dot.className = "dot" + (state ? " " + state : "");
  }

  function fail(message) {
    el.errorText.textContent = message;
    show("error");
  }

  // --- load ---------------------------------------------------------------

  function load() {
    show(null);
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      // Almost always the secure-context rule rather than an old browser, and
      // saying so saves someone half an hour.
      fail(
        "This browser will not give a page the microphone over an insecure " +
        "connection. Open this page over https and try again."
      );
      return;
    }
    fetch(apiBase())
      .then(function (r) {
        if (r.status === 404) throw new Error("Session not found. Check the link.");
        if (r.status === 410) throw new Error("This session has ended.");
        if (!r.ok) throw new Error("Could not load the session.");
        return r.json();
      })
      .then(function (info) {
        el.event.textContent = info.event_name || "Speaker";
        var others = (info.languages || []).filter(function (l) { return !l.is_source; });
        el.subtitle.textContent = others.length
          ? "Translating into " + others.map(function (l) { return l.native; }).join(", ")
          : "No translation languages configured yet";
        // Said before the microphone opens, not after. A recording the
        // speaker learns about later is a complaint; one they were told
        // about first is a feature.
        if (info.recording) {
          el.subtitle.textContent += " · This session is being recorded.";
        }
        show("start");
        listInputs();
      })
      .catch(function (e) { fail(e.message); });
  }

  // --- choosing an input ----------------------------------------------------

  function isDefaultish(device) {
    // Chrome lists synthetic "default"/"communications" entries that follow
    // the OS. Real hardware is what a venue wants to pin.
    return device.deviceId === "default" || device.deviceId === "communications";
  }

  function renderInputs(devices) {
    el.inputDevice.innerHTML = "";
    var auto = document.createElement("option");
    auto.value = "";
    auto.textContent = "Default input";
    el.inputDevice.appendChild(auto);

    var named = 0;
    devices.forEach(function (d) {
      // Without permission a browser reports inputs with no label AND no
      // deviceId. An option that cannot be selected is noise, so those are
      // left out and the "List inputs" button is what the operator sees.
      if (d.kind !== "audioinput" || isDefaultish(d) || !d.deviceId) return;
      var opt = document.createElement("option");
      opt.value = d.deviceId;
      opt.textContent = d.label || ("Input " + (named + 1));
      if (d.label) named++;
      if (d.deviceId === chosenInput) opt.selected = true;
      el.inputDevice.appendChild(opt);
    });

    // Hide it only when we KNOW there is nothing to choose: labels readable
    // and exactly one input. Without permission a browser reports one
    // unnamed input whether it is a phone with one microphone or a Mac with
    // a sound card, and hiding on that guess would hide the control that
    // unlocks the names.
    var inputs = devices.filter(function (d) {
      return d.kind === "audioinput" && !isDefaultish(d) && d.deviceId;
    });
    el.inputPick.hidden = named > 0 && inputs.length <= 1;
    el.inputList.hidden = named > 0;
    el.inputHint.textContent = named
      ? "Pick the desk send rather than the built-in microphone. Processing is left on for a phone mic and turned off for anything else."
      : "Allow the microphone once to see input names.";
    updateGoSub();
  }

  function updateGoSub() {
    var opt = el.inputDevice.options[el.inputDevice.selectedIndex];
    var picked = opt && opt.value;
    el.goSub.textContent = picked ? "uses " + opt.textContent : "uses this device's default input";
  }

  function listInputs() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.enumerateDevices) return;
    navigator.mediaDevices.enumerateDevices().then(renderInputs, function () {});
  }

  function unlockInputNames() {
    // A gesture, so the permission prompt is allowed. The stream is released
    // immediately: this is only to learn the labels.
    navigator.mediaDevices.getUserMedia({ audio: true }).then(function (s) {
      s.getTracks().forEach(function (t) { t.stop(); });
      listInputs();
    }, function () {
      el.inputHint.textContent = "Microphone permission was refused, so inputs cannot be listed.";
    });
  }

  function audioConstraints() {
    var opt = el.inputDevice.options[el.inputDevice.selectedIndex];
    var id = opt && opt.value;
    if (!id) {
      // The phone case, unchanged: a handset playing a translation into the
      // room would otherwise feed back into its own microphone.
      return { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 };
    }
    // A chosen device is a desk send or a sound card. Every one of those
    // cures hurts it: AGC pumps on a mixed feed and noise suppression eats
    // the tail of a sentence.
    return {
      deviceId: { exact: id },
      echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: 1
    };
  }

  el.inputDevice.addEventListener("change", function () {
    chosenInput = el.inputDevice.value || null;
    try {
      if (chosenInput) window.localStorage.setItem(INPUT_KEY, chosenInput);
      else window.localStorage.removeItem(INPUT_KEY);
    } catch (e) { /* private browsing; the choice just will not persist */ }
    updateGoSub();
  });
  el.inputList.addEventListener("click", unlockInputNames);

  // --- speaking -----------------------------------------------------------

  el.go.addEventListener("click", function () {
    show("live");
    status("Asking for the microphone…", "connecting");

    // Inside the gesture, before any await. echoCancellation is on because a
    // phone that is also playing a translation would otherwise feed back into
    // its own microphone. A venue takes a desk send and needs none of this.
    navigator.mediaDevices.getUserMedia({ audio: audioConstraints() })
      .then(function (mediaStream) {
        listInputs();   // labels are readable now that permission is granted
        return publish(mediaStream);
      }).catch(function (err) {
      if (err && err.name === "NotAllowedError") {
        fail("Microphone permission was refused. Allow it in your browser settings and try again.");
      } else if (err && err.name === "NotFoundError") {
        fail("No microphone found on this device.");
      } else if (err && (err.name === "OverconstrainedError" || err.name === "NotReadableError")) {
        // The remembered device is gone, or something else holds it open.
        fail("That input is not available: " + err.name +
             ". Pick another in the Input list and try again.");
        listInputs();
      } else {
        fail("Could not open the microphone: " + (err && err.name ? err.name : "unknown error"));
      }
    });
  });

  function publish(mediaStream) {
    stream = mediaStream;
    startMeter(mediaStream);
    status("Connecting…", "connecting");

    fetch(apiBase() + "/speak", { method: "POST" })
      .then(function (r) {
        if (r.status === 409) throw new Error("Someone is already speaking in this session.");
        if (r.status === 410) throw new Error("This session has ended.");
        if (!r.ok) throw new Error("Could not join as the speaker.");
        return r.json();
      })
      .then(function (grant) {
        room = new LK.Room();
        room
          .on(LK.RoomEvent.Disconnected, function () { status("Disconnected", "bad"); })
          .on(LK.RoomEvent.Reconnecting, function () { status("Reconnecting…", "connecting"); })
          .on(LK.RoomEvent.Reconnected, function () { status("Live", "live"); })
          .on(LK.RoomEvent.ParticipantConnected, updateListeners)
          .on(LK.RoomEvent.ParticipantDisconnected, updateListeners);

        // A speaker publishes and never subscribes. Subscribing would pull
        // every translated track back down the same connection carrying the
        // one that matters, and on a phone that is battery and bandwidth for
        // audio nobody is listening to.
        return room.connect(grant.url, grant.token, { autoSubscribe: false })
          .then(function () {
            var track = new LK.LocalAudioTrack(mediaStream.getAudioTracks()[0]);
            return room.localParticipant.publishTrack(track, {
              name: grant.track_name,
              source: LK.Track.Source.Microphone,
              dtx: false,          // never gate the speaker's own voice
              red: true            // redundancy: venue wifi drops packets
            });
          });
      })
      .then(function () {
        status("Live", "live");
        startedAt = Date.now();
        ticker = setInterval(tick, 1000);
        updateListeners();
      })
      .catch(function (e) { cleanup(); fail(e.message); });
  }

  function tick() {
    var seconds = Math.floor((Date.now() - startedAt) / 1000);
    el.sent.textContent = seconds < 60
      ? seconds + "s"
      : Math.floor(seconds / 60) + "m " + (seconds % 60) + "s";
  }

  function updateListeners() {
    if (!room) return;
    el.listeners.textContent = room.remoteParticipants ? room.remoteParticipants.size : 0;
  }

  // --- level meter --------------------------------------------------------

  function startMeter(mediaStream) {
    var Ctx = window.AudioContext || window.webkitAudioContext;
    if (!Ctx) return;
    var ctx = new Ctx();
    var source = ctx.createMediaStreamSource(mediaStream);
    var analyser = ctx.createAnalyser();
    analyser.fftSize = 512;
    source.connect(analyser);
    var data = new Uint8Array(analyser.fftSize);

    meter = { ctx: ctx, raf: 0 };
    (function frame() {
      analyser.getByteTimeDomainData(data);
      var peak = 0;
      for (var i = 0; i < data.length; i++) {
        var v = Math.abs(data[i] - 128) / 128;
        if (v > peak) peak = v;
      }
      // Perceptual rather than linear: a linear bar barely moves for speech.
      var pct = Math.min(100, Math.round(Math.sqrt(peak) * 130));
      el.bar.style.width = pct + "%";
      el.bar.className = "meter-fill" + (pct > 92 ? " hot" : "");
      if (pct > 12 && !sawSound) {
        sawSound = true;
        el.levelNote.textContent = "Microphone is picking you up.";
      }
      meter.raf = requestAnimationFrame(frame);
    })();
  }

  // --- teardown -----------------------------------------------------------

  function cleanup() {
    if (ticker) { clearInterval(ticker); ticker = null; }
    if (meter) {
      cancelAnimationFrame(meter.raf);
      if (meter.ctx && meter.ctx.close) meter.ctx.close();
      meter = null;
    }
    if (room) { room.disconnect(); room = null; }
    if (stream) {
      // Release the mic properly. Left open, iOS keeps the recording
      // indicator lit and the page holds the device.
      stream.getTracks().forEach(function (t) { t.stop(); });
      stream = null;
    }
  }

  el.stop.addEventListener("click", function () { cleanup(); show("start"); });
  el.retry.addEventListener("click", function () { cleanup(); load(); });
  window.addEventListener("pagehide", cleanup);

  load();
})();
