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
  var sessionId = location.pathname.split("/").filter(Boolean).pop();

  var el = {
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
    fetch("/api/sessions/" + encodeURIComponent(sessionId))
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
        show("start");
      })
      .catch(function (e) { fail(e.message); });
  }

  // --- speaking -----------------------------------------------------------

  el.go.addEventListener("click", function () {
    show("live");
    status("Asking for the microphone…", "connecting");

    // Inside the gesture, before any await. echoCancellation is on because a
    // phone that is also playing a translation would otherwise feed back into
    // its own microphone. A venue takes a desk send and needs none of this.
    navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
        channelCount: 1
      }
    }).then(publish).catch(function (err) {
      if (err && err.name === "NotAllowedError") {
        fail("Microphone permission was refused. Allow it in your browser settings and try again.");
      } else if (err && err.name === "NotFoundError") {
        fail("No microphone found on this device.");
      } else {
        fail("Could not open the microphone: " + (err && err.name ? err.name : "unknown error"));
      }
    });
  });

  function publish(mediaStream) {
    stream = mediaStream;
    startMeter(mediaStream);
    status("Connecting…", "connecting");

    fetch("/api/sessions/" + encodeURIComponent(sessionId) + "/speak", { method: "POST" })
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
