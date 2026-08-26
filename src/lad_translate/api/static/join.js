/* Listener join page.
 *
 * Two things here are load-bearing and easy to get wrong.
 *
 * 1. AUTOPLAY. iOS Safari and Android Chrome refuse to play audio that was not
 *    started by a user gesture. The tap on a language button is that gesture,
 *    so room.startAudio() is called inside the handler, before any await. Move
 *    it after an await and it is no longer inside the gesture: the room
 *    connects, tracks arrive, and the audience hears silence with no error.
 *
 * 2. SINGLE SUBSCRIPTION. autoSubscribe is off and exactly one track is
 *    subscribed. The default pulls every published track, so a listener on a
 *    five language event would download five audio streams instead of one:
 *    five times the bandwidth and battery, for four streams nobody hears. The
 *    architecture assumes about 50kbps per listener, and that assumption only
 *    holds if this stays off.
 */

(function () {
  "use strict";

  var LK = window.LivekitClient;
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
    event: document.getElementById("event"),
    subtitle: document.getElementById("subtitle"),
    picker: document.getElementById("picker"),
    languages: document.getElementById("languages"),
    playing: document.getElementById("playing"),
    dot: document.getElementById("dot"),
    statusText: document.getElementById("status-text"),
    nowLanguage: document.getElementById("now-language"),
    volume: document.getElementById("volume"),
    change: document.getElementById("change"),
    error: document.getElementById("error"),
    errorText: document.getElementById("error-text"),
    retry: document.getElementById("retry"),
    audio: document.getElementById("audio")
  };

  var room = null;
  var listenerId = null;
  var wantedTrack = null;
  var info = null;
  var audioTimer = null;
  var resolvedSession = "";

  // How long to wait for the chosen language's track before saying so.
  // A listener whose language is configured but not being published gets a
  // valid token and a working connection, and would otherwise sit on
  // 'waiting for audio' for the whole event with nothing to act on.
  var AUDIO_TIMEOUT_MS = 15000;

  // --- view helpers --------------------------------------------------------

  function show(section) {
    el.picker.hidden = section !== "picker";
    el.playing.hidden = section !== "playing";
    el.error.hidden = section !== "error";
  }

  function status(text, state) {
    el.statusText.textContent = text;
    el.dot.className = "dot" + (state ? " " + state : "");
  }

  function fail(message) {
    el.errorText.textContent = message;
    show("error");
  }

  // --- load ----------------------------------------------------------------

  function load() {
    show(null);
    el.subtitle.textContent = "Loading…";
    fetch(apiBase())
      .then(function (r) {
        if (r.status === 404) throw new Error("This session was not found. Check the QR code.");
        if (r.status === 410) throw new Error("This session has ended.");
        if (!r.ok) throw new Error("Could not load the session. Please try again.");
        return r.json();
      })
      .then(function (data) {
        info = data;
        el.event.textContent = data.event_name || "Live Translation";
        el.subtitle.textContent = "Choose a language to start listening";
        renderLanguages(data.languages || []);
        show("picker");
      })
      .catch(function (err) { fail(err.message); });
  }

  function renderLanguages(languages) {
    el.languages.innerHTML = "";
    if (!languages.length) {
      fail("No languages are available for this session yet.");
      return;
    }
    languages.forEach(function (lang) {
      var button = document.createElement("button");
      button.className = "lang" + (lang.available === false ? " unavailable" : "");
      button.type = "button";
      if (lang.rtl) button.setAttribute("dir", "rtl");

      var native = document.createElement("span");
      native.textContent = lang.native;
      var english = document.createElement("span");
      english.className = "english" + (lang.is_source ? " source" : "");
      // Still selectable when unavailable: a session that is still starting up
      // has not published its tracks yet, and blocking the choice would be
      // wrong more often than it would be right.
      english.textContent = lang.available === false ? "not yet on air" : lang.english;

      button.appendChild(native);
      button.appendChild(english);
      // Not addEventListener with await inside: see the autoplay note above.
      button.addEventListener("click", function () { choose(lang); });
      el.languages.appendChild(button);
    });
  }

  // --- joining -------------------------------------------------------------

  function choose(lang) {
    show("playing");
    el.nowLanguage.textContent = lang.native;
    el.nowLanguage.setAttribute("dir", lang.rtl ? "rtl" : "ltr");
    // The original is relayed straight from the room, so it arrives without
    // the transcription and synthesis delay the translations carry.
    el.subtitle.textContent = lang.is_source
      ? "Original audio, relayed live"
      : "Translated live";
    status("Connecting…", "connecting");

    // Inside the gesture, synchronously. Unlocks the element for iOS before
    // any network call has a chance to end the gesture's privileged window.
    el.audio.muted = false;
    var unlock = el.audio.play();
    if (unlock && unlock.catch) unlock.catch(function () { /* nothing to play yet */ });

    connect(lang).catch(function (err) {
      fail(err && err.message ? err.message : "Could not connect. Please try again.");
    });
  }

  function connect(lang) {
    return fetch(apiBase() + "/join", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ language: lang.code })
    })
      .then(function (r) {
        if (r.status === 410) throw new Error("This session has ended.");
        if (!r.ok) throw new Error("Could not join. Please try again.");
        return r.json();
      })
      .then(function (grant) {
        listenerId = grant.listener_id;
        wantedTrack = grant.track_name;
        // Under a /room/ URL the page never saw a session id. The API resolves
        // it, so keep what it returns for the leave call.
        resolvedSession = grant.session_id || (info && info.session_id) || "";

        room = new LK.Room({
          // Nothing is published from a phone, so no capture defaults are set.
          adaptiveStream: false,
          dynacast: false
        });
        wireEvents();

        return room.connect(grant.url, grant.token, {
          // See the single subscription note at the top. Leaving this on would
          // quietly multiply every listener's bandwidth by the language count.
          autoSubscribe: false
        });
      })
      .then(function () {
        // startAudio must still be reachable from the original gesture on
        // iOS. It is called again on the resume button if the browser
        // decides otherwise.
        return room.startAudio().catch(function () { blockedPlayback(); });
      })
      .then(function () {
        subscribeToWanted();
        status("Connected, waiting for audio…", "warn");
        clearTimeout(audioTimer);
        audioTimer = setTimeout(function () {
          fail(
            "Connected, but no audio is being sent for this language yet. " +
            "It may not have started. Try another language, or try again shortly."
          );
        }, AUDIO_TIMEOUT_MS);
      });
  }

  function wireEvents() {
    room
      .on(LK.RoomEvent.TrackPublished, function () { subscribeToWanted(); })
      .on(LK.RoomEvent.TrackSubscribed, function (track, publication) {
        if (publication.trackName !== wantedTrack) {
          // Should not happen with autoSubscribe off, but an unexpected track
          // is bandwidth nobody asked for. Drop it rather than play it.
          publication.setSubscribed(false);
          return;
        }
        clearTimeout(audioTimer);
        track.attach(el.audio);
        el.audio.volume = el.volume.value / 100;
        status("Listening", "live");
      })
      .on(LK.RoomEvent.TrackUnsubscribed, function (track) {
        track.detach(el.audio);
        status("Audio stopped", "warn");
      })
      .on(LK.RoomEvent.Reconnecting, function () {
        status("Reconnecting…", "connecting");
      })
      .on(LK.RoomEvent.Reconnected, function () {
        // Subscriptions do not always survive a reconnect, so re-assert.
        subscribeToWanted();
        status("Listening", "live");
      })
      .on(LK.RoomEvent.Disconnected, function () {
        status("Disconnected", "bad");
      })
      .on(LK.RoomEvent.AudioPlaybackStatusChanged, function () {
        if (!room.canPlaybackAudio) blockedPlayback();
      });
  }

  function subscribeToWanted() {
    if (!room) return;
    room.remoteParticipants.forEach(function (participant) {
      participant.trackPublications.forEach(function (publication) {
        if (publication.trackName === wantedTrack && !publication.isSubscribed) {
          publication.setSubscribed(true);
        }
      });
    });
  }

  function blockedPlayback() {
    // The browser refused playback despite the gesture. Give the listener an
    // obvious second chance rather than leaving them in silence.
    status("Tap anywhere to start audio", "warn");
    var resume = function () {
      if (room) room.startAudio();
      el.audio.play().catch(function () {});
      document.removeEventListener("click", resume);
      document.removeEventListener("touchend", resume);
    };
    document.addEventListener("click", resume);
    document.addEventListener("touchend", resume);
  }

  // --- controls ------------------------------------------------------------

  el.volume.addEventListener("input", function () {
    el.audio.volume = el.volume.value / 100;
  });

  el.change.addEventListener("click", function () {
    leave();
    disconnect();
    show("picker");
  });

  el.retry.addEventListener("click", function () {
    disconnect();
    load();
  });

  function disconnect() {
    clearTimeout(audioTimer);
    if (room) { room.disconnect(); room = null; }
    listenerId = null;
    wantedTrack = null;
  }

  function leave() {
    if (!listenerId) return;
    // Beacon rather than fetch: a normal request is cancelled when the page
    // goes away. Best effort even so, so listener counts from this are a
    // floor rather than a census.
    if (!resolvedSession) return;
    var url = "/api/listeners/" + encodeURIComponent(listenerId) +
              "/leave?session_id=" + encodeURIComponent(resolvedSession);
    if (navigator.sendBeacon) navigator.sendBeacon(url, new Blob());
  }

  // pagehide fires on iOS where unload does not.
  window.addEventListener("pagehide", leave);
  window.addEventListener("beforeunload", leave);

  load();
})();
