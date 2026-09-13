"""
A session's whole transcript, in the formats someone actually asks for.

The console's live panel is for reading along. This is for afterwards: the
client who wants the Arabic checked by a human, the producer cutting
subtitles onto the recording, the archive that has to hold what was said.

FOUR FORMATS, and the reason each exists.

  txt   One file, everything: source line, then each language under it.
        What you paste into an email.
  json  The rows as they are stored, timings and latencies included. What
        another program reads.
  srt   Subtitles, one language per file, timed from the audio positions.
  vtt   The same for the web.

SUBTITLES LINE UP WITH source.wav, NOT WITH THE TRANSLATED TRACKS. The
timings here are t_audio - a position in the speaker's audio - so an SRT
dropped onto source.wav from a recording is correct. The translated WAVs in
that recording sit where each phrase was handed to playout, which is a
second or so earlier than the audience heard it and is not the same clock.
Said here because a producer will otherwise discover it by eye.

NO AUTHENTICATION HERE. These routes are mounted on the console, behind the
same Google sign-in as everything else. A transcript is the content of
someone's talk; it does not get a door of its own that is easier to open.
"""

from __future__ import annotations

import io
import json
from datetime import UTC, datetime

from ..api.languages import describe

FORMATS = ("txt", "json", "srt", "vtt")


def _clock(seconds: float, comma: bool = True) -> str:
    """SRT wants 00:00:01,250; WebVTT wants 00:00:01.250."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(int(seconds), 3600)
    minutes, whole = divmod(rest, 60)
    millis = round((seconds - int(seconds)) * 1000)
    if millis == 1000:                      # 1.9996 rounds to a whole second
        whole, millis = whole + 1, 0
    sep = "," if comma else "."
    return f"{hours:02d}:{minutes:02d}:{whole:02d}{sep}{millis:03d}"


def by_chunk(rows) -> list[dict]:
    """Rows to one entry per chunk, source plus each language."""
    chunks: dict[int, dict] = {}
    for row in rows:
        entry = chunks.setdefault(
            row["chunk_id"],
            {
                "chunk_id": row["chunk_id"],
                "source": row["source_text"],
                "start": float(row["t_audio_start"]),
                "end": float(row["t_audio_end"]),
                "languages": {},
            },
        )
        entry["languages"][row["language"]] = {
            "text": row["translated_text"],
            "latency_s": float(row["latency_s"]) if row["latency_s"] is not None else None,
        }
    return [chunks[k] for k in sorted(chunks)]


def as_text(session: dict, rows) -> str:
    chunks = by_chunk(rows)
    languages = sorted({lang for c in chunks for lang in c["languages"]})
    out = io.StringIO()
    out.write(f"{session.get('event_name') or 'Live Translation'}\n")
    out.write(f"room {session.get('room_name') or ''}   session {session['session_id']}\n")
    started = session.get("started_at")
    if started:
        out.write(f"started {started}\n")
    names = ", ".join(f"{describe(c).english} ({c})" for c in languages)
    out.write(f"languages {names or 'none'}\n")
    out.write(f"exported {datetime.now(UTC).isoformat(timespec='seconds')}\n")
    out.write("\n" + "-" * 72 + "\n\n")
    if not chunks:
        out.write("This session produced no transcript.\n")
        return out.getvalue()
    for c in chunks:
        out.write(f"[{_clock(c['start'], comma=False)}]  {c['source']}\n")
        for code in sorted(c["languages"]):
            text = c["languages"][code]["text"]
            out.write(f"    {code.upper():<4s}{text if text else '- no translation'}\n")
        out.write("\n")
    return out.getvalue()


def as_json(session: dict, rows) -> str:
    return json.dumps(
        {
            "session_id": session["session_id"],
            "room": session.get("room_name"),
            "event_name": session.get("event_name"),
            "status": session.get("status"),
            "started_at": str(session["started_at"]) if session.get("started_at") else None,
            "ended_at": str(session["ended_at"]) if session.get("ended_at") else None,
            "exported_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "timing": (
                "t_audio seconds from the start of the speaker's audio; these line up "
                "with source.wav in a recording, not with the translated tracks"
            ),
            "chunks": by_chunk(rows),
        },
        ensure_ascii=False,
        indent=2,
    )


def as_subtitles(rows, language: str, vtt: bool = False) -> str:
    """
    One language as SRT or WebVTT.

    `language` may be the source, spelled "source" or "en": a producer
    cutting the original wants the speaker's own words with the same
    timings, and having to export that from somewhere else would be silly.
    """
    chunks = by_chunk(rows)
    out = io.StringIO()
    if vtt:
        out.write("WEBVTT\n\n")
    index = 0
    for c in chunks:
        text = c["source"] if language == "source" else (c["languages"].get(language) or {}).get("text")
        if not text:
            continue
        index += 1
        # A chunk whose end is not after its start would make a cue no player
        # shows; give it a readable minimum rather than dropping the line.
        end = max(c["end"], c["start"] + 0.5)
        if not vtt:
            out.write(f"{index}\n")
        out.write(f"{_clock(c['start'], comma=not vtt)} --> {_clock(end, comma=not vtt)}\n")
        out.write(f"{text}\n\n")
    return out.getvalue()


def render(session: dict, rows, fmt: str, language: str | None) -> tuple[str, str, str]:
    """Returns (body, media type, filename)."""
    stamp = str(session.get("started_at") or "")[:19].replace(" ", "T").replace(":", "")
    base = f"{session.get('room_name') or 'room'}-{stamp or session['session_id'][:8]}"
    if fmt == "json":
        return as_json(session, rows), "application/json; charset=utf-8", f"{base}.json"
    if fmt in ("srt", "vtt"):
        code = language or "source"
        body = as_subtitles(rows, code, vtt=(fmt == "vtt"))
        media = "text/vtt; charset=utf-8" if fmt == "vtt" else "application/x-subrip; charset=utf-8"
        return body, media, f"{base}-{code}.{fmt}"
    return as_text(session, rows), "text/plain; charset=utf-8", f"{base}.txt"
