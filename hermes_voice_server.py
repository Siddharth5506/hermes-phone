#!/usr/bin/env python3
"""
Hermes Phone - PC-side voice session server (Milestone 1).

Listens for the Android phone over WebSocket (LAN or Tailscale), receives
streamed PCM audio, runs STT, feeds Hermes the request, streams TTS back.

Reuses Hermes' existing infrastructure wherever possible:
  - tools.transcription_tools  -> speech-to-text
  - tools.tts_tool / tts_streaming -> text-to-speech
Falls back to a built-in pipeline if imports fail, so this server is
testable standalone before deep integration.

Run:  python hermes_voice_server.py [--port 8765]
"""
import argparse
import asyncio
import base64
import json
import logging
import struct
import sys
import time

try:
    import websockets
except ImportError:
    print("pip install websockets numpy first")
    sys.exit(1)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("hermes-voice")

SAMPLE_RATE = 16000

# ---------------------------------------------------------------------------
# Hermes integration layer — swap internals here without touching protocol.
# ---------------------------------------------------------------------------

def transcribe(pcm_bytes: bytes) -> str:
    """PCM16 16k mono -> text. Uses Hermes' transcription tooling if importable."""
    try:
        sys.path.insert(0, r"C:\Users\siddh\AppData\Local\hermes\hermes-agent")
        from tools.transcription_tools import transcribe_audio  # noqa
        # write temp wav and delegate
        import wave, tempfile, os
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm_bytes)
        try:
            return transcribe_audio(path) or ""
        finally:
            os.unlink(path)
    except Exception as e:
        log.warning("Hermes STT unavailable (%s); falling back", e)
        return fallback_transcribe(pcm_bytes)


def fallback_transcribe(pcm_bytes: bytes) -> str:
    """Local whisper.cpp/faster-whisper if installed; else empty."""
    try:
        from faster_whisper import WhisperModel
        global _whisper
        if "_whisper" not in globals():
            _whisper = WhisperModel("small", device="cpu", compute_type="int8")
        import tempfile, os, wave
        fd, path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        with wave.open(path, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm_bytes)
        try:
            segments, _ = _whisper.transcribe(path, language="en")
            return " ".join(s.text for s in segments).strip()
        finally:
            os.unlink(path)
    except ImportError:
        log.error("No STT backend available. pip install faster-whisper")
        return ""


def ask_hermes(text: str, device: str = "phone") -> str:
    """
    Send the user's request to the Hermes agent and get a short spoken reply.
    Milestone 1: routes through Hermes CLI in one-shot mode so we reuse the
    full agent (tools, memory, skills). Voice replies kept concise (spec §22).
    """
    prompt = (
        f"You are Jarvis, replying by VOICE from phone '{device}'. "
        f"Answer in at most two short sentences. No markdown. "
        f"User said: {text}"
    )
    try:
        import subprocess
        r = subprocess.run(
            ["hermes", "run", "--message", prompt],
            capture_output=True, text=True, timeout=60,
        )
        out = (r.stdout or "").strip()
        if out:
            return out[-800:]  # last lines = actual answer
    except Exception as e:
        log.error("hermes CLI failed: %s", e)
    return "Sorry, I couldn't reach my brain just now."


def tts_chunks(text: str):
    """Yield PCM16 16k mono byte chunks. Uses Hermes TTS if available."""
    try:
        sys.path.insert(0, r"C:\Users\siddh\AppData\Local\hermes\hermes-agent")
        from tools.tts_tool import text_to_speech_tool  # noqa
        import soundfile as sf  # noqa
        result = text_to_speech_tool(text=text)
        path = result.get("path") if isinstance(result, dict) else None
        if path and __import__("os").path.exists(path):
            data, sr = sf.read(path, dtype="int16")
            if sr != SAMPLE_RATE:
                # naive linear resample; good enough for M1, replace later
                import numpy as np
                x = np.linspace(0, len(data) - 1, int(len(data) * SAMPLE_RATE / sr))
                data = np.interp(x, range(len(data)), data).astype("int16")
            pcm = data.tobytes()
            for i in range(0, len(pcm), 3200):  # 100ms chunks
                yield pcm[i:i + 3200]
            return
    except Exception as e:
        log.warning("Hermes TTS unavailable (%s)", e)
    yield b""  # nothing to play


# ---------------------------------------------------------------------------
# WebSocket protocol (mirrors android/app HermesConnection.kt)
# ---------------------------------------------------------------------------

async def handle(ws):
    peer = getattr(ws, "remote_address", "?")
    log.info("phone connected: %s", peer)
    await ws.send(json.dumps({"type": "welcome"}))
    audio_buf = bytearray()
    device = "phone"

    async for raw in ws:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue
        t = msg.get("type")

        if t == "hello":
            device = msg.get("device", "phone")
            code = msg.get("pairing_code", "")
            log.info("hello from %s pairing=%s", device, code)
            # TODO(M4): validate against real pairing store; accept-all for M1 LAN trust
        elif t == "audio_start":
            audio_buf.clear()
            t0["speech_end"] = time.time()
        elif t == "audio":
            audio_buf += base64.b64decode(msg["pcm"])
        elif t == "audio_end":
            pcm = bytes(audio_buf)
            dur = len(pcm) / 2 / SAMPLE_RATE
            log.info("utterance %.1fs from %s", dur, device)
            await ws.send(json.dumps({"type": "transcript_partial", "text": "..."}))
            text = transcribe(pcm)
            if not text.strip():
                await ws.send(json.dumps({"type": "transcript_final", "text": ""}))
                await speak(ws, "I didn't catch that.")
                continue
            await ws.send(json.dumps({"type": "transcript_final", "text": text}))
            reply = ask_hermes(text, device)
            await ws.send(json.dumps({"type": "response_text", "text": reply}))
            await speak(ws, reply)
        elif t == "interrupt":
            pass  # M5: cancel in-flight generation


t0 = {}

async def speak(ws, text):
    first = True
    for chunk in tts_chunks(text):
        if not chunk:
            break
        await ws.send(json.dumps({
            "type": "tts_audio",
            "pcm": base64.b64encode(chunk).decode(),
        }))
        if first:
            log.info("first TTS chunk sent (%.0f ms after end of speech)",
                     (time.time() - t0.get("speech_end", 0)) * 1000)
            first = False
    await ws.send(json.dumps({"type": "tts_end"}))


async def main(port):
    # Accept any path (websockets>=14 no longer exposes ws.path the old way;
    # path filtering caused false rejects of valid phone connections)
    async with websockets.serve(handle, "0.0.0.0", port):
        log.info("Hermes voice server listening on ws://0.0.0.0:%d/voice", port)
        await asyncio.Future()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()
    asyncio.run(main(args.port))
