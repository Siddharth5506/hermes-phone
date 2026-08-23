# Hermes Phone

Android voice bridge for the Hermes AI agent — "Jarvis" wake word, on-device
detection (openWakeWord), streamed voice to the Hermes PC.

## Components

| Path | What |
|---|---|
| `android/` | The phone app (Kotlin). Wake word -> mic streaming -> spoken replies. |
| `.github/workflows/android-build.yml` | Builds the APK in GitHub Actions cloud — **no Android Studio needed**. |
| `hermes_voice_server.py` | PC-side WebSocket server: STT -> Hermes agent -> streaming TTS. |

## Install

### 1. Get the APK
GitHub repo → **Actions** → latest "Build Hermes Phone APK" run → download
`hermes-phone-debug` artifact → unzip → copy APK to phone → install
(allow "install unknown apps" when prompted).

### 2. PC side
```
pip install websockets numpy soundfile faster-whisper
python hermes_voice_server.py --port 8765
```

### 3. Network
Install **Tailscale** on PC + phone (both free), sign into the same account.
The phone reaches the PC at `ws://<tailscale-name>:8765` from anywhere —
same WiFi or mobile data, automatically.

### 4. Pair
1. Start the server on the PC.
2. In the app: Settings → Hermes URLs = `ws://<pc-tailscale-hostname>:8765`
3. Enter any pairing code for now (Milestone 1 trusts LAN/Tailscale).

## Say

> "Jarvis" → pause → your request.

## Milestones

- [x] M0 Architecture audit
- [ ] M1 Basic voice bridge (wake → STT → agent → TTS)  ← current
- [ ] M2 Voice → Android actions (open app, screenshot…)
- [ ] M3 Low-latency full-duplex streaming
- [ ] M4 Connection reliability + pairing store
- [ ] M5 Conversation mode & interruption
