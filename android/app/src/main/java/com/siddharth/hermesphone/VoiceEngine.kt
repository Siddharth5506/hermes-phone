package com.siddharth.hermesphone

import android.content.Context
import android.content.SharedPreferences
import android.os.Handler
import android.os.Looper

/**
 * Voice session state machine (spec §33).
 *
 * OFFLINE -> STANDBY -> WAKE_DETECTED -> LISTENING -> TRANSCRIBING
 *   -> THINKING -> SPEAKING -> STANDBY
 * ERROR / RECONNECTING overlay on any state.
 *
 * Single owner prevents race conditions: no listening-while-speaking,
 * one voice session at a time, TTS always stopped before returning to standby.
 */
class VoiceEngine(private val context: Context, private val ui: Ui) :
    HermesConnection.Listener, WakeWordDetector.Listener {

    override fun onWakeDetected() {} // wake handled via AudioEngine callback path

    interface Ui {
        fun onStateChanged(state: String)
        fun onTranscript(text: String, partial: Boolean)
        /** Live debug line — shown in the debug overlay. Called at most ~2x/sec. */
        fun onDebug(line: String)
    }

    companion object {
        const val OFFLINE = "OFFLINE"
        const val STANDBY = "STANDBY"
        const val WAKE_DETECTED = "WAKE_DETECTED"
        const val LISTENING = "LISTENING"
        const val TRANSCRIBING = "TRANSCRIBING"
        const val THINKING = "THINKING"
        const val SPEAKING = "SPEAKING"
        // Conversation mode (spec §35): stay open for follow-ups this long after speaking.
        const val CONVERSATION_WINDOW_MS = 8000L
    }

    lateinit var audio: AudioEngine
    lateinit var wake: WakeWordDetector
    lateinit var conn: HermesConnection
    var tts: TtsPlayer? = null

    @Volatile var state = OFFLINE; private set
    private var lastSpeechRms = 0f
    private var silenceFrames = 0
    private var speechStarted = false
    private var ttsEndAt = 0L
    private val prefs: SharedPreferences =
        context.getSharedPreferences("hermes", Context.MODE_PRIVATE)

    // Debug loop: pushes live mic/wake/conn stats to the UI twice a second.
    private val dbgHandler = Handler(Looper.getMainLooper())
    private var framesSeen = 0L
    private var wakeErrors = 0
    private var lastWakeErr: String? = null

    private val dbgTick = object : Runnable {
        override fun run() {
            val w = if (::wake.isInitialized) wake else null
            val c = if (::conn.isInitialized) conn else null
            // Multi-line, no horizontal scroll: everything visible on small screens.
            val line = buildString {
                append("ws=").append(c?.state ?: "?")
                append("  state=").append(state)
                append("\nframes=").append(framesSeen)
                append("  rms=").append("%.0f".format(lastSpeechRms))
                append("\nscore=").append("%.3f".format(w?.lastScore ?: 0f))
                append("  armed=").append(w?.armed ?: false)
                append("  inf=").append(w?.inferenceCount ?: 0L)
                lastWakeErr?.let { append("\nERR=").append(it) }
            }
            ui.onDebug(line)
            dbgHandler.postDelayed(this, 500)
        }
    }

    fun start() {
        audio = AudioEngine(::onAudioFrame)
        wake = WakeWordDetector(context).also { it.start() }
        conn = HermesConnection(context, this)
        conn.endpoints = loadEndpoints()
        conn.pairingCode = prefs.getString("pairing_code", "") ?: ""
        tts = TtsPlayer(context)
        conn.connect()
        setState(STANDBY)
        audio.start()
        dbgHandler.post(dbgTick)
    }

    fun stop() {
        dbgHandler.removeCallbacks(dbgTick)
        setState(OFFLINE)
        conn.disconnect()
        audio.stop()
        wake.stop()
        tts?.release()
    }

    fun savePairing(code: String) {
        prefs.edit().putString("pairing_code", code.trim()).apply()
        conn.pairingCode = code.trim()
        if (state != OFFLINE) { conn.disconnect(); conn.connect() }
    }

    fun saveEndpoints(list: List<String>) {
        prefs.edit().putString("endpoints", list.joinToString(",")).apply()
        conn.endpoints = list
        if (state != OFFLINE) { conn.disconnect(); conn.connect() }
    }

    private fun loadEndpoints(): List<String> {
        val saved = prefs.getString("endpoints", null)
        if (!saved.isNullOrBlank()) return saved.split(",").map { it.trim() }.filter { it.isNotBlank() }
        // Default: Tailscale IP of the PC + MagicDNS fallback.
        return listOf(
            "ws://100.97.83.70:8765/voice",
            "ws://hermes-pc:8765/voice"
        )
    }

    /** All mic frames flow through here — the single dispatch point. */
    private fun onAudioFrame(frame: ShortArray, streaming: Boolean) {
        framesSeen++
        lastSpeechRms = audio.rms(frame)
        when (state) {
            STANDBY -> {
                if (wake.feed(frame)) {
                    setState(WAKE_DETECTED)
                    onWake()
                }
            }
            LISTENING -> {
                conn.sendAudioChunk(pcm16ToBytes(frame))
                handleVad(frame)
            }
        }
    }

    private fun pcm16ToBytes(frame: ShortArray): ByteArray {
        val out = ByteArray(frame.size * 2)
        for (i in frame.indices) {
            out[i * 2] = (frame[i].toInt() and 0xFF).toByte()
            out[i * 2 + 1] = (frame[i].toInt() shr 8).toByte()
        }
        return out
    }

    private fun onWake() {
        if (conn.state != "CONNECTED") { setState(STANDBY); return }
        wake.reset()
        tts?.stopNow()               // interrupt any playback ("Jarvis" barge-in, spec §34)
        silenceFrames = 0; speechStarted = false
        sessionStartNanos = System.nanoTime()
        conn.sendAudioStart()
        setState(LISTENING)
    }

    /** Energy VAD with pre-roll buffering handled by keeping detector window warm. */
    private fun handleVad(frame: ShortArray) {
        val r = audio.rms(frame)
        if (r > AudioEngine.SILENCE_RMS) { speechStarted = true; silenceFrames = 0 }
        else if (speechStarted) {
            silenceFrames++
            // ~700ms of trailing silence after speech = end of utterance (tune experimentally)
            if (silenceFrames > 87) endUtterance()
        }
        // hard cap 12s so a stuck-open mic can't stream forever
        if (System.nanoTime() - sessionStartNanos > 12_000_000_000) endUtterance()
    }

    private var sessionStartNanos = 0L
    private fun endUtterance() {
        if (state != LISTENING) return
        conn.sendAudioEnd()
        setState(TRANSCRIBING)
    }

    // ---- HermesConnection.Listener ----
    override fun onState(s: String) {
        if (s == "DISCONNECTED" && state !in listOf(OFFLINE)) setState(RECONNECTING_STATE(s))
    }
    private fun RECONNECTING_STATE(unused: String) = "RECONNECTING"

    override fun onTranscriptPartial(text: String) {
        ui.onTranscript(text, partial = true)
        if (state == TRANSCRIBING || state == LISTENING) setState(THINKING)
    }

    override fun onTranscriptFinal(text: String) {
        ui.onTranscript(text, partial = false)
        setState(THINKING)
    }

    override fun onResponseText(text: String) {
        // Server streams TTS chunks via onTtsChunk; local synthesis is only a fallback
        // when the server sends no audio within 1.5s.
        tts?.let { t ->
            t.resetStream()
            Thread {
                Thread.sleep(1500)
                if (!t.startedSpeaking && state == THINKING) {
                    // No server TTS arrived — synthesize locally so the user still hears a reply.
                    t.speakLocal(text) { ttsEndAt = System.currentTimeMillis(); setState(STANDBY) }
                }
            }.start()
        }
        setState(SPEAKING)
    }

    override fun onTtsChunk(pcm: ByteArray) { tts?.enqueue(pcm) }
    override fun onTtsEnd() { /* stream end; playback finishes naturally */ }

    fun interrupt() {
        tts?.stopNow()
        conn.sendInterrupt()
        if (state == SPEAKING || state == THINKING) setState(STANDBY)
    }

    private fun setState(s: String) {
        state = s
        ui.onStateChanged(s)
    }
}
