package com.siddharth.hermesphone

import android.content.Context
import android.content.SharedPreferences

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
    }

    fun stop() {
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
        // Default: Tailscale MagicDNS name of the PC + common LAN fallback.
        return listOf(
            "ws://hermes-pc:8765/voice",
            "ws://100.100.100.100:8765/voice" // placeholder until user sets real tailnet IP
        )
    }

    /** All mic frames flow through here — the single dispatch point. */
    private fun onAudioFrame(frame: ShortArray, streaming: Boolean) {
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

    private fun onWake() {
        if (conn.state != "CONNECTED") { setState(STANDBY); return }
        wake.reset()
        tts?.stopNow()               // interrupt any playback ("Jarvis" barge-in, spec §34)
        silenceFrames = 0; speechStarted = false
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

    override fun onResponseText(text: String) {}

    override fun onTtsChunk(pcm: ByteArray) {
        if (state == THINKING || state == TRANSCRIBING) setState(SPEAKING)
        tts?.enqueue(pcm)
    }

    override fun onTtsEnd() {
        ttsEndAt = System.currentTimeMillis()
        // Conversation mode: brief pause, then back to standby (wake required again).
        Thread { Thread.sleep(CONVERSATION_WINDOW_MS); if (state == SPEAKING) setState(STANDBY) }.start()
        setState(STANDBY)
        tts?.stopNow()
    }

    /** User tapped "stop" or said a barge-in keyword during SPEAKING. */
    fun interrupt() {
        tts?.stopNow()
        conn.sendInterrupt()
        setState(STANDBY)
    }

    private fun pcm16ToBytes(frame: ShortArray): ByteArray {
        val b = ByteArray(frame.size * 2)
        for (i in frame.indices) {
            b[2 * i] = (frame[i].toInt() and 0xFF).toByte()
            b[2 * i + 1] = (frame[i].toInt() shr 8).toByte()
        }
        return b
    }

    private fun setState(s: String) {
        if (s == state) return
        if ((state == LISTENING) && s != LISTENING && s != TRANSCRIBING) conn.sendAudioEnd()
        state = s
        ui.onStateChanged(s)
    }
}
