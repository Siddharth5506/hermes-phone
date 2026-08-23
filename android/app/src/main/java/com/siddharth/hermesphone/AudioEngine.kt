package com.siddharth.hermesphone

import android.annotation.SuppressLint
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import kotlin.math.abs
import kotlin.math.min

/**
 * Single owner of the microphone (spec §33 — no concurrent sessions).
 *
 * States:
 *   STANDBY   -> mic at 16kHz, frames go ONLY to local wake word detector.
 *                Nothing is transmitted. No network audio. (spec §52)
 *   LISTENING -> wake fired; frames stream to Hermes over WebSocket
 *                until VoiceEngine signals end-of-speech / session end.
 *
 * The mic indicator will show while the foreground service runs — that is
 * Android law (spec §13) and we work within it, not around it.
 */
class AudioEngine(
    private val onFrame: (ShortArray, Boolean) -> Unit // (samples, isStreaming)
) {
    companion object {
        const val SAMPLE_RATE = 16000
        private const val FRAME_SAMPLES = 128 // 8ms
        private const val CHANNEL = AudioFormat.CHANNEL_IN_MONO
        private const val ENCODING = AudioFormat.ENCODING_PCM_16BIT
        // Simple energy VAD: below this RMS we consider silence (tune experimentally, spec §19)
        const val SILENCE_RMS = 350f
    }

    @Volatile var streaming = false
        private set

    private var record: AudioRecord? = null
    private var thread: Thread? = null

    @SuppressLint("MissingPermission") // permission checked by VoiceService before start()
    fun start() {
        if (record != null) return
        val minBuf = AudioRecord.getMinBufferSize(SAMPLE_RATE, CHANNEL, ENCODING)
        record = AudioRecord(
            MediaRecorder.AudioSource.VOICE_RECOGNITION,
            SAMPLE_RATE, CHANNEL, ENCODING,
            maxOf(minBuf, FRAME_SAMPLES * 4)
        )
        record?.startRecording()
        thread = Thread({
            val buf = ShortArray(FRAME_SAMPLES)
            while (record != null && record!!.state == AudioRecord.STATE_INITIALIZED) {
                val n = record!!.read(buf, 0, FRAME_SAMPLES)
                if (n > 0) onFrame(buf.copyOf(n), streaming)
            }
        }, "AudioEngine").apply { priority = Thread.MAX_PRIORITY - 1; start() }
    }

    /** Called when wake word fires: subsequent frames are streamed to Hermes. */
    fun beginStreaming() { streaming = true }

    /** End of voice session: back to STANDBY (wake-only). */
    fun endStreaming() {
        streaming = false
    }

    fun stop() {
        streaming = false
        try { record?.stop() } catch (_: Exception) {}
        record?.release(); record = null
        thread = null
    }

    /** Rough energy level of a frame, for VAD + debug UI. */
    fun rms(frame: ShortArray): Float {
        var sum = 0L
        for (s in frame) sum += s.toLong() * s
        return kotlin.math.sqrt(sum.toDouble() / frame.size).toFloat()
    }
}
