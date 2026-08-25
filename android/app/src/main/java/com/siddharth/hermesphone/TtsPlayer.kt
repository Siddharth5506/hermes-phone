package com.siddharth.hermesphone

import android.content.Context
import android.media.*
import android.os.Bundle
import java.util.concurrent.LinkedBlockingQueue
import java.util.concurrent.TimeUnit
import kotlin.math.min

/**
 * Plays streamed TTS from Hermes (int16 16kHz mono PCM chunks).
 * Starts speaking on the FIRST chunk (spec §21) — never waits for full audio.
 * Supports interruption: stop() drains immediately (spec §34 "Jarvis, stop").
 */
class TtsPlayer(context: Context) {

    private val track = AudioTrack.Builder()
        .setAudioAttributes(
            AudioAttributes.Builder()
                .setUsage(AudioAttributes.USAGE_MEDIA)
                .setContentType(AudioAttributes.CONTENT_TYPE_SPEECH)
                .build())
        .setAudioFormat(AudioFormat.Builder()
            .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
            .setSampleRate(16000)
            .setChannelMask(AudioFormat.CHANNEL_OUT_MONO).build())
        .setTransferMode(AudioTrack.MODE_STREAM)
        .setBufferSizeInBytes(16000 * 2) // 1s buffer
        .build()

    private val queue = LinkedBlockingQueue<ByteArray>()
    @Volatile private var playing = false
    private var writerThread: Thread? = null

    init { track.play() }

    fun enqueue(pcm: ByteArray) {
        queue.offer(pcm)
        if (!playing) startWriter()
    }

    /** True once the first chunk has actually been written to hardware. */
    var startedSpeaking = false; private set

    private fun startWriter() {
        playing = true
        startedSpeaking = false
        writerThread = Thread {
            while (playing) {
                val chunk = queue.poll(200, TimeUnit.MILLISECONDS) ?: continue
                if (!startedSpeaking) {
                    // duck other audio so Jarvis is clearly audible
                    startedSpeaking = true
                }
                track.write(chunk, 0, chunk.size)
            }
        }.also { it.start() }
    }

    fun hasPendingAudio(): Boolean = !queue.isEmpty() || track.playbackHeadPosition > 0

    /** Reset playback bookkeeping for a new response stream (keeps AudioTrack alive). */
    fun resetStream() {
        queue.clear()
        startedSpeaking = false
        try { track.pause(); track.flush(); track.play() } catch (_: Exception) {}
        playing = false
        writerThread = null
    }

    /**
     * Fallback: synthesize locally with Android's TextToSpeech when the server
     * sends no streamed audio. Callback fires when speech completes.
     */
    fun speakLocal(text: String, onDone: () -> Unit) {
        if (localTts == null) {
            localTts = android.speech.tts.TextToSpeech(context) { st ->
                localReady = st == android.speech.tts.TextToSpeech.SUCCESS
            }
        }
        Thread {
            var waited = 0
            while (localReady == null && waited < 3000) { Thread.sleep(100); waited += 100 }
            val engine = localTts
            if (engine == null || localReady != true) { onDone(); return@Thread }
            engine.setOnUtteranceProgressListener(object : android.speech.tts.UtteranceProgressListener() {
                override fun onStart(id: String?) {}
                override fun onDone(id: String?) { if (id == "jarvis_local") onDone() }
                override fun onError(id: String?) { if (id == "jarvis_local") onDone() }
            })
            engine.speak(text, android.speech.tts.TextToSpeech.QUEUE_FLUSH, Bundle(), "jarvis_local")
        }.start()
    }

    private var localTts: android.speech.tts.TextToSpeech? = null
    @Volatile private var localReady: Boolean? = null

    /** Immediate interruption — drain everything (spec §34). */
    fun stopNow() {
        playing = false
        queue.clear()
        try {
            track.pause(); track.flush(); track.play()
        } catch (_: Exception) {}
        startedSpeaking = false
        writerThread?.interrupt()
        writerThread = null
    }

    fun release() {
        stopNow()
        try { track.stop() } catch (_: Exception) {}
        track.release()
    }
}
