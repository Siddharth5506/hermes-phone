package com.siddharth.hermesphone

import android.content.Context
import android.media.*
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

    fun hasPending(): Boolean = !queue.isEmpty() || track.playbackHeadPosition > 0 &&
            (track.bufferSizeInBytes - track.availableFramesToWrite() * 2) > 0

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
