package com.siddharth.hermesphone

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import java.nio.FloatBuffer

/**
 * Local wake-word detection using openWakeWord (ONNX).
 * Uses openWakeWord's OFFICIAL ONNX models for every stage — no hand-rolled DSP:
 *   1280-sample (80ms) chunks -> melspectrogram.onnx -> [T,1,32] mel frames
 *   -> 76-frame patches -> embedding_model.onnx -> 96-dim embeddings
 *   -> sliding window of 16 embeddings [1,16,96] -> hey_jarvis_v0.1.onnx -> score
 */
class WakeWordDetector(private val context: Context) {

    interface Listener {
        fun onWakeDetected()
    }

    private var jarvisSession: OrtSession? = null
    private var embeddingSession: OrtSession? = null
    private var melspecSession: OrtSession? = null

    private val pcmBuf = ShortArray(MEL_CHUNK)
    private var pcmPos = 0
    private val melHistory = ArrayDeque<FloatArray>()
    private val embHistory = ArrayDeque<FloatArray>()

    var enabled = true
        private set
    var lastScore = 0f
        private set
    var armed = false
        private set
    var lastError: String? = null
        private set
    var inferenceCount = 0L
        private set

    companion object {
        const val SAMPLE_RATE = 16000
        const val MEL_CHUNK = 1280
        const val MEL_PATCH_FRAMES = 76
        const val EMB_HISTORY_LEN = 16
        const val N_MELS = 32
        const val MODEL_FILE = "hey_jarvis_v0.1.onnx"
        const val EMBEDDING_MODEL = "embedding_model.onnx"
        const val MELSPECTROGRAM_MODEL = "melspectrogram.onnx"
        const val THRESHOLD = 0.5f
    }

    fun start() {
        if (jarvisSession != null) return
        val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
        melspecSession = env.createSession(loadBytes(MELSPECTROGRAM_MODEL), OrtSession.SessionOptions())
        embeddingSession = env.createSession(loadBytes(EMBEDDING_MODEL), OrtSession.SessionOptions())
        jarvisSession = env.createSession(loadBytes(MODEL_FILE), OrtSession.SessionOptions())
    }

    private fun loadBytes(name: String): ByteArray =
        context.assets.open(name).use { it.readBytes() }

    fun feed(frame: ShortArray): Boolean {
        if (!enabled || jarvisSession == null || embeddingSession == null || melspecSession == null) return false
        for (s in frame) {
            pcmBuf[pcmPos++] = s
            if (pcmPos >= pcmBuf.size) {
                pcmPos = 0
                processChunk()
                if (lastScore > THRESHOLD) {
                    reset()
                    return true
                }
            }
        }
        return false
    }

    private fun processChunk() {
        try {
            val env = ai.onnxruntime.OrtEnvironment.getEnvironment()

            // 1) PCM -> mel frames via oww's official melspectrogram ONNX
            val f = FloatArray(pcmBuf.size)
            for (i in pcmBuf.indices) f[i] = pcmBuf[i] / 32768f
            val shape2 = longArrayOf(1, f.size.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(f), shape2).use { t ->
                melspecSession!!.run(mapOf("input" to t)).use { out ->
                    val arr = out[0].value as Array<Array<Array<FloatArray>>> // [T,1,32]
                    for (i in arr.indices) {
                        melHistory.addLast(arr[i][0])
                        while (melHistory.size > MEL_PATCH_FRAMES + 64) melHistory.removeFirst()
                    }
                }
            }

            // 2) 76-frame patch -> embedding
            while (melHistory.size >= MEL_PATCH_FRAMES) {
                val patch = FloatArray(MEL_PATCH_FRAMES * N_MELS)
                var taken = 0
                for (frame in melHistory) {
                    if (taken >= MEL_PATCH_FRAMES) break
                    System.arraycopy(frame, 0, patch, taken * N_MELS, N_MELS)
                    taken++
                }
                repeat(taken) { melHistory.removeFirst() }

                val shape4 = longArrayOf(1, MEL_PATCH_FRAMES.toLong(), N_MELS.toLong(), 1L)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(patch), shape4).use { t ->
                    embeddingSession!!.run(mapOf("input_1" to t)).use { out ->
                        val emb = (out[0].value as Array<Array<Array<FloatArray>>>)[0][0][0]
                        embHistory.addLast(emb.copyOf())
                        while (embHistory.size > EMB_HISTORY_LEN) embHistory.removeFirst()
                    }
                }
            }

            // 3) 16 embeddings -> jarvis score
            armed = embHistory.size >= EMB_HISTORY_LEN
            if (armed) {
                val flat = FloatArray(EMB_HISTORY_LEN * 96)
                var idx = 0
                for (e in embHistory) for (v in e) flat[idx++] = v
                val shape3 = longArrayOf(1, EMB_HISTORY_LEN.toLong(), 96L)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape3).use { t ->
                    jarvisSession!!.run(mapOf("x.1" to t)).use { out ->
                        lastScore = (out[0].value as Array<FloatArray>)[0][0]
                        inferenceCount++
                    }
                }
            }
        } catch (e: Exception) {
            lastError = e.message?.take(80) ?: e.javaClass.simpleName
        }
    }

    fun reset() {
        pcmBuf.fill(0)
        pcmPos = 0
        melHistory.clear()
        embHistory.clear()
        lastScore = 0f
        armed = false
    }

    fun stop() {
        jarvisSession?.close(); jarvisSession = null
        embeddingSession?.close(); embeddingSession = null
        melspecSession?.close(); melspecSession = null
    }
}
