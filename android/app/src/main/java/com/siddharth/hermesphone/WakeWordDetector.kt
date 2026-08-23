package com.siddharth.hermesphone

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import java.nio.FloatBuffer

/**
 * Local wake-word detection using openWakeWord (ONNX).
 * Runs entirely on-device: no audio leaves the phone until wake fires.
 *
 * openWakeWord melspectrogram models expect 16kHz mono float PCM.
 * The "hey_jarvis" model consumes 16ms frames (256 samples) accumulated
 * into an 80-frame (~1.28s) sliding window per inference.
 *
 * Modular by design (spec §67): swap WakeWordDetector implementations
 * without touching audio capture or transport.
 */
class WakeWordDetector(private val context: Context) {

    interface Listener {
        fun onWakeDetected()
    }

    private var session: OrtSession? = null
    private var embeddingSession: OrtSession? = null
    private val melBuf = FloatArray(FRAMES_PER_INFERENCE * 16) // raw input buffer
    private var melPos = 0
    var enabled = true
        private set

    companion object {
        const val SAMPLE_RATE = 16000
        const val FRAME_SAMPLES = 128          // 8ms chunks fed by AudioEngine
        const val FRAMES_PER_INFERENCE = 16    // run detector every 128ms
        const val MODEL_FILE = "hey_jarvis_v0.1.onnx"
        const val EMBEDDING_MODEL = "openwakeword_features_model.onnx"
        const val THRESHOLD = 0.5f
    }

    fun start() {
        if (session != null) return
        val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
        embeddingSession = env.createSession(loadBytes(EMBEDDING_MODEL), OrtSession.SessionOptions())
        session = env.createSession(loadBytes(MODEL_FILE), OrtSession.SessionOptions())
    }

    private fun loadBytes(name: String): ByteArray =
        context.assets.open(name).use { it.readBytes() }

    /**
     * Feed one 128-sample frame of 16kHz mono PCM.
     * Returns true exactly once when the wake word fires.
     */
    fun feed(frame: ShortArray): Boolean {
        if (!enabled || session == null) return false
        for (s in frame) {
            melBuf[melPos++] = s / 32768f
            if (melPos >= melBuf.size) {
                melPos = 0
                return runInference()
            }
        }
        return false
    }

    private fun runInference(): Boolean {
        val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
        try {
            val shape = longArrayOf(1, melBuf.size.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(melBuf), shape).use { features ->
                embeddingSession!!.run(mapOf("input" to features)).use { embOut ->
                    @Suppress("UNCHECKED_CAST")
                    val emb = (embOut[0].value as Array<FloatArray>)[0]
                    val embShape = longArrayOf(1, emb.size.toLong())
                    OnnxTensor.createTensor(env, FloatBuffer.wrap(emb), embShape).use { embTensor ->
                        session!!.run(mapOf("input" to embTensor)).use { out ->
                            val score = (out[0].value as Array<FloatArray>)[0][0]
                            if (score > THRESHOLD) {
                                reset()
                                return true
                            }
                        }
                    }
                }
            }
        } catch (_: Exception) {
            // Inference failure must never crash the audio loop; detector degrades to silent.
        }
        return false
    }

    /** Clear the sliding window so post-wake speech isn't re-detected. */
    fun reset() {
        melBuf.fill(0f)
        melPos = 0
    }

    fun stop() {
        session?.close(); session = null
        embeddingSession?.close(); embeddingSession = null
    }
}
