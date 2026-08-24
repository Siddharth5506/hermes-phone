package com.siddharth.hermesphone

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import java.nio.FloatBuffer

/**
 * Local wake-word detection using openWakeWord (ONNX).
 * Runs entirely on-device: no audio leaves the phone until wake fires.
 *
 * Uses openWakeWord's OWN melspectrogram.onnx model (bundled in assets) so the
 * feature extraction exactly matches what the embedding/wakeword models were
 * trained on — a hand-rolled mel implementation does NOT match and scores
 * never cross threshold.
 *
 * Pipeline (mirrors openwakeword.utils.AudioFeatures):
 *   1280-sample (80ms) chunks -> melspectrogram.onnx -> [1,1,T,32] mel frames
 *   -> 76-frame patches -> embedding_model.onnx -> 96-dim embeddings
 *   -> sliding window of 16 embeddings [1,16,96] -> hey_jarvis_v0.1.onnx -> score
 *
 * Modular by design (spec §67): swap WakeWordDetector implementations
 * without touching audio capture or transport.
 */
class WakeWordDetector(private val context: Context) {

    interface Listener {
        fun onWakeDetected()
    }

    private var jarvisSession: OrtSession? = null
    private var embeddingSession: OrtSession? = null
    private var melspecSession: OrtSession? = null

    // Rolling raw-PCM buffer; we run features every MEL_CHUNK samples (oww feeds
    // 1280-sample chunks; feeding more at once is fine — output is cumulative).
    private val pcmBuf = ShortArray(MEL_CHUNK)
    private var pcmPos = 0

    // Rolling mel history: each row is one 32-dim mel frame.
    private val melHistory = ArrayDeque<FloatArray>()

    // Rolling embedding history for the jarvis model.
    private val embHistory = ArrayDeque<FloatArray>()

    var enabled = true
        private set
    /** Last raw wake score, for debug UI. */
    var lastScore = 0f
        private set

    companion object {
        const val SAMPLE_RATE = 16000
        const val MEL_CHUNK = 1280          // 80ms — oww's canonical feature chunk
        const val MEL_PATCH_FRAMES = 76     // mel frames per embedding patch
        const val EMB_HISTORY_LEN = 16      // jarvis model window
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

    /**
     * Feed one frame of 16kHz mono PCM. Returns true exactly once when the
     * wake word fires.
     */
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

    /** Run melspec + embeddings on the latest MEL_CHUNK of audio, update histories, score. */
    private fun processChunk() {
        try {
            val env = ai.onnxruntime.OrtEnvironment.getEnvironment()

            // 1) PCM -> mel frames via oww's own ONNX model
            val f = FloatArray(pcmBuf.size)
            for (i in pcmBuf.indices) f[i] = pcmBuf[i] / 32768f
            val shape2 = longArrayOf(1, f.size.toLong())
            OnnxTensor.createTensor(env, FloatBuffer.wrap(f), shape2).use { t ->
                melspecSession!!.run(mapOf("input" to t)).use { out ->
                    val arr = out[0].value as Array<Array<Array<FloatArray>>> // [1,1,T,32]
                    val time = arr[0][0].size
                    for (i in 0 until time) {
                        melHistory.addLast(arr[0][0][i])
                        while (melHistory.size > MEL_PATCH_FRAMES + 64) melHistory.removeFirst()
                    }
                }
            }

            // 2) When we have >= 76 mel frames, make ONE new patch per chunk and embed it
            while (melHistory.size >= MEL_PATCH_FRAMES) {
                val patch = FloatArray(MEL_PATCH_FRAMES * N_MELS)
                var idx = 0
                val it = melHistory.iterator()
                val first = melHistory.first()
                // take oldest MEL_PATCH_FRAMES frames
                var taken = 0
                for (frame in melHistory) {
                    if (taken >= MEL_PATCH_FRAMES) break
                    System.arraycopy(frame, 0, patch, taken * N_MELS, N_MELS)
                    taken++
                }
                // consume those frames from the front
                repeat(taken) { melHistory.removeFirst() }

                val shape4 = longArrayOf(1, MEL_PATCH_FRAMES.toLong(), N_MELS.toLong(), 1L)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(patch), shape4).use { t ->
                    embeddingSession!!.run(mapOf("input_1" to t)).use { out ->
                        val emb = (out[0].value as Array<Array<Array<FloatArray>>>)[0][0][0] // 96 floats
                        embHistory.addLast(emb.copyOf())
                        while (embHistory.size > EMB_HISTORY_LEN) embHistory.removeFirst()
                    }
                }
            }

            // 3) Score when we have 16 embeddings
            if (embHistory.size == EMB_HISTORY_LEN) {
                val flat = FloatArray(EMB_HISTORY_LEN * 96)
                var idx = 0
                for (e in embHistory) for (v in e) flat[idx++] = v
                val shape3 = longArrayOf(1, EMB_HISTORY_LEN.toLong(), 96L)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape3).use { t ->
                    jarvisSession!!.run(mapOf("x.1" to t)).use { out ->
                        lastScore = (out[0].value as Array<FloatArray>)[0][0]
                    }
                }
            }
        } catch (_: Exception) {
            // Inference failure must never crash the audio loop; detector degrades to silent.
        }
    }

    /** Clear buffers so post-wake speech isn't re-detected. */
    fun reset() {
        pcmBuf.fill(0)
        pcmPos = 0
        melHistory.clear()
        embHistory.clear()
        lastScore = 0f
    }

    fun stop() {
        jarvisSession?.close(); jarvisSession = null
        embeddingSession?.close(); embeddingSession = null
        melspecSession?.close(); melspecSession = null
    }
}
