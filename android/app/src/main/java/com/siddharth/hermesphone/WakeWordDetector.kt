package com.siddharth.hermesphone

import ai.onnxruntime.OrtSession
import ai.onnxruntime.OnnxTensor
import android.content.Context
import java.nio.FloatBuffer
import kotlin.math.cos
import kotlin.math.PI
import kotlin.math.ln
import kotlin.math.pow
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Local wake-word detection using openWakeWord (ONNX).
 * Runs entirely on-device: no audio leaves the phone until wake fires.
 *
 * openWakeWord pipeline (matches the reference Python implementation):
 *   16kHz mono PCM -> melspectrogram (64 mel bins, hop 8ms) -> embedding model
 *   (input [1,76,32,1] mel patches -> [96]-dim embeddings, one per ~1.28s window)
 *   -> "hey_jarvis" model ([1,16,96] = 16 embedding frames -> score)
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

    // Raw PCM ring buffer; inference runs once per EMBEDDING_WINDOW_SAMPLES.
    private val pcmBuf = FloatArray(EMBEDDING_WINDOW_SAMPLES)
    private var pcmPos = 0
    private var pcmFilled = 0

    // Rolling embedding history for the jarvis model: 16 x 96 floats.
    private val embHistory = ArrayDeque<FloatArray>()

    var enabled = true
        private set
    /** Last raw wake score, for debug UI. */
    var lastScore = 0f
        private set

    companion object {
        const val SAMPLE_RATE = 16000

        // melspectrogram params — must match openWakeWord defaults
        const val N_MELS = 32              // model input is [1,76,32,1]
        const val N_FFT = 510              // oww default
        const val HOP = 128                // 8ms at 16kHz
        const val MEL_FRAMES_PER_EMB = 76  // mel frames per embedding patch

        // Embedding window: 76 mel frames * 8ms = ~0.608s of audio per patch,
        // but the reference pipeline feeds 1.28s and takes the last embedding.
        const val EMBEDDING_WINDOW_SAMPLES = 20480  // 1.28s @16kHz

        const val MODEL_FILE = "hey_jarvis_v0.1.onnx"
        const val EMBEDDING_MODEL = "embedding_model.onnx"
        const val THRESHOLD = 0.5f
        const val EMB_HISTORY_LEN = 16
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
     * Feed one frame of 16kHz mono PCM. Returns true exactly once when the
     * wake word fires.
     */
    fun feed(frame: ShortArray): Boolean {
        if (!enabled || session == null || embeddingSession == null) return false
        for (s in frame) {
            pcmBuf[pcmPos++] = s / 32768f
            if (pcmPos >= pcmBuf.size) { pcmPos = 0 }
            if (pcmFilled < pcmBuf.size) pcmFilled++
            if (pcmPos == 0 && pcmFilled == pcmBuf.size) {
                // buffer wrapped with a full window -> run one inference step
                if (runInference(pcmBuf.copyOf())) return true
            }
        }
        return false
    }

    // ------------------------------------------------------------------
    // Melspectrogram (Hann window + mel filterbank, log-scaled), matching
    // openWakeWord's librosa defaults closely enough for detection.
    // ------------------------------------------------------------------
    private val hann: FloatArray by lazy {
        FloatArray(N_FFT + 1) { n ->
            (0.5 - 0.5 * cos(2.0 * PI * n / N_FFT)).toFloat()
        }
    }

    private val melFilters: Array<FloatArray> by lazy { buildMelFilterbank() }

    private fun hzToMel(f: Float): Float = 2595f * kotlin.math.log10(1f + f / 700f)
    private fun melToHz(m: Float): Float = (700.0 * (10.0.pow(m / 2595.0) - 1.0)).toFloat()

    private fun buildMelFilterbank(): Array<FloatArray> {
        val nFftBins = N_FFT / 2 + 1
        val fMin = 0f; val fMax = SAMPLE_RATE / 2f
        val melMin = hzToMel(fMin); val melMax = hzToMel(fMax)
        // N_MELS+2 evenly spaced mel points -> triangular filters
        val melPts = FloatArray(N_MELS + 2) { i ->
            melToHz(melMin + (melMax - melMin) * i / (N_MELS + 1))
        }
        val binOf = FloatArray(N_MELS + 2) { i -> melPts[i] * N_FFT / SAMPLE_RATE }
        return Array(N_MELS) { m ->
            val w = FloatArray(nFftBins)
            val lo = binOf[m]; val mid = binOf[m + 1]; val hi = binOf[m + 2]
            for (k in 0 until nFftBins) {
                val kf = k.toFloat()
                var v = 0f
                if (kf in lo..mid && mid > lo) v = (kf - lo) / (mid - lo)
                else if (kf in mid..hi && hi > mid) v = (hi - kf) / (hi - mid)
                w[k] = v
            }
            w
        }
    }

    /** Coherent power spectrum via Goertzel-free direct DFT would be too slow;
     *  use a small radix-2 FFT (N_FFT=512 via zero-padding of 510-sample window). */
    private fun magnitudeSpectrum(window: FloatArray, outMag: FloatArray) {
        val fftN = 512
        val re = FloatArray(fftN); val im = FloatArray(fftN)
        for (i in window.indices) re[i] = window[i]
        for (i in 0 until fftN) {
            re[i] *= hann[i.coerceAtMost(hann.size - 1)]
        }
        fft(re, im)
        for (k in outMag.indices) {
            outMag[k] = re[k] * re[k] + im[k] * im[k]
        }
    }

    private fun fft(re: FloatArray, im: FloatArray) {
        val n = re.size
        // bit reversal
        var j = 0
        for (i in 1 until n) {
            var bit = n shr 1
            while (j and bit != 0) { j = j xor bit; bit = bit shr 1 }
            j = j or bit
            if (i < j) { re[i] = re[j].also { re[j] = re[i] }; im[i] = im[j].also { im[j] = im[i] } }
        }
        var len = 2
        while (len <= n) {
            val ang = -2.0 * PI / len
            val wr = cos(ang).toFloat(); val wi = sin(ang).toFloat()
            var i = 0
            while (i < n) {
                var cwr = 1f; var cwi = 0f
                for (k in 0 until len / 2) {
                    val ur = re[i + k]; val ui = im[i + k]
                    val vr = re[i + k + len / 2] * cwr - im[i + k + len / 2] * cwi
                    val vi = re[i + k + len / 2] * cwi + im[i + k + len / 2] * cwr
                    re[i + k] = ur + vr; im[i + k] = ui + vi
                    re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi
                    val nwr = cwr * wr - cwi * wi
                    cwi = cwr * wi + cwi * wr; cwr = nwr
                }
                i += len
            }
            len = len shl 1
        }
    }

    /**
     * Compute a [MEL_FRAMES_PER_EMB x N_MELS] log-mel spectrogram from a
     * window of PCM samples, then run the two-stage ONNX pipeline.
     */
    private fun runInference(samples: FloatArray): Boolean {
        try {
            val nMelFrames = (samples.size - N_FFT) / HOP + 1  // ~154 frames
            val mag = FloatArray(N_FFT / 2 + 1)
            val melFrames = ArrayList<FloatArray>(nMelFrames)
            val window = FloatArray(N_FFT)
            for (t in 0 until nMelFrames) {
                val off = t * HOP
                for (i in 0 until N_FFT) window[i] = samples[off + i]
                magnitudeSpectrum(window, mag)
                val frame = FloatArray(N_MELS)
                for (m in 0 until N_MELS) {
                    var e = 0f
                    val filt = melFilters[m]
                    for (k in mag.indices) e += filt[k] * mag[k]
                    frame[m] = ln(e + 1e-10f)
                }
                melFrames.add(frame)
            }

            // Sliding patches of MEL_FRAMES_PER_EMB frames, stride 8 (oww default):
            // each patch -> one 96-dim embedding.
            val stride = 8
            embHistory.clear()
            var p = 0
            while (p + MEL_FRAMES_PER_EMB <= melFrames.size) {
                val patch = FloatArray(MEL_FRAMES_PER_EMB * N_MELS)
                var idx = 0
                for (r in 0 until MEL_FRAMES_PER_EMB) {
                    val fr = melFrames[p + r]
                    for (c in 0 until N_MELS) patch[idx++] = fr[c]
                }
                // shape [1,76,32,1]
                val shape4 = longArrayOf(1, MEL_FRAMES_PER_EMB.toLong(), N_MELS.toLong(), 1L)
                val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
                OnnxTensor.createTensor(env, FloatBuffer.wrap(patch), shape4).use { t ->
                    embeddingSession!!.run(mapOf("input_1" to t)).use { out ->
                        val emb = (out[0].value as Array<FloatArray>)[0] // 96 floats
                        embHistory.addLast(emb.copyOf())
                        if (embHistory.size > EMB_HISTORY_LEN) embHistory.removeFirst()
                    }
                }
                p += stride
            }

            // jarvis model needs exactly 16 embedding frames [1,16,96]
            if (embHistory.size >= EMB_HISTORY_LEN) {
                val flat = FloatArray(EMB_HISTORY_LEN * 96)
                var idx = 0
                for (e in embHistory) for (v in e) flat[idx++] = v
                val env = ai.onnxruntime.OrtEnvironment.getEnvironment()
                val shape3 = longArrayOf(1, EMB_HISTORY_LEN.toLong(), 96L)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(flat), shape3).use { t ->
                    session!!.run(mapOf("x.1" to t)).use { out ->
                        val score = (out[0].value as Array<FloatArray>)[0][0]
                        lastScore = score
                        if (score > THRESHOLD) {
                            reset()
                            return true
                        }
                    }
                }
            }
        } catch (_: Exception) {
            // Inference failure must never crash the audio loop; detector degrades to silent.
        }
        return false
    }

    /** Clear buffers so post-wake speech isn't re-detected. */
    fun reset() {
        pcmBuf.fill(0f)
        pcmPos = 0; pcmFilled = 0
        embHistory.clear()
    }

    fun stop() {
        session?.close(); session = null
        embeddingSession?.close(); embeddingSession = null
    }
}
