package com.siddharth.hermesphone

import android.content.Context
import android.media.AudioAttributes
import android.media.AudioFocusRequest
import android.media.AudioManager
import android.media.MediaPlayer
import okhttp3.*
import okio.ByteString
import okio.ByteString.Companion.toByteString
import org.json.JSONObject
import java.util.concurrent.TimeUnit

/**
 * Persistent connection to Hermes PC (spec §27-29).
 * Modular transport: voice pipeline does not care whether the endpoint is
 * LAN IP or Tailscale IP — ConnectionManager just picks the first reachable.
 *
 * Protocol (JSON frames over WSS/WS):
 *  phone -> hermes: {type:"hello", device, pairing_code}
 *                   {type:"audio_start"}
 *                   {type:"audio", pcm: <base64 int16 16k mono>}
 *                   {type:"audio_end"}
 *  hermes -> phone: {type:"welcome"}
 *                   {type:"transcript_partial", text}
 *                   {type:"transcript_final", text}
 *                   {type:"response_text", text}
 *                   {type:"tts_audio", pcm: <base64 int16 16k mono>} (streamed chunks)
 *                   {type:"tts_end"}
 *                   {type:"action_result", ok, detail}
 *                   {type:"stop"}
 */
class HermesConnection(private val context: Context, private val listener: Listener) {

    interface Listener {
        fun onState(state: String)          // CONNECTING / CONNECTED / DISCONNECTED
        fun onTranscriptPartial(text: String)
        fun onTranscriptFinal(text: String)
        fun onTtsChunk(pcm: ByteArray)      // int16 16k mono
        fun onTtsEnd()
        fun onResponseText(text: String)
    }

    private val client = OkHttpClient.Builder()
        .pingInterval(15, TimeUnit.SECONDS)
        .connectTimeout(8, TimeUnit.SECONDS)
        .readTimeout(0, TimeUnit.MILLISECONDS) // persistent
        .build()

    private var ws: WebSocket? = null
    @Volatile var state = "DISCONNECTED"; private set

    // Endpoints tried in order (spec §25): Tailscale first (works LAN+away), then LAN.
    var endpoints: List<String> = emptyList()
    private var endpointIdx = 0
    private var backoffMs = 1000L
    private val maxBackoff = 30000L
    @Volatile private var wantConnected = false
    @Volatile var pairingCode: String = ""

    fun connect() {
        if (endpoints.isEmpty()) return
        wantConnected = true
        backoffMs = 1000L
        dial()
    }

    private fun dial() {
        if (!wantConnected) return
        setState("CONNECTING")
        val url = endpoints[endpointIdx % endpoints.size]
        val req = Request.Builder().url(url).build()
        ws = client.newWebSocket(req, object : WebSocketListener() {
            override fun onOpen(webSocket: WebSocket, response: Response) {
                backoffMs = 1000L
                setState("CONNECTED")
                val hello = JSONObject().put("type", "hello")
                    .put("device", android.os.Build.MODEL)
                    .put("pairing_code", pairingCode)
                webSocket.send(hello.toString())
            }
            override fun onMessage(webSocket: WebSocket, text: String) = onFrame(JSONObject(text))
            override fun onMessage(webSocket: WebSocket, bytes: ByteString) {}
            override fun onFailure(webSocket: WebSocket, t: Throwable, response: Response?) {
                setState("DISCONNECTED")
                scheduleReconnect()
            }
            override fun onClosed(webSocket: WebSocket, code: Int, reason: String) {
                setState("DISCONNECTED")
                scheduleReconnect()
            }
        })
    }

    private fun onFrame(j: JSONObject) = when (j.optString("type")) {
        "welcome" -> {}
        "transcript_partial" -> listener.onTranscriptPartial(j.optString("text"))
        "transcript_final" -> listener.onTranscriptFinal(j.optString("text"))
        "response_text" -> listener.onResponseText(j.optString("text"))
        "tts_audio" -> listener.onTtsChunk(android.util.Base64.decode(j.getString("pcm"), android.util.Base64.DEFAULT))
        "tts_end" -> listener.onTtsEnd()
        else -> {}
    }

    private fun scheduleReconnect() {
        if (!wantConnected) return
        endpointIdx++ // try next endpoint next time
        Thread {
            Thread.sleep(backoffMs)
            backoffMs = min(backoffMs * 2, maxBackoff) // exponential backoff (spec §29)
            dial()
        }.start()
    }

    // ---- outbound ----
    fun sendAudioStart() = send(JSONObject().put("type", "audio_start"))
    fun sendAudioChunk(pcm: ByteArray) =
        send(JSONObject().put("type", "audio")
            .put("pcm", android.util.Base64.encodeToString(pcm, android.util.Base64.NO_WRAP)))
    fun sendAudioEnd() = send(JSONObject().put("type", "audio_end"))
    fun sendInterrupt() = send(JSONObject().put("type", "interrupt"))

    private fun send(j: JSONObject): Boolean {
        val w = ws ?: return false
        return if (state == "CONNECTED") w.send(j.toString()) else false
    }

    fun disconnect() {
        wantConnected = false
        ws?.close(1000, "bye"); ws = null
        setState("DISCONNECTED")
    }

    private fun setState(s: String) {
        state = s
        listener.onState(s)
    }
}
