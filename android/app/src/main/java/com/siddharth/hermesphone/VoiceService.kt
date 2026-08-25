package com.siddharth.hermesphone

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.IBinder
import androidx.core.app.NotificationCompat

/**
 * Foreground service so Android permits continuous mic access while remaining
 * honest with the user (persistent notification + mic indicator, spec §13).
 */
class VoiceService : Service(), VoiceEngine.Ui {

    companion object {
        const val CHANNEL_ID = "hermes_voice"
        const val NOTIF_ID = 1
        @Volatile var engine: VoiceEngine? = null
        /** Activity that should receive live UI updates (set in onResume, cleared in onPause). */
        @Volatile var activeUi: VoiceEngine.Ui? = null
    }

    /** Forward UI callbacks to whichever screen is visible (or notification if none). */
    private fun uiTarget(): VoiceEngine.Ui? = activeUi

    override fun onCreate() {
        super.onCreate()
        if (checkSelfPermission(android.Manifest.permission.RECORD_AUDIO)
            != PackageManager.PERMISSION_GRANTED) { stopSelf(); return }
        createChannel()
        startForeground(NOTIF_ID, buildNotification("Starting…"))
        val e = VoiceEngine(applicationContext, this)
        engine = e
        e.start()
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        when (intent?.action) {
            "STOP" -> { engine?.stop(); stopSelf() }
            "INTERRUPT" -> engine?.interrupt()
        }
        return START_STICKY // survive process kill; reconnect logic handles the rest
    }

    override fun onDestroy() {
        engine?.stop(); engine = null
        super.onDestroy()
    }

    override fun onBind(intent: Intent?): IBinder? = null

    private fun createChannel() {
        val ch = NotificationChannel(CHANNEL_ID, "Hermes Voice", NotificationManager.IMPORTANCE_LOW)
        getSystemService(NotificationManager::class.java).createNotificationChannel(ch)
    }

    private fun buildNotification(text: String): Notification {
        val open = PendingIntent.getActivity(
            this, 0, Intent(this, MainActivity::class.java),
            PendingIntent.FLAG_IMMUTABLE)
        return NotificationCompat.Builder(this, CHANNEL_ID)
            .setContentTitle("Hermes")
            .setContentText(text)
            .setSmallIcon(android.R.drawable.ic_btn_speak_now)
            .setContentIntent(open)
            .setOngoing(true)
            .build()
    }

    // ---- VoiceEngine.Ui ----
    override fun onStateChanged(state: String) {
        val label = when (state) {
            VoiceEngine.STANDBY -> "Listening for “Jarvis”"
            VoiceEngine.LISTENING -> "Listening…"
            VoiceEngine.THINKING -> "Thinking…"
            VoiceEngine.SPEAKING -> "Speaking…"
            "RECONNECTING" -> "Reconnecting to Hermes…"
            else -> state.lowercase().replaceFirstChar { it.uppercase() }
        }
        startForeground(NOTIF_ID, buildNotification(label))
        activeUi?.onStateChanged(state)
    }

    override fun onTranscript(text: String, partial: Boolean) {
        activeUi?.onTranscript(text, partial)
    }

    override fun onDebug(line: String) {
        activeUi?.onDebug(line)
    }
}
