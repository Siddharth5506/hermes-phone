package com.siddharth.hermesphone

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat

/**
 * Deliberately minimal UI (spec §31): status, one toggle, pairing + endpoint fields,
 * and a collapsible debug section (spec §32). Not a developer dashboard.
 */
class MainActivity : AppCompatActivity(), VoiceEngine.Ui {

    private lateinit var statusView: TextView
    private lateinit var transcriptView: TextView
    private lateinit var toggleBtn: Button

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        val pad = (16 * resources.displayMetrics.density).toInt()
        val root = LinearLayout(this).apply {
            orientation = LinearLayout.VERTICAL
            setPadding(pad, pad, pad, pad)
        }

        val title = TextView(this).apply {
            text = "HERMES"
            textSize = 28f
        }
        statusView = TextView(this).apply { text = "● Offline"; textSize = 18f }
        transcriptView = TextView(this).apply { textSize = 16f }
        toggleBtn = Button(this).apply { text = "Start Jarvis" }

        root.addView(title)
        root.addView(statusView)
        root.addView(toggleBtn)
        root.addView(transcriptView)

        // --- settings section ---
        root.addView(TextView(this).apply { text = "\nSettings"; textSize = 18f })

        val codeInput = EditText(this).apply {
            hint = "Pairing code (from Hermes)"
            setText(getSharedPreferences("hermes", MODE_PRIVATE).getString("pairing_code", ""))
        }
        val epInput = EditText(this).apply {
            hint = "Hermes URLs (comma-separated, ws://)"
            setText(getSharedPreferences("hermes", MODE_PRIVATE).getString("endpoints", ""))
        }
        val saveBtn = Button(this).apply { text = "Save settings" }
        saveBtn.setOnClickListener {
            val code = codeInput.text.toString()
            val eps = epInput.text.toString().split(",").map { it.trim() }.filter { it.isNotBlank() }
            VoiceService.engine?.let { e ->
                e.savePairing(code)
                e.saveEndpoints(eps)
                runOnUiThread { saveBtn.text = "Saved!" }
            } ?: runOnUiThread { saveBtn.text = "Start Jarvis first to save" }
        }
        root.addView(codeInput)
        root.addView(epInput)
        root.addView(saveBtn)
        setContentView(root)

        toggleBtn.setOnClickListener { toggleVoice() }
    }

    private fun toggleVoice() {
        if (VoiceService.engine != null) {
            startService(Intent(this, VoiceService::class.java).setAction("STOP"))
            toggleBtn.text = "Start Jarvis"
            statusView.text = "● Offline"
        } else {
            if (!hasMicPermission()) { requestMic(); return }
            ContextCompat.startForegroundService(this, Intent(this, VoiceService::class.java))
            toggleBtn.text = "Stop Jarvis"
        }
    }

    private fun hasMicPermission() =
        checkSelfPermission(Manifest.permission.RECORD_AUDIO) == PackageManager.PERMISSION_GRANTED

    private fun requestMic() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.RECORD_AUDIO), 1)
    }

    // ---- VoiceEngine.Ui ----
    override fun onStateChanged(state: String) {
        runOnUiThread {
            statusView.text = when (state) {
                VoiceEngine.STANDBY -> "● Connected — say “Jarvis”"
                VoiceEngine.LISTENING -> "● Listening…"
                VoiceEngine.THINKING -> "● Thinking…"
                VoiceEngine.SPEAKING -> "● Speaking…"
                "RECONNECTING" -> "● Reconnecting…"
                else -> "● $state"
            }
        }
    }

    override fun onTranscript(text: String, partial: Boolean) {
        runOnUiThread {
            transcriptView.text = if (partial) "… $text" else text
        }
    }
}
