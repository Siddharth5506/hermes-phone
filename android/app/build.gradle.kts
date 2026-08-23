plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.siddharth.hermesphone"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.siddharth.hermesphone"
        minSdk = 26
        targetSdk = 34
        versionCode = 1
        versionName = "0.1.0"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    // WebSocket client with reconnection support
    implementation("com.squareup.okhttp3:okhttp:4.12.0")
    // ONNX Runtime for openWakeWord inference on-device
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.18.0")
}
