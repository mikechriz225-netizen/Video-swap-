package com.videoswap.app

import android.app.Application
import android.util.Log
import androidx.work.Configuration

/**
 * Android Application entry point.
 * Initializes WorkManager configuration and sets up application lifecycle singletons.
 */
class VideoSwapApp : Application(), Configuration.Provider {

    override fun onCreate() {
        super.onCreate()
        instance = this
        Log.i(TAG, "VideoSwapApp initialized on Unisoc T7300 architecture")
    }

    override val workManagerConfiguration: Configuration
        get() = Configuration.Builder()
            .setMinimumLoggingLevel(if (BuildConfig.DEBUG) Log.DEBUG else Log.INFO)
            .build()

    companion object {
        private const val TAG = "VideoSwapApp"
        lateinit var instance: VideoSwapApp
            private set
    }
}
