package com.videoswap.app.video

import android.util.Log
import com.arthenica.ffmpegkit.FFprobeKit
import java.io.File

data class VideoInfo(
    val width: Int,
    val height: Int,
    val durationSeconds: Double,
    val fps: Double,
    val codec: String,
    val bitrate: Long,
    val hasAudio: Boolean,
    val fileSize: Long
)

/**
 * Utility for probing video parameters via FFprobeKit
 */
object VideoProber {
    private const val TAG = "VideoProber"

    fun probe(filePath: String): VideoInfo {
        Log.i(TAG, "Probing video media information: $filePath")
        val session = FFprobeKit.getMediaInformation(filePath)
        val info = session.mediaInformation
            ?: throw IllegalStateException("Failed to probe video: $filePath. Return code: ${session.returnCode}")

        val videoStream = info.streams?.firstOrNull { it.type == "video" }
            ?: throw IllegalStateException("No video stream found in file: $filePath")
        val audioStream = info.streams?.firstOrNull { it.type == "audio" }

        val width = videoStream.width?.toInt() ?: 0
        val height = videoStream.height?.toInt() ?: 0
        val duration = info.duration?.toDoubleOrNull() ?: 0.0
        val fps = parseFrameRate(videoStream.averageFrameRate ?: "30/1")
        val codec = videoStream.codec ?: "unknown"
        val bitrate = info.bitrate?.toLongOrNull() ?: 0L
        val fileLength = File(filePath).length()

        val videoInfo = VideoInfo(
            width = width,
            height = height,
            durationSeconds = duration,
            fps = fps,
            codec = codec,
            bitrate = bitrate,
            hasAudio = audioStream != null,
            fileSize = fileLength
        )

        Log.i(TAG, "Video Probed: ${width}x${height}, ${duration}s, ${fps}fps, codec=$codec, audio=${audioStream != null}")
        return videoInfo
    }

    private fun parseFrameRate(rateStr: String): Double {
        return try {
            if (rateStr.contains("/")) {
                val parts = rateStr.split("/")
                parts[0].toDouble() / parts[1].toDouble()
            } else {
                rateStr.toDoubleOrNull() ?: 30.0
            }
        } catch (e: Exception) {
            30.0
        }
    }
}
