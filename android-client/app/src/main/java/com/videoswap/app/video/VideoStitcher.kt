package com.videoswap.app.video

import android.util.Log
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFprobeKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File
import java.io.FileWriter

/**
 * Client-side video stitching and verification using Mobile-FFmpeg.
 * Concat chunks into a single preview MP4 or verify downloaded videos.
 */
object VideoStitcher {
    private const val TAG = "VideoStitcher"

    /**
     * Concatenates a sequential list of video chunks into a single output video.
     */
    fun stitchChunks(
        chunkPaths: List<String>,
        outputPath: String,
        onProgress: ((current: Int, total: Int) -> Unit)? = null
    ): Boolean {
        if (chunkPaths.isEmpty()) {
            Log.e(TAG, "No chunk paths provided for stitching")
            return false
        }

        val outputFile = File(outputPath)
        outputFile.parentFile?.mkdirs()

        // Create concat manifest file
        val concatManifest = File.createTempFile("concat_manifest", ".txt", outputFile.parentFile)
        try {
            FileWriter(concatManifest).use { writer ->
                for (path in chunkPaths) {
                    writer.write("file '$path'\n")
                }
            }

            Log.i(TAG, "Stitching ${chunkPaths.size} chunks into $outputPath via concat demuxer...")
            val cmd = "-y -f concat -safe 0 -i \"${concatManifest.absolutePath}\" -c copy \"$outputPath\""
            val session = FFmpegKit.execute(cmd)

            if (ReturnCode.isSuccess(session.returnCode)) {
                Log.i(TAG, "Stitching completed successfully: $outputPath (${outputFile.length()} bytes)")
                onProgress?.invoke(chunkPaths.size, chunkPaths.size)
                return true
            } else {
                Log.w(TAG, "Lossless concat failed, attempting re-encode concat: ${session.failStackTrace}")
                val fallbackCmd = "-y -f concat -safe 0 -i \"${concatManifest.absolutePath}\" -c:v libx264 -preset ultrafast -c:a aac \"$outputPath\""
                val fallbackSession = FFmpegKit.execute(fallbackCmd)
                val success = ReturnCode.isSuccess(fallbackSession.returnCode)
                if (success) {
                    Log.i(TAG, "Fallback re-encoded stitch successful: $outputPath")
                    onProgress?.invoke(chunkPaths.size, chunkPaths.size)
                } else {
                    Log.e(TAG, "Both concat approaches failed: ${fallbackSession.failStackTrace}")
                }
                return success
            }
        } catch (e: Exception) {
            Log.e(TAG, "Exception during video stitching: ${e.message}", e)
            return false
        } finally {
            concatManifest.delete()
        }
    }

    /**
     * Verifies that the video file exists, has non-zero size, and can be probed by FFprobe.
     */
    fun verifyVideo(videoPath: String): Boolean {
        val file = File(videoPath)
        if (!file.exists() || file.length() == 0L) {
            return false
        }
        val session = FFprobeKit.getMediaInformation(videoPath)
        val info = session.mediaInformation
        val hasVideo = info?.streams?.any { it.type == "video" } == true
        val duration = info?.duration?.toDoubleOrNull() ?: 0.0
        return hasVideo && duration > 0.0
    }
}
