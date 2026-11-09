package com.videoswap.app.video

import android.util.Log
import com.arthenica.ffmpegkit.FFmpegKit
import com.arthenica.ffmpegkit.FFprobeKit
import com.arthenica.ffmpegkit.ReturnCode
import java.io.File

data class ChunkInfo(
    val index: Int,
    val path: String,
    val startTime: Double,
    val endTime: Double,
    val duration: Double,
    val fileSizeBytes: Long
)

/**
 * Keyframe-aware video slicing engine using Mobile-FFmpeg (FFmpegKit).
 * Slices arbitrary length videos into 5-to-10 second micro-chunks at exact I-frames.
 */
object VideoChunker {
    private const val TAG = "VideoChunker"

    /**
     * Query all I-frame (keyframe) timestamps in seconds using ffprobe.
     */
    fun findKeyframes(videoPath: String): List<Double> {
        Log.i(TAG, "Searching for keyframes in: $videoPath")
        val cmd = "-select_streams v -show_frames -show_entries frame=pkt_pts_time,pict_type -of csv $videoPath"
        val session = FFprobeKit.execute(cmd)

        val output = session.output ?: ""
        val keyframes = mutableListOf<Double>()

        for (line in output.lines()) {
            val parts = line.split(",")
            if (parts.size >= 3 && parts[0] == "frame") {
                val ptsTime = parts[1].toDoubleOrNull()
                val pictType = parts[2]
                if (pictType == "I" && ptsTime != null) {
                    keyframes.add(ptsTime)
                }
            }
        }

        if (keyframes.isEmpty() || keyframes.first() != 0.0) {
            keyframes.add(0, 0.0)
        }

        val sorted = keyframes.distinct().sorted()
        Log.i(TAG, "Discovered ${sorted.size} keyframes: ${sorted.take(5)}...")
        return sorted
    }

    /**
     * Split video into 5-to-10 second micro-chunks at nearest keyframes.
     */
    fun chunkVideo(
        videoPath: String,
        outputDir: File,
        minChunkSec: Double = 5.0,
        maxChunkSec: Double = 10.0,
        onProgress: ((completedChunks: Int, totalEstimate: Int) -> Unit)? = null
    ): List<ChunkInfo> {
        outputDir.mkdirs()

        val info = VideoProber.probe(videoPath)
        val totalDuration = info.durationSeconds

        if (totalDuration <= maxChunkSec) {
            // No splitting needed; copy or link single chunk
            val singleChunk = File(outputDir, "chunk_0000.mp4")
            File(videoPath).copyTo(singleChunk, overwrite = true)
            return listOf(
                ChunkInfo(
                    index = 0,
                    path = singleChunk.absolutePath,
                    startTime = 0.0,
                    endTime = totalDuration,
                    duration = totalDuration,
                    fileSizeBytes = singleChunk.length()
                )
            )
        }

        val keyframes = findKeyframes(videoPath)
        val splitPoints = mutableListOf<Double>()
        splitPoints.add(0.0)

        var lastSplit = 0.0
        for (kf in keyframes) {
            val elapsed = kf - lastSplit
            if (elapsed >= minChunkSec) {
                splitPoints.add(kf)
                lastSplit = kf
            }
        }

        if (splitPoints.last() < totalDuration) {
            splitPoints.add(totalDuration)
        }

        val chunks = mutableListOf<ChunkInfo>()
        val totalChunks = splitPoints.size - 1

        for (i in 0 until totalChunks) {
            val start = splitPoints[i]
            val end = splitPoints[i + 1]
            val chunkDuration = end - start
            val chunkFile = File(outputDir, String.format("chunk_%04d.mp4", i))

            val ffmpegCmd = "-y -ss $start -to $end -i \"$videoPath\" -c copy -avoid_negative_ts 1 \"${chunkFile.absolutePath}\""
            Log.d(TAG, "Executing chunk command: $ffmpegCmd")

            val session = FFmpegKit.execute(ffmpegCmd)
            if (ReturnCode.isSuccess(session.returnCode)) {
                val chunkInfo = ChunkInfo(
                    index = i,
                    path = chunkFile.absolutePath,
                    startTime = start,
                    endTime = end,
                    duration = chunkDuration,
                    fileSizeBytes = chunkFile.length()
                )
                chunks.add(chunkInfo)
                Log.i(TAG, "Chunk $i successfully created: ${chunkDuration}s (${chunkFile.length()} bytes)")
                onProgress?.invoke(i + 1, totalChunks)
            } else {
                Log.e(TAG, "Failed to create chunk $i: returnCode=${session.returnCode}, logs=${session.failStackTrace}")
                // Fallback with re-encoding if copy fails
                val reencodeCmd = "-y -ss $start -to $end -i \"$videoPath\" -c:v libx264 -preset ultrafast -c:a aac \"${chunkFile.absolutePath}\""
                val retrySession = FFmpegKit.execute(reencodeCmd)
                if (ReturnCode.isSuccess(retrySession.returnCode)) {
                    chunks.add(
                        ChunkInfo(i, chunkFile.absolutePath, start, end, chunkDuration, chunkFile.length())
                    )
                    onProgress?.invoke(i + 1, totalChunks)
                } else {
                    throw RuntimeException("Fatal error chunking video at index $i: ${retrySession.failStackTrace}")
                }
            }
        }

        return chunks
    }
}
