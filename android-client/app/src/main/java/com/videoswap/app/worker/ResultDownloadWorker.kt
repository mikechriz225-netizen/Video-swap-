package com.videoswap.app.worker

import android.content.Context
import android.util.Log
import androidx.work.BackoffPolicy
import androidx.work.Constraints
import androidx.work.CoroutineWorker
import androidx.work.NetworkType
import androidx.work.OneTimeWorkRequest
import androidx.work.OneTimeWorkRequestBuilder
import androidx.work.WorkerParameters
import androidx.work.workDataOf
import com.videoswap.app.data.api.ApiClient
import com.videoswap.app.data.local.JobDatabase
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.TimeUnit

/**
 * Background-safe WorkManager worker for downloading the final stitched video result.
 */
class ResultDownloadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val TAG = "ResultDownloadWorker"
        const val KEY_JOB_ID = "job_id"
        const val KEY_SERVER_URL = "server_url"
        const val KEY_OUTPUT_PATH = "output_path"
        const val KEY_DOWNLOAD_PROGRESS = "download_progress"

        fun buildRequest(
            jobId: String,
            serverUrl: String
        ): OneTimeWorkRequest {
            val data = workDataOf(
                KEY_JOB_ID to jobId,
                KEY_SERVER_URL to serverUrl
            )
            return OneTimeWorkRequestBuilder<ResultDownloadWorker>()
                .setInputData(data)
                .setConstraints(
                    Constraints.Builder()
                        .setRequiredNetworkType(NetworkType.CONNECTED)
                        .build()
                )
                .setBackoffCriteria(
                    BackoffPolicy.EXPONENTIAL,
                    15_000L,
                    TimeUnit.MILLISECONDS
                )
                .addTag("download_job_$jobId")
                .build()
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val jobId = inputData.getString(KEY_JOB_ID) ?: return@withContext Result.failure()
        val serverUrl = inputData.getString(KEY_SERVER_URL) ?: return@withContext Result.failure()

        Log.i(TAG, "[$jobId] Starting background download of final stitched video")

        try {
            val api = ApiClient.create(serverUrl)
            val response = api.downloadResult(jobId)

            if (!response.isSuccessful || response.body() == null) {
                Log.w(TAG, "[$jobId] Download response failed: code=${response.code()}")
                if (runAttemptCount < 3) Result.retry() else Result.failure()
            }

            val body = response.body()!!
            val outputDir = File(applicationContext.filesDir, "results").apply { mkdirs() }
            val outputFile = File(outputDir, "${jobId}_final.mp4")

            val totalBytes = body.contentLength()
            var bytesCopied = 0L

            body.byteStream().use { input ->
                FileOutputStream(outputFile).use { output ->
                    val buffer = ByteArray(32 * 1024)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesCopied += read
                        if (totalBytes > 0) {
                            val pct = (bytesCopied.toFloat() / totalBytes * 100f)
                            setProgressAsync(workDataOf(KEY_DOWNLOAD_PROGRESS to pct))
                        }
                    }
                    output.flush()
                }
            }

            // Update Room database with path & completed status
            val db = JobDatabase.getInstance(applicationContext)
            val existing = db.jobDao().getJob(jobId)
            if (existing != null) {
                db.jobDao().updateJob(
                    existing.copy(
                        status = "COMPLETED",
                        outputFilePath = outputFile.absolutePath,
                        progress = 100.0,
                        updatedAt = System.currentTimeMillis()
                    )
                )
            }

            Log.i(TAG, "[$jobId] Successfully downloaded final video: ${outputFile.absolutePath} (${outputFile.length()} bytes)")
            Result.success(workDataOf(KEY_OUTPUT_PATH to outputFile.absolutePath))
        } catch (e: Exception) {
            Log.e(TAG, "[$jobId] Exception during download: ${e.message}", e)
            if (runAttemptCount < 3) Result.retry() else Result.failure()
        }
    }
}
