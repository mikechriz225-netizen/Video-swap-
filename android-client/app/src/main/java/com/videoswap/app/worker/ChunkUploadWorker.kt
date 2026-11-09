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
import com.videoswap.app.data.api.ProgressRequestBody
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.util.concurrent.TimeUnit

/**
 * Background-safe WorkManager worker for uploading an individual video chunk.
 */
class ChunkUploadWorker(
    context: Context,
    params: WorkerParameters
) : CoroutineWorker(context, params) {

    companion object {
        const val TAG = "ChunkUploadWorker"
        const val KEY_JOB_ID = "job_id"
        const val KEY_CHUNK_INDEX = "chunk_index"
        const val KEY_CHUNK_PATH = "chunk_path"
        const val KEY_SERVER_URL = "server_url"
        const val KEY_UPLOAD_PROGRESS = "upload_progress"

        fun buildRequest(
            jobId: String,
            chunkIndex: Int,
            chunkPath: String,
            serverUrl: String
        ): OneTimeWorkRequest {
            val data = workDataOf(
                KEY_JOB_ID to jobId,
                KEY_CHUNK_INDEX to chunkIndex,
                KEY_CHUNK_PATH to chunkPath,
                KEY_SERVER_URL to serverUrl
            )
            return OneTimeWorkRequestBuilder<ChunkUploadWorker>()
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
                .addTag("upload_job_$jobId")
                .build()
        }
    }

    override suspend fun doWork(): Result = withContext(Dispatchers.IO) {
        val jobId = inputData.getString(KEY_JOB_ID) ?: return@withContext Result.failure()
        val chunkIndex = inputData.getInt(KEY_CHUNK_INDEX, -1)
        val chunkPath = inputData.getString(KEY_CHUNK_PATH) ?: return@withContext Result.failure()
        val serverUrl = inputData.getString(KEY_SERVER_URL) ?: return@withContext Result.failure()

        Log.i(TAG, "[$jobId] Starting upload for chunk $chunkIndex from $chunkPath")

        val file = File(chunkPath)
        if (!file.exists() || file.length() == 0L) {
            Log.e(TAG, "[$jobId] Chunk file does not exist or empty: $chunkPath")
            return@withContext Result.failure()
        }

        try {
            val api = ApiClient.create(serverUrl)

            val baseBody = file.asRequestBody("video/mp4".toMediaTypeOrNull())
            val progressBody = ProgressRequestBody(baseBody) { bytesWritten, totalBytes ->
                if (totalBytes > 0) {
                    val pct = (bytesWritten.toDouble() / totalBytes * 100.0).toFloat()
                    setProgressAsync(workDataOf(KEY_UPLOAD_PROGRESS to pct))
                }
            }

            val part = MultipartBody.Part.createFormData("chunk_file", file.name, progressBody)
            val response = api.uploadChunk(jobId, chunkIndex, part)

            if (response.isSuccessful && response.body() != null) {
                Log.i(TAG, "[$jobId] Chunk $chunkIndex uploaded successfully (status=${response.body()?.status})")
                Result.success(
                    workDataOf(
                        "job_id" to jobId,
                        "chunk_index" to chunkIndex,
                        "status" to (response.body()?.status ?: "UPLOADED")
                    )
                )
            } else {
                Log.w(TAG, "[$jobId] Chunk $chunkIndex upload failed: code=${response.code()}")
                if (runAttemptCount < 3) Result.retry() else Result.failure()
            }
        } catch (e: Exception) {
            Log.e(TAG, "[$jobId] Exception uploading chunk $chunkIndex: ${e.message}", e)
            if (runAttemptCount < 3) Result.retry() else Result.failure()
        }
    }
}
