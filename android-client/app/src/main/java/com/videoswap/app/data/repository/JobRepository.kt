package com.videoswap.app.data.repository

import android.content.Context
import android.net.Uri
import android.util.Log
import com.videoswap.app.data.api.ApiClient
import com.videoswap.app.data.api.VideoSwapApi
import com.videoswap.app.data.local.JobDao
import com.videoswap.app.data.local.JobEntity
import com.videoswap.app.data.local.SettingsStore
import com.videoswap.app.data.models.HealthResponse
import com.videoswap.app.data.models.JobCreateResponse
import com.videoswap.app.data.models.JobStatusResponse
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaTypeOrNull
import okhttp3.MultipartBody
import okhttp3.RequestBody.Companion.asRequestBody
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

/**
 * Repository orchestrating API calls, file transfers, and local Room caching.
 */
class JobRepository(
    private val context: Context,
    private val jobDao: JobDao,
    private val settingsStore: SettingsStore
) {
    companion object {
        private const val TAG = "JobRepository"
    }

    private suspend fun getApi(): VideoSwapApi {
        val serverUrl = settingsStore.serverUrl.first()
        return ApiClient.create(serverUrl)
    }

    suspend fun testConnection(): Result<HealthResponse> = withContext(Dispatchers.IO) {
        try {
            val api = getApi()
            val response = api.healthCheck()
            if (response.isSuccessful && response.body() != null) {
                Result.success(response.body()!!)
            } else {
                Result.failure(Exception("Server returned status ${response.code()}: ${response.message()}"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Health check failed: ${e.message}", e)
            Result.failure(e)
        }
    }

    suspend fun createJob(
        videoFile: File,
        referenceFile: File
    ): Result<JobCreateResponse> = withContext(Dispatchers.IO) {
        try {
            val api = getApi()

            val videoBody = videoFile.asRequestBody("video/mp4".toMediaTypeOrNull())
            val videoPart = MultipartBody.Part.createFormData("source_video", videoFile.name, videoBody)

            val refBody = referenceFile.asRequestBody("image/*".toMediaTypeOrNull())
            val refPart = MultipartBody.Part.createFormData("reference_image", referenceFile.name, refBody)

            val response = api.createJob(videoPart, refPart)
            if (response.isSuccessful && response.body() != null) {
                val body = response.body()!!
                // Cache job in local DB
                jobDao.insertJob(
                    JobEntity(
                        id = body.jobId,
                        status = "PROCESSING",
                        totalChunks = body.totalChunks,
                        completedChunks = 0,
                        progress = 0.0,
                        sourceVideoUri = videoFile.absolutePath,
                        referenceImageUri = referenceFile.absolutePath
                    )
                )
                Result.success(body)
            } else {
                val errorMsg = response.errorBody()?.string() ?: response.message()
                Result.failure(Exception("Server rejected job creation (${response.code()}): $errorMsg"))
            }
        } catch (e: Exception) {
            Log.e(TAG, "Job creation request failed: ${e.message}", e)
            Result.failure(e)
        }
    }

    suspend fun getJobStatus(jobId: String): Result<JobStatusResponse> = withContext(Dispatchers.IO) {
        try {
            val api = getApi()
            val response = api.getJobStatus(jobId)
            if (response.isSuccessful && response.body() != null) {
                val status = response.body()!!
                // Update local DB
                val existing = jobDao.getJob(jobId)
                if (existing != null) {
                    jobDao.updateJob(
                        existing.copy(
                            status = status.status,
                            totalChunks = status.totalChunks,
                            completedChunks = status.completedChunks,
                            progress = status.progressPercent,
                            errorMessage = status.errorMessage,
                            updatedAt = System.currentTimeMillis()
                        )
                    )
                }
                Result.success(status)
            } else {
                Result.failure(Exception("Failed to fetch status: ${response.code()}"))
            }
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    suspend fun downloadResult(
        jobId: String,
        onProgress: ((Float) -> Unit)? = null
    ): Result<File> = withContext(Dispatchers.IO) {
        try {
            val api = getApi()
            val response = api.downloadResult(jobId)
            if (!response.isSuccessful || response.body() == null) {
                return@withContext Result.failure(Exception("Download failed with status ${response.code()}"))
            }

            val body = response.body()!!
            val outputDir = File(context.filesDir, "results").apply { mkdirs() }
            val outputFile = File(outputDir, "${jobId}_final.mp4")

            val totalBytes = body.contentLength()
            var bytesCopied = 0L

            body.byteStream().use { input ->
                FileOutputStream(outputFile).use { output ->
                    val buffer = ByteArray(16 * 1024)
                    var read: Int
                    while (input.read(buffer).also { read = it } != -1) {
                        output.write(buffer, 0, read)
                        bytesCopied += read
                        if (totalBytes > 0 && onProgress != null) {
                            onProgress(bytesCopied.toFloat() / totalBytes)
                        }
                    }
                    output.flush()
                }
            }

            // Update DB with local path
            val existing = jobDao.getJob(jobId)
            if (existing != null) {
                jobDao.updateJob(existing.copy(outputFilePath = outputFile.absolutePath, status = "COMPLETED"))
            }

            Log.i(TAG, "Downloaded result video to ${outputFile.absolutePath} (${outputFile.length()} bytes)")
            Result.success(outputFile)
        } catch (e: Exception) {
            Log.e(TAG, "Error downloading result: ${e.message}", e)
            Result.failure(e)
        }
    }

    suspend fun cancelJob(jobId: String): Result<Unit> = withContext(Dispatchers.IO) {
        try {
            val api = getApi()
            api.cancelJob(jobId)
            val existing = jobDao.getJob(jobId)
            if (existing != null) {
                jobDao.updateJob(existing.copy(status = "CANCELLED"))
            }
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    fun getAllJobs(): Flow<List<JobEntity>> = jobDao.getAllJobs()

    suspend fun deleteJobLocally(jobId: String) = withContext(Dispatchers.IO) {
        jobDao.deleteJob(jobId)
    }

    suspend fun copyUriToTempFile(uri: Uri, prefix: String, suffix: String): File = withContext(Dispatchers.IO) {
        val tempFile = File.createTempFile(prefix, suffix, context.cacheDir)
        context.contentResolver.openInputStream(uri)?.use { input ->
            FileOutputStream(tempFile).use { output ->
                input.copyTo(output)
            }
        } ?: throw IllegalStateException("Could not open input stream for $uri")
        tempFile
    }
}
