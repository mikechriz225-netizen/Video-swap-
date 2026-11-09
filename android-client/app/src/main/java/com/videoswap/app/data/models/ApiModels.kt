package com.videoswap.app.data.models

import com.google.gson.annotations.SerializedName

/**
 * Health check response from backend
 */
data class HealthResponse(
    @SerializedName("status") val status: String,
    @SerializedName("gpu") val gpu: String? = null,
    @SerializedName("vram_free_gb") val vramFreeGb: Double? = 0.0
)

/**
 * Response when a new job is created
 */
data class JobCreateResponse(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("message") val message: String,
    @SerializedName("total_chunks") val totalChunks: Int,
    @SerializedName("estimated_duration_seconds") val estimatedDurationSeconds: Double
)

/**
 * Response when an individual chunk is uploaded
 */
data class ChunkUploadResponse(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("chunk_index") val chunkIndex: Int,
    @SerializedName("chunk_id") val chunkId: String,
    @SerializedName("status") val status: String
)

/**
 * Progress of an individual chunk within a job
 */
data class ChunkProgress(
    @SerializedName("chunk_index") val chunkIndex: Int,
    @SerializedName("chunk_id") val chunkId: String,
    @SerializedName("status") val status: String,
    @SerializedName("progress_percent") val progressPercent: Double = 0.0,
    @SerializedName("error_message") val errorMessage: String? = null
)

/**
 * Full job status report
 */
data class JobStatusResponse(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("status") val status: String,
    @SerializedName("total_chunks") val totalChunks: Int = 0,
    @SerializedName("completed_chunks") val completedChunks: Int = 0,
    @SerializedName("failed_chunks") val failedChunks: Int = 0,
    @SerializedName("progress_percent") val progressPercent: Double = 0.0,
    @SerializedName("output_url") val outputUrl: String? = null,
    @SerializedName("error_message") val errorMessage: String? = null,
    @SerializedName("created_at") val createdAt: String? = null,
    @SerializedName("updated_at") val updatedAt: String? = null,
    @SerializedName("chunks") val chunks: List<ChunkProgress> = emptyList()
)

/**
 * Response for uploading character reference image(s)
 */
data class ReferenceUploadResponse(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("reference_id") val referenceId: String,
    @SerializedName("message") val message: String
)

/**
 * Inference configuration payload
 */
data class ProcessingConfig(
    @SerializedName("inference_steps") val inferenceSteps: Int = 30,
    @SerializedName("guidance_scale") val guidanceScale: Double = 7.5,
    @SerializedName("controlnet_scale") val controlnetScale: Double = 0.8,
    @SerializedName("ip_adapter_scale") val ipAdapterScale: Double = 0.6,
    @SerializedName("enable_lip_sync") val enableLipSync: Boolean = true,
    @SerializedName("enable_temporal_consistency") val enableTemporalConsistency: Boolean = true
)

/**
 * Cancellation response
 */
data class CancelResponse(
    @SerializedName("job_id") val jobId: String,
    @SerializedName("message") val message: String
)

/**
 * Standard error response
 */
data class ErrorResponse(
    @SerializedName("detail") val detail: String
)
