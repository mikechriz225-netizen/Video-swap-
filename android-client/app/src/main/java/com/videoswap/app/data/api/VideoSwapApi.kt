package com.videoswap.app.data.api

import com.videoswap.app.data.models.CancelResponse
import com.videoswap.app.data.models.ChunkUploadResponse
import com.videoswap.app.data.models.HealthResponse
import com.videoswap.app.data.models.JobCreateResponse
import com.videoswap.app.data.models.JobStatusResponse
import com.videoswap.app.data.models.ReferenceUploadResponse
import okhttp3.MultipartBody
import okhttp3.RequestBody
import okhttp3.ResponseBody
import retrofit2.Response
import retrofit2.http.DELETE
import retrofit2.http.GET
import retrofit2.http.Multipart
import retrofit2.http.POST
import retrofit2.http.Part
import retrofit2.http.Path
import retrofit2.http.Streaming

/**
 * Retrofit service interface for the Video Swap Pipeline API
 */
interface VideoSwapApi {

    @GET("health")
    suspend fun healthCheck(): Response<HealthResponse>

    @Multipart
    @POST("api/v1/jobs/create")
    suspend fun createJob(
        @Part sourceVideo: MultipartBody.Part,
        @Part referenceImage: MultipartBody.Part,
        @Part("config") config: RequestBody? = null
    ): Response<JobCreateResponse>

    @Multipart
    @POST("api/v1/jobs/{jobId}/chunks/{chunkIndex}")
    suspend fun uploadChunk(
        @Path("jobId") jobId: String,
        @Path("chunkIndex") chunkIndex: Int,
        @Part chunkFile: MultipartBody.Part
    ): Response<ChunkUploadResponse>

    @Multipart
    @POST("api/v1/jobs/{jobId}/reference")
    suspend fun uploadReference(
        @Path("jobId") jobId: String,
        @Part files: List<MultipartBody.Part>
    ): Response<ReferenceUploadResponse>

    @GET("api/v1/jobs/{jobId}/status")
    suspend fun getJobStatus(
        @Path("jobId") jobId: String
    ): Response<JobStatusResponse>

    @GET("api/v1/jobs/{jobId}/download")
    @Streaming
    suspend fun downloadResult(
        @Path("jobId") jobId: String
    ): Response<ResponseBody>

    @GET("api/v1/jobs/{jobId}/chunks/{chunkIndex}/download")
    @Streaming
    suspend fun downloadChunk(
        @Path("jobId") jobId: String,
        @Path("chunkIndex") chunkIndex: Int
    ): Response<ResponseBody>

    @DELETE("api/v1/jobs/{jobId}")
    suspend fun cancelJob(
        @Path("jobId") jobId: String
    ): Response<CancelResponse>
}
