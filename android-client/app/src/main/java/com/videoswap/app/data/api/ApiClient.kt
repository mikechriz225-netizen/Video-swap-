package com.videoswap.app.data.api

import android.util.Log
import okhttp3.Interceptor
import okhttp3.MediaType
import okhttp3.OkHttpClient
import okhttp3.RequestBody
import okhttp3.Response
import okhttp3.logging.HttpLoggingInterceptor
import okio.Buffer
import okio.BufferedSink
import okio.ForwardingSink
import okio.Sink
import okio.buffer
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import java.io.IOException
import java.util.concurrent.TimeUnit

/**
 * Singleton client for Retrofit and OkHttpClient with progress tracking and retry logic.
 */
object ApiClient {
    private const val TAG = "ApiClient"

    private var currentBaseUrl: String = "http://localhost:8000"
    private var apiInstance: VideoSwapApi? = null

    /**
     * Interceptor that retries failed network calls with exponential backoff (1s, 2s, 4s).
     */
    private val retryInterceptor = Interceptor { chain ->
        val request = chain.request()
        var response: Response? = null
        var exception: IOException? = null
        val maxRetries = 3

        for (attempt in 0 until maxRetries) {
            try {
                response = chain.proceed(request)
                if (response.isSuccessful || response.code < 500) {
                    return@Interceptor response
                }
            } catch (e: IOException) {
                exception = e
                Log.w(TAG, "Request failed (attempt ${attempt + 1}/$maxRetries): ${e.message}")
            }

            if (attempt < maxRetries - 1) {
                val backoffMs = (1000L * (1 shl attempt))
                try {
                    Thread.sleep(backoffMs)
                } catch (ignored: InterruptedException) {
                    Thread.currentThread().interrupt()
                    break
                }
            }
        }

        response ?: throw (exception ?: IOException("Request failed after $maxRetries attempts"))
    }

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.HEADERS
    }

    private val okHttpClient = OkHttpClient.Builder()
        .connectTimeout(60, TimeUnit.SECONDS)
        .readTimeout(120, TimeUnit.SECONDS)
        .writeTimeout(120, TimeUnit.SECONDS)
        .addInterceptor(retryInterceptor)
        .addInterceptor(loggingInterceptor)
        .retryOnConnectionFailure(true)
        .build()

    @Synchronized
    fun create(baseUrl: String): VideoSwapApi {
        val sanitized = if (baseUrl.endsWith("/")) baseUrl else "$baseUrl/"
        if (apiInstance == null || currentBaseUrl != sanitized) {
            currentBaseUrl = sanitized
            Log.i(TAG, "Creating Retrofit instance with baseUrl: $currentBaseUrl")
            val retrofit = Retrofit.Builder()
                .baseUrl(currentBaseUrl)
                .client(okHttpClient)
                .addConverterFactory(GsonConverterFactory.create())
                .build()
            apiInstance = retrofit.create(VideoSwapApi::class.java)
        }
        return apiInstance!!
    }

    @Synchronized
    fun updateBaseUrl(newUrl: String): VideoSwapApi {
        apiInstance = null
        return create(newUrl)
    }

    fun getClient(): VideoSwapApi {
        return apiInstance ?: create(currentBaseUrl)
    }
}

/**
 * Custom RequestBody wrapper that monitors byte upload progress.
 */
class ProgressRequestBody(
    private val delegate: RequestBody,
    private val onProgress: (bytesWritten: Long, totalBytes: Long) -> Unit
) : RequestBody() {

    override fun contentType(): MediaType? = delegate.contentType()

    override fun contentLength(): Long = delegate.contentLength()

    override fun writeTo(sink: BufferedSink) {
        val countingSink = CountingSink(sink)
        val bufferedSink = countingSink.buffer()
        delegate.writeTo(bufferedSink)
        bufferedSink.flush()
    }

    private inner class CountingSink(delegate: Sink) : ForwardingSink(delegate) {
        private var bytesWritten = 0L

        override fun write(source: Buffer, byteCount: Long) {
            super.write(source, byteCount)
            bytesWritten += byteCount
            onProgress(bytesWritten, contentLength())
        }
    }
}
