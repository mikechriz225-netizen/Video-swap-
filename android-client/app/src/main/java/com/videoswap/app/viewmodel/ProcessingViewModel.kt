package com.videoswap.app.viewmodel

import android.app.Application
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.videoswap.app.data.local.JobDatabase
import com.videoswap.app.data.local.SettingsStore
import com.videoswap.app.data.models.JobStatusResponse
import com.videoswap.app.data.repository.JobRepository
import com.videoswap.app.ui.components.ChunkProgressUi
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

data class ProcessingUiState(
    val status: String = "PENDING",
    val progress: Float = 0f,
    val totalChunks: Int = 0,
    val completedChunks: Int = 0,
    val chunks: List<ChunkProgressUi> = emptyList(),
    val isDownloading: Boolean = false,
    val downloadProgress: Float = 0f,
    val resultFilePath: String? = null,
    val errorMessage: String? = null
)

class ProcessingViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: JobRepository
    private var pollingJob: Job? = null

    private val _uiState = MutableStateFlow(ProcessingUiState())
    val uiState: StateFlow<ProcessingUiState> = _uiState.asStateFlow()

    init {
        val db = JobDatabase.getInstance(application)
        val settings = SettingsStore(application)
        repository = JobRepository(application, db.jobDao(), settings)
    }

    fun startPolling(jobId: String) {
        pollingJob?.cancel()
        pollingJob = viewModelScope.launch {
            while (isActive) {
                val result = repository.getJobStatus(jobId)
                result.onSuccess { statusResponse ->
                    updateStatus(statusResponse)

                    if (statusResponse.status == "COMPLETED") {
                        triggerDownload(jobId)
                        return@launch
                    } else if (statusResponse.status == "FAILED") {
                        _uiState.value = _uiState.value.copy(
                            status = "FAILED",
                            errorMessage = statusResponse.errorMessage ?: "Pipeline processing failed on server"
                        )
                        return@launch
                    }
                }.onFailure { err ->
                    Log.w("ProcessingViewModel", "Status poll error: ${err.message}")
                }

                delay(3000)
            }
        }
    }

    private fun updateStatus(status: JobStatusResponse) {
        val uiChunks = status.chunks.map { chunk ->
            ChunkProgressUi(
                index = chunk.chunkIndex,
                status = chunk.status,
                progress = chunk.progressPercent.toFloat()
            )
        }

        _uiState.value = _uiState.value.copy(
            status = status.status,
            progress = status.progressPercent.toFloat(),
            totalChunks = status.totalChunks,
            completedChunks = status.completedChunks,
            chunks = uiChunks
        )
    }

    fun triggerDownload(jobId: String) {
        viewModelScope.launch {
            _uiState.value = _uiState.value.copy(isDownloading = true, downloadProgress = 0f)
            val downloadResult = repository.downloadResult(jobId) { pct ->
                _uiState.value = _uiState.value.copy(downloadProgress = pct * 100f)
            }

            downloadResult.onSuccess { file ->
                _uiState.value = _uiState.value.copy(
                    isDownloading = false,
                    resultFilePath = file.absolutePath
                )
            }.onFailure { err ->
                _uiState.value = _uiState.value.copy(
                    isDownloading = false,
                    errorMessage = "Download error: ${err.message}"
                )
            }
        }
    }

    fun cancelJob(jobId: String, onCancelled: () -> Unit) {
        pollingJob?.cancel()
        viewModelScope.launch {
            withContext(Dispatchers.IO) {
                repository.cancelJob(jobId)
            }
            onCancelled()
        }
    }

    override fun onCleared() {
        super.onCleared()
        pollingJob?.cancel()
    }
}
