package com.videoswap.app.viewmodel

import android.app.Application
import android.net.Uri
import android.util.Log
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import com.videoswap.app.data.local.JobDatabase
import com.videoswap.app.data.local.SettingsStore
import com.videoswap.app.data.repository.JobRepository
import com.videoswap.app.video.VideoChunker
import com.videoswap.app.video.VideoInfo
import com.videoswap.app.video.VideoProber
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

data class HomeUiState(
    val selectedVideoUri: Uri? = null,
    val selectedReferenceUri: Uri? = null,
    val videoInfo: VideoInfo? = null,
    val isChunking: Boolean = false,
    val isUploading: Boolean = false,
    val chunkProgress: String = "",
    val errorMessage: String? = null
)

class HomeViewModel(application: Application) : AndroidViewModel(application) {

    private val repository: JobRepository
    private val _uiState = MutableStateFlow(HomeUiState())
    val uiState: StateFlow<HomeUiState> = _uiState.asStateFlow()

    init {
        val db = JobDatabase.getInstance(application)
        val settings = SettingsStore(application)
        repository = JobRepository(application, db.jobDao(), settings)
    }

    fun setVideoUri(uri: Uri) {
        viewModelScope.launch {
            try {
                _uiState.value = _uiState.value.copy(errorMessage = null)
                val tempFile = repository.copyUriToTempFile(uri, "source_probe_", ".mp4")
                val info = withContext(Dispatchers.IO) {
                    VideoProber.probe(tempFile.absolutePath)
                }

                if (info.durationSeconds > 600.0) {
                    _uiState.value = _uiState.value.copy(
                        errorMessage = "Video length (${info.durationSeconds.toInt()}s) exceeds the maximum 10-minute limit."
                    )
                    tempFile.delete()
                    return@launch
                }

                _uiState.value = _uiState.value.copy(
                    selectedVideoUri = uri,
                    videoInfo = info
                )
            } catch (e: Exception) {
                Log.e("HomeViewModel", "Failed to probe selected video: ${e.message}", e)
                _uiState.value = _uiState.value.copy(
                    selectedVideoUri = uri,
                    errorMessage = "Unable to analyze video parameters: ${e.message}"
                )
            }
        }
    }

    fun setReferenceUri(uri: Uri) {
        _uiState.value = _uiState.value.copy(
            selectedReferenceUri = uri,
            errorMessage = null
        )
    }

    fun clearError() {
        _uiState.value = _uiState.value.copy(errorMessage = null)
    }

    fun startSwap(onJobCreated: (String) -> Unit) {
        val videoUri = _uiState.value.selectedVideoUri ?: return
        val refUri = _uiState.value.selectedReferenceUri ?: return

        viewModelScope.launch {
            try {
                _uiState.value = _uiState.value.copy(
                    isChunking = true,
                    chunkProgress = "Preparing media files..."
                )

                val context = getApplication<Application>()
                val videoFile = repository.copyUriToTempFile(videoUri, "source_", ".mp4")
                val refFile = repository.copyUriToTempFile(refUri, "ref_", ".png")

                // Analyze and chunk if needed
                val info = withContext(Dispatchers.IO) {
                    VideoProber.probe(videoFile.absolutePath)
                }

                if (info.durationSeconds > 10.0) {
                    _uiState.value = _uiState.value.copy(chunkProgress = "Slicing video at I-frame boundaries...")
                    val chunkDir = File(context.cacheDir, "chunks_${System.currentTimeMillis()}")
                    withContext(Dispatchers.IO) {
                        VideoChunker.chunkVideo(
                            videoPath = videoFile.absolutePath,
                            outputDir = chunkDir,
                            minChunkSec = 5.0,
                            maxChunkSec = 10.0
                        ) { curr, total ->
                            _uiState.value = _uiState.value.copy(chunkProgress = "Sliced chunk $curr / $total")
                        }
                    }
                }

                _uiState.value = _uiState.value.copy(
                    isChunking = false,
                    isUploading = true,
                    chunkProgress = "Dispatching job to server pipeline..."
                )

                val result = repository.createJob(videoFile, refFile)
                result.onSuccess { response ->
                    _uiState.value = _uiState.value.copy(
                        isUploading = false,
                        chunkProgress = ""
                    )
                    onJobCreated(response.jobId)
                }.onFailure { err ->
                    _uiState.value = _uiState.value.copy(
                        isChunking = false,
                        isUploading = false,
                        errorMessage = err.message ?: "Failed to dispatch swap job to server"
                    )
                }
            } catch (e: Exception) {
                Log.e("HomeViewModel", "Swap initialization error: ${e.message}", e)
                _uiState.value = _uiState.value.copy(
                    isChunking = false,
                    isUploading = false,
                    errorMessage = e.message
                )
            }
        }
    }
}
