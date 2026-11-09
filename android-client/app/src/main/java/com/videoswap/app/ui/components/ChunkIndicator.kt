package com.videoswap.app.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import com.videoswap.app.ui.theme.CyberCyan
import com.videoswap.app.ui.theme.ErrorCrimson
import com.videoswap.app.ui.theme.MintGreen
import com.videoswap.app.ui.theme.NeonPurple

/**
 * Horizontal row of status indicator circles representing each chunk's progress.
 */
@Composable
fun ChunkStatusRow(
    chunks: List<ChunkProgressUi>,
    modifier: Modifier = Modifier
) {
    if (chunks.isEmpty()) return

    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = "Chunk Pipeline State",
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(horizontal = 16.dp, vertical = 4.dp)
        )

        LazyRow(
            modifier = Modifier.fillMaxWidth(),
            contentPadding = PaddingValues(horizontal = 16.dp, vertical = 8.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            items(chunks) { chunk ->
                ChunkDot(chunk = chunk)
            }
        }
    }
}

@Composable
private fun ChunkDot(chunk: ChunkProgressUi) {
    val (dotColor, borderColor) = when (chunk.status.uppercase()) {
        "COMPLETED" -> Pair(MintGreen, MintGreen.copy(alpha = 0.5f))
        "PROCESSING" -> Pair(NeonPurple, CyberCyan)
        "UPLOADED", "QUEUED" -> Pair(CyberCyan, CyberCyan.copy(alpha = 0.5f))
        "FAILED" -> Pair(ErrorCrimson, ErrorCrimson.copy(alpha = 0.5f))
        else -> Pair(Color.Gray.copy(alpha = 0.4f), Color.Transparent)
    }

    Column(horizontalAlignment = Alignment.CenterHorizontally) {
        Box(
            modifier = Modifier
                .size(24.dp)
                .clip(CircleShape)
                .background(dotColor)
                .border(2.dp, borderColor, CircleShape),
            contentAlignment = Alignment.Center
        ) {
            Text(
                text = "${chunk.index + 1}",
                style = MaterialTheme.typography.labelSmall.copy(
                    fontSize = 10.sp,
                    fontWeight = FontWeight.Bold,
                    color = Color.White
                )
            )
        }
        Spacer(modifier = Modifier.height(2.dp))
        Text(
            text = chunk.status.take(4).lowercase(),
            style = MaterialTheme.typography.labelSmall.copy(fontSize = 8.sp),
            color = MaterialTheme.colorScheme.onSurfaceVariant
        )
    }
}
