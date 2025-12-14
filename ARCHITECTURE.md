# Architecture & Logic Flow

This document explains the internal logic of the Lecture Summarization tool.

## High-Level Data Flow

```mermaid
graph TD
    A[Input: Video/URL] -->|ffmpeg / yt-dlp| B[Audio File (.wav)]
    B -->|Whisper (Local AI)| C[Raw Transcript (.txt)]
    C -->|LangChain Splitter| D[Text Chunks]
    D -->|Llama 3 (NVIDIA NIM)| E[Topic Extraction]
    E -->|Llama 3 (NVIDIA NIM)| F[Section Summaries]
    F -->|fpdf| G[Final PDF Output]
```

## 1. Transcription Layer (`transcribe.py`)
This component is responsible for converting raw media into text.

*   **Input Handling**:
    *   If a **YouTube URL** is provided, `yt-dlp` extracts the audio stream as a WAV file.
    *   If a **Local Video** is provided, `ffmpeg` strips the video track and converts the audio to 16kHz WAV (optimal for Whisper).
*   **Speech-to-Text**:
    *   Uses OpenAI's **Whisper** model locally.
    *   **Optimization**: Automatically detects Apple Silicon (MPS) to run up to 5x faster on Mac.

## 2. Summarization Layer (`summarize.py`)
This component transforms a wall of text into structured notes.

*   **Chunking**: The transcript is unlimited in length, but LLMs have context limits. We use `RecursiveCharacterTextSplitter` to break the text into 1000-token chunks with overlap to preserve context.
*   **Step A: Topic Extraction**:
    *   The LLM reads each chunk and identifies logical sections (e.g., "Intro", "Neural Network Basics", "Backpropagation").
    *   It effectively creates a "Table of Contents" on the fly.
*   **Step B: Content Summarization**:
    *   For each identified topic, the LLM generates a detailed summary.
    *   It is strictly instructed to preserve technical definitions, laws, and examples while removing filler words.

## 3. Presentation Layer (`summarize.py`)
This component generates the final readable artifact.

*   **PDF Generation**:
    *   Uses `fpdf` to construct a document.
    *   Dynamically inserts headers for each Topic identified in Step 2.
    *   Renders the AI-generated summaries as formatted text blocks under the corresponding headers.

## Technology Stack
*   **Engine**: Python 3.9+
*   **Audio AI**: OpenAI Whisper (Local)
*   **Text AI**: Llama 3.1 405B (via NVIDIA NIM)
*   **Orchestration**: LangChain
*   **Media Processing**: FFmpeg
