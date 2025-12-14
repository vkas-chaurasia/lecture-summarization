# Lecture Summarization

This project was completed as part of the **Large Language Model and It's application** module (3 ECTS credits) at the **ZHAW Center for Artificial Intelligence**.
This repository contains a LLM project that processes video and audio content to generate structured lecture summaries.

## Project Structure

- **lecture_summarization/**: Source code for the CLI application.
  - `transcribe.py`: Audio extraction and transcription.
  - `summarize.py`: Summarization and PDF generation.
  - `cli.py`: CLI entry point and argument parsing.
- **data/**: Data storage for inputs (audio, video) and outputs (transcripts, summaries).
- **docs/**: Documentation and presentations.


## Prerequisites
- **FFmpeg**: Required for audio processing.
  - Mac: `brew install ffmpeg`
  - Linux: `sudo apt install ffmpeg`
  - Windows: Download from official website or use `choco install ffmpeg`

## Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast dependency management.

1. **Clone the repository**:
   ```bash
   git clone https://github.com/vkas-chaurasia/lecture-summarization.git
   cd lecture-summarization
   ```
2. **Install the package**:
   ```bash
   ```bash
   # Create virtual environment (custom name to avoid conflicts)
   uv venv .ls-env
   source .ls-env/bin/activate
   
   # Install in editable mode
   uv pip install -e .
   ```
   *Note: This automatically installs all dependencies and creates the `lecture-cli` command.*
3. **Set up environment variables**:
   - Copy `.env.example` to `.env`.
   - Add your `NVIDIA_API_KEY`.

## Usage

You can run the tool using the `lecture-cli` command.

### 1. Full Pipeline (Processing a Video)
To transcribe a video and immediately generate a summary PDF:
```bash
lecture-cli full-pipeline --video data/videos/my_lecture.mp4
```
*Note: You can also provide a YouTube URL instead of a file path.*

### 2. Transcription Only
To only convert video to text (saved in `data/transcripts/`):
```bash
lecture-cli transcribe --video data/videos/my_lecture.mp4
```

### 3. Summarization Only
If you already have a transcript file and want to generate a PDF summary:
```bash
lecture-cli summarize --transcript data/transcripts/my_lecture_transcript.txt
```

### Arguments
- `--video`: Path to a local video/audio file (MP4, WAV, MP3) OR a YouTube URL.
- `--transcript`: Path to a text file containing the transcript.
- `--model`: (Optional) Whisper model size. Default is `base`. Options: `tiny`, `base`, `small`, `medium`, `large`.

### Example Workflow
1. **Activate Environment**: `source .ls-env/bin/activate`
2. **Download & Process**: `lecture-cli full-pipeline --video "https://www.youtube.com/watch?v=example"`
3. **View Output**: The PDF will be generated in `data/summary/`.
