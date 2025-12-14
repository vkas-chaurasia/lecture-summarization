import os
import subprocess
import yt_dlp
import whisper

def get_video_title(url):
    """Extract utility to get video title from URL."""
    with yt_dlp.YoutubeDL({'quiet': True}) as ydl:
        info = ydl.extract_info(url, download=False)
        return info.get('title', 'audio').strip().replace(' ', '_')

def download_video_from_url(url, output_dir):
    """Download video from YouTube as WAV audio."""
    title = get_video_title(url)
    output_filename = f"{title}.wav"
    output_path = os.path.join(output_dir, output_filename)

    os.makedirs(output_dir, exist_ok=True)

    ydl_opts = {
        'outtmpl': os.path.join(output_dir, title + '.%(ext)s'),
        'format': 'bestaudio/best',
        'quiet': True,
        'no_warnings': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    print(f"Downloaded and saved as: {output_path}")
    return output_path

def convert_mp4_to_wav(input_video, output_audio):
    """Convert local MP4 video file to WAV audio valid for Whisper."""
    output_dir = os.path.dirname(output_audio)
    os.makedirs(output_dir, exist_ok=True)

    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        output_audio
    ]
    subprocess.run(cmd, check=True)
    print(f"Converted {input_video} to {output_audio}")

def transcribe_with_whisper(model_name="base", wav_path="data/audio/GDPR.wav", output_file="data/transcripts/GDPR_transcript.txt"):
    """Run transcription using the openai-whisper Python package."""
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    import torch
    
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Loading Whisper model: {model_name} on device: {device}...")
    model = whisper.load_model(model_name, device=device)
    
    print(f"Transcribing {wav_path}...")
    result = model.transcribe(wav_path)
    text = result["text"]

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(text.strip())

    print(f"Transcription saved to: {output_file}")
    return text.strip()

def process_video_pipeline(input_path, output_dir, model_name="base"):
    """End-to-end pipeline: Input (URL/File) -> WAV -> Transcript."""
    
    # Determine audio output path
    filename = os.path.splitext(os.path.basename(input_path))[0]
    audio_output_path = os.path.join(output_dir, "audio", f"{filename}.wav")
    transcript_output_path = os.path.join(output_dir, "transcripts", f"{filename}_transcript.txt")
    
    # Step 1: Get Audio
    if input_path.startswith("http"):
        print("Detected URL, downloading...")
        chk_path = download_video_from_url(input_path, os.path.dirname(audio_output_path))
        audio_output_path = chk_path # Update path in case title differed
    else:
        print(f"Detected local file: {input_path}")
        convert_mp4_to_wav(input_path, audio_output_path)

    # Step 2: Transcribe
    transcribe_with_whisper(model_name, audio_output_path, transcript_output_path)
    
    return transcript_output_path
