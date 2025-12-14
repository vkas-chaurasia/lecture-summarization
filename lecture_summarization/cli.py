import argparse
import os
import sys
from dotenv import load_dotenv

# Load env before importing modules that might need it (or handle inside)
load_dotenv()

from .transcribe import process_video_pipeline
from .summarize import generate_summary_pipeline

def main():
    parser = argparse.ArgumentParser(description="Lecture Summarization CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: full-pipeline
    parser_full = subparsers.add_parser("full-pipeline", help="Run transcription and summarization")
    parser_full.add_argument("--video", required=True, help="Path to video/audio file or YouTube URL")
    parser_full.add_argument("--model", default="base.en", help="Whisper model name (default: base.en)")

    # Command: transcribe
    parser_transcribe = subparsers.add_parser("transcribe", help="Run only transcription")
    parser_transcribe.add_argument("--video", required=True, help="Path to video/audio file or YouTube URL")
    parser_transcribe.add_argument("--model", default="base.en", help="Whisper model name (default: base.en)")

    # Command: summarize
    parser_summary = subparsers.add_parser("summarize", help="Run only summarization from transcript")
    parser_summary.add_argument("--transcript", required=True, help="Path to transcript text file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Common output directory base
    # We will create data/ if it doesn't exist, but modules handle subdirs
    base_data_dir = os.path.join(os.getcwd(), "data")

    try:
        if args.command == "transcribe":
            print(f"Starting transcription for: {args.video}")
            transcript = process_video_pipeline(args.video, base_data_dir, model_name=args.model)
            print(f"Done. Transcript at: {transcript}")

        elif args.command == "summarize":
            print(f"Starting summarization for: {args.transcript}")
            summary_dir = os.path.join(base_data_dir, "summary")
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                print("Error: NVIDIA_API_KEY not set in .env or environment.")
                sys.exit(1)
            
            pdf = generate_summary_pipeline(args.transcript, summary_dir, api_key)
            print(f"Done. PDF at: {pdf}")

        elif args.command == "full-pipeline":
            print(f"Starting full pipeline for: {args.video}")
            
            # 1. Transcribe
            transcript = process_video_pipeline(args.video, base_data_dir, model_name=args.model)
            
            # 2. Summarize
            summary_dir = os.path.join(base_data_dir, "summary")
            api_key = os.getenv("NVIDIA_API_KEY")
            if not api_key:
                print("Error: NVIDIA_API_KEY not set in .env or environment. Transcription finished but summarization aborted.")
                print(f"Transcript available at: {transcript}")
                sys.exit(1)

            pdf = generate_summary_pipeline(transcript, summary_dir, api_key)
            print(f"Full pipeline complete. PDF at: {pdf}")

    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
