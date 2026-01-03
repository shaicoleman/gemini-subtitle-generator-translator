#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import pathlib
import argparse
import time
import queue
import shutil
import subprocess
from datetime import datetime

try:
    from split_audio import split_audio
    from transcript import run_transcription, DEFAULT_MAX_WORKERS
    from combine_transcripts import generate_srt
except ImportError as e:
    print(f"Error: Could not import script modules. Ensure split_audio.py, transcript.py, and combine_transcripts.py are in the same directory. Details: {e}")
    sys.exit(1)

def is_video_file(filepath):
    """Checks if a file is a video file based on its extension."""
    video_extensions = ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm', '.m4v']
    _, ext = os.path.splitext(filepath)
    return ext.lower() in video_extensions

def convert_video_to_mp3(video_path, output_dir=None, progress_queue=None, audio_stream=0):
    """Converts a video file to an MP3 audio file using ffmpeg."""
    if not os.path.isfile(video_path):
        error_msg = f"Error: Input video file '{video_path}' not found."
        if progress_queue: progress_queue.put(error_msg)
        print(error_msg)
        return None
    
    video_path_obj = pathlib.Path(video_path)
    
    if not output_dir:
        output_dir = video_path_obj.parent
    else:
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    mp3_filename = f"{video_path_obj.stem}.mp3"
    mp3_path = os.path.join(output_dir, mp3_filename)
    
    status_msg = f"Converting video to MP3: {video_path} -> {mp3_path}"
    if progress_queue: progress_queue.put(status_msg)
    print(status_msg)
    
    try:
        cmd = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", f"0:a:{audio_stream}", "-vn", mp3_path, "-y"]
        if sys.platform == 'win32':
             creation_flags = subprocess.CREATE_NO_WINDOW
        else:
             creation_flags = 0

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            creationflags=creation_flags
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            error_msg = f"Video conversion failed. FFmpeg error: {stderr}"
            if progress_queue: progress_queue.put(error_msg)
            print(error_msg)
            return None
        
        success_msg = f"Video successfully converted to MP3: {mp3_path}"
        if progress_queue: progress_queue.put(success_msg)
        print(success_msg)
        return mp3_path
    
    except Exception as e:
        error_msg = f"An error occurred during video conversion: {str(e)}"
        if progress_queue: progress_queue.put(error_msg)
        print(error_msg)
        return None

def run_pipeline(params, progress_queue=None, control_queue=None):
    """Runs the complete processing pipeline."""
    input_file = params.get('input_audio')
    output_dir = params.get('output_dir')
    api_key = params.get('api_key')
    content = params.get('content', 'both')
    first_chunk_offset = params.get('first_chunk_offset', 0.0)
    max_chunk_length = params.get('max_chunk_length', 300)
    silence_length = params.get('silence_length', 500)
    silence_threshold = params.get('silence_threshold', -40)
    cleanup = params.get('cleanup', False)
    target_language = params.get('target_language', 'English')
    model_name = params.get('model_name', 'gemini-3-flash-preview')
    skip_split = params.get('skip_split', False)
    audio_chunks_dir = params.get('audio_chunks_dir', None)
    max_workers = params.get('max_workers', DEFAULT_MAX_WORKERS)
    skip_existing = params.get('skip_existing', True)
    audio_stream = params.get('audio_stream', 0)

    if not input_file or not os.path.isfile(input_file):
        error_msg = f"Error: Input file '{input_file}' does not exist."
        if progress_queue: progress_queue.put(error_msg)
        print(error_msg)
        return False
    
    if not api_key:
        error_msg = "Error: API key not provided."
        if progress_queue: progress_queue.put(error_msg)
        print(error_msg)
        return False
    
    input_path = pathlib.Path(input_file)
    if not output_dir:
        output_dir = os.path.join(input_path.parent, input_path.stem)
    
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    input_audio = input_file
    total_start_time = time.time()
    
    # Video Conversion
    if is_video_file(input_file):
        start_msg = "\n--- Pre-processing: Video to MP3 ---"
        if progress_queue: progress_queue.put(start_msg)
        print(start_msg)
        
        convert_start = time.time()
        mp3_path = convert_video_to_mp3(input_file, output_dir, progress_queue, audio_stream)
        
        if not mp3_path or not os.path.isfile(mp3_path):
            error_msg = "Error: Video to MP3 conversion failed. Cannot continue."
            if progress_queue: progress_queue.put(error_msg)
            print(error_msg)
            return False
        
        input_audio = mp3_path
        convert_end = time.time()
        if progress_queue: progress_queue.put(f"Video to MP3 conversion complete ({convert_end - convert_start:.2f}s)")
    
    if skip_split and audio_chunks_dir and os.path.isdir(audio_chunks_dir):
        audio_chunk_dir = audio_chunks_dir
        if progress_queue: progress_queue.put(f"Skipping split, using: {audio_chunk_dir}")
    else:
        audio_chunk_dir = os.path.join(output_dir, "audio_chunks")

    intermediate_dir = os.path.join(output_dir, "intermediate_transcripts")
    srt_file = os.path.join(output_dir, f"{input_path.stem}.srt")
    
    # Step 1: Split Audio
    if not skip_split:
        start_msg = f"\n--- Step 1: Splitting Audio (Max: {max_chunk_length}s) ---"
        if progress_queue: progress_queue.put(start_msg)
        print(start_msg)
        
        step1_start = time.time()
        try:
            chunk_files = split_audio(
                input_audio, 
                audio_chunk_dir, 
                max_chunk_length=max_chunk_length * 1000,
                min_silence_len=silence_length,
                silence_thresh=silence_threshold,
                progress_queue=progress_queue
            )
            
            if not chunk_files:
                error_msg = "Error: Audio splitting failed."
                if progress_queue: progress_queue.put(error_msg)
                print(error_msg)
                return False
                
            step1_end = time.time()
            if progress_queue: progress_queue.put(f"Splitting complete ({step1_end - step1_start:.2f}s)")
        except Exception as e:
            error_msg = f"An error occurred during audio splitting: {e}"
            if progress_queue: progress_queue.put(error_msg)
            print(error_msg)
            return False
    
    # Step 2: Transcribe Audio
    start_msg = f"\n--- Step 2: Transcribing Audio (Model: {model_name}) ---"
    if progress_queue: progress_queue.put(start_msg)
    print(start_msg)
    
    step2_start = time.time()
    try:
        from transcript import get_system_instruction
        custom_system_instruction = get_system_instruction(target_language, content)
        
        transcription_success = run_transcription(
            api_key=api_key,
            audio_dir=audio_chunk_dir,
            intermediate_dir=intermediate_dir,
            system_instruction=custom_system_instruction,
            model_name=model_name,
            progress_queue=progress_queue,
            max_workers=max_workers,
            skip_existing=skip_existing
        )
        
        if not transcription_success:
            return False
            
        step2_end = time.time()
        if progress_queue: progress_queue.put(f"Transcription complete ({step2_end - step2_start:.2f}s)")
    except Exception as e:
        error_msg = f"An error occurred during transcription: {e}"
        if progress_queue: progress_queue.put(error_msg)
        print(error_msg)
        return False
    
    # Step 3: Combine Transcripts
    start_msg = f"\n--- Step 3: Combining to SRT ---"
    if progress_queue: progress_queue.put(start_msg)
    print(start_msg)
    
    step3_start = time.time()
    
    while True:
        try:
            srt_result = generate_srt(
                transcript_dir=intermediate_dir,
                audio_dir=audio_chunk_dir,
                output_srt_file=srt_file,
                content_choice=content,
                first_chunk_offset=first_chunk_offset,
                progress_queue=progress_queue
            )
            
            if srt_result is True:
                step3_end = time.time()
                if progress_queue: progress_queue.put(f"SRT generation complete ({step3_end - step3_start:.2f}s)")
                break
            
            elif srt_result == 'PARSE_ERROR':
                if control_queue:
                    if progress_queue: progress_queue.put("Waiting for user correction...")
                    try:
                        retry_signal = control_queue.get(block=True)
                        if retry_signal == 'RETRY_COMBINE':
                            continue
                        elif retry_signal == 'STOP_PROCESSING':
                            return False
                    except Exception as e:
                        return False
                else:
                    return False
            else:
                return False
                
        except Exception as e:
            if progress_queue: progress_queue.put(f"Error during SRT generation: {e}")
            return False
    
    # Cleanup
    if cleanup:
        if progress_queue: progress_queue.put("\n--- Cleaning up ---")
        try:
            if not (skip_split and audio_chunks_dir and audio_chunk_dir == audio_chunks_dir):
                shutil.rmtree(audio_chunk_dir)
            shutil.rmtree(intermediate_dir)
            if is_video_file(input_file) and input_audio != input_file:
                os.remove(input_audio)
        except Exception as e:
            print(f"Cleanup error: {e}")
    
    total_end_time = time.time()
    final_msg = f"\nProcessing complete! Total time: {total_end_time - total_start_time:.2f}s\nSRT file: {srt_file}"
    if progress_queue: progress_queue.put(final_msg)
    print(final_msg)
    
    return True

def main():
    parser = argparse.ArgumentParser(description="A one-stop tool for audio/video transcription and subtitle generation.")
    parser.add_argument("input_file", help="Path to the input audio or video file.")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Your Google AI API key. Defaults to GEMINI_API_KEY environment variable.")
    parser.add_argument("--output-dir", help="Specify the output directory.")
    parser.add_argument("--target-language", default="English", help="Target language for translation (default: English)")
    parser.add_argument("--content", choices=['transcript', 'translation', 'both'], default='both', help="Output content type (default: both)")
    parser.add_argument("--max-chunk-length", type=int, default=300, help="Maximum audio chunk length in seconds (default: 300)")
    parser.add_argument("--silence-length", type=int, default=500, help="Minimum silence length in ms for splitting (default: 500)")
    parser.add_argument("--silence-threshold", type=int, default=-40, help="Silence threshold in dB (default: -40)")
    parser.add_argument("--first-chunk-offset", type=float, default=0.0, help="Time offset for first chunk in seconds (default: 0.0)")
    parser.add_argument("--model-name", default="gemini-3-flash-preview", help="Gemini model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--skip-split", action="store_true", help="Skip audio splitting step")
    parser.add_argument("--audio-chunks-dir", help="Directory containing pre-split audio chunks")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS, help=f"Maximum parallel workers for API calls (default: {DEFAULT_MAX_WORKERS})")
    parser.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", help="Re-process chunks even if output exists")
    parser.add_argument("--cleanup", action="store_true", help="Remove intermediate files after processing")
    parser.add_argument("--audio-stream", type=int, default=0, help="Audio stream index to extract from video (default: 0)")

    args = parser.parse_args()
    params = vars(args)
    params['input_audio'] = params.pop('input_file')
    
    success = run_pipeline(params)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
