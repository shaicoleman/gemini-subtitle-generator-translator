import os
import subprocess
import re
import pathlib
import math
import time
import argparse
import sys

# --- Configuration ---
OUTPUT_DIR = "audio_chunks"
MAX_CHUNK_LENGTH_SEC = 5 * 60  # 5 minutes
MIN_SILENCE_LENGTH_SEC = 0.5
SILENCE_THRESH_DB = -40
# ---------------------

def get_audio_duration_ffmpeg(input_file):
    """Gets the audio duration in seconds using ffprobe."""
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        if sys.platform == 'win32':
             creation_flags = subprocess.CREATE_NO_WINDOW
        else:
             creation_flags = 0

        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, creationflags=creation_flags)
        stdout, stderr = process.communicate(timeout=60)
        
        if process.returncode != 0:
            print(f"Error: ffprobe failed to get duration. Command: {' '.join(command)}\n{stderr}")
            return None
        return float(stdout.strip())
    except Exception as e:
        print(f"Error: An unknown error occurred with ffprobe for {input_file}: {e}")
        return None

def detect_silence_with_ffmpeg(input_file, min_silence_duration_sec, noise_tolerance_db, progress_queue=None):
    """Detects silence in an audio file using ffmpeg silencedetect."""
    msg = f"Detecting silence with ffmpeg (Threshold: {noise_tolerance_db}dB, Min Duration: {min_silence_duration_sec}s)..."
    if progress_queue: progress_queue.put(msg)
    print(msg)

    command = [
        'ffmpeg', '-i', input_file,
        '-af', f'silencedetect=noise={noise_tolerance_db}dB:d={min_silence_duration_sec}',
        '-f', 'null', '-'
    ]
    silence_points_sec = []
    try:
        if sys.platform == 'win32':
             creation_flags = subprocess.CREATE_NO_WINDOW
        else:
             creation_flags = 0
             
        process = subprocess.Popen(command, stderr=subprocess.PIPE, universal_newlines=True, creationflags=creation_flags)
        
        current_start = None
        for line in process.stderr:
            start_match = re.search(r'silence_start: (\d+\.?\d*)', line)
            if start_match:
                current_start = float(start_match.group(1))
            
            end_match = re.search(r'silence_end: (\d+\.?\d*)', line)
            if end_match and current_start is not None:
                current_end = float(end_match.group(1))
                if current_end > current_start:
                    silence_points_sec.append((current_start, current_end))
                current_start = None

        process.wait(timeout=300)
    except Exception as e:
        print(f"Error: An error occurred during silence detection with ffmpeg: {e}")
        return []

    if progress_queue: progress_queue.put(f"Detected {len(silence_points_sec)} silence periods.")
    return silence_points_sec

def find_optimal_split_points_sec(audio_length_sec, silence_points_sec, max_chunk_length_sec):
    """Calculates split points based on silence detection."""
    split_points = []
    current_chunk_start = 0.0

    for start_sec, end_sec in silence_points_sec:
        silence_midpoint = (start_sec + end_sec) / 2.0
        
        if silence_midpoint - current_chunk_start > max_chunk_length_sec:
            while (silence_midpoint - current_chunk_start) > max_chunk_length_sec:
                new_split = current_chunk_start + max_chunk_length_sec
                split_points.append(new_split)
                current_chunk_start = new_split
        
        split_points.append(silence_midpoint)
        current_chunk_start = silence_midpoint

    while (audio_length_sec - current_chunk_start) > max_chunk_length_sec:
        new_split = current_chunk_start + max_chunk_length_sec
        split_points.append(new_split)
        current_chunk_start = new_split

    final_split_points = sorted(list(set(p for p in split_points if 0 < p < audio_length_sec)))
    return final_split_points

def split_audio(input_file, output_dir, max_chunk_length=MAX_CHUNK_LENGTH_SEC * 1000,
                min_silence_len=int(MIN_SILENCE_LENGTH_SEC * 1000), silence_thresh=SILENCE_THRESH_DB, progress_queue=None):
    """
    Splits an audio file into chunks using ffmpeg. 
    automatically handles re-encoding if input is not mp3.
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    max_chunk_length_sec = max_chunk_length / 1000.0
    min_silence_len_sec = min_silence_len / 1000.0

    if progress_queue: progress_queue.put(f"Loading audio info: {input_file}")
    total_length_sec = get_audio_duration_ffmpeg(input_file)
    if total_length_sec is None:
        msg = f"Error: Could not get duration for {input_file}. Aborting split."
        if progress_queue: progress_queue.put(msg)
        print(msg)
        return []
    
    if progress_queue: progress_queue.put(f"Total audio duration: {total_length_sec:.2f} seconds")

    silence_points_sec = detect_silence_with_ffmpeg(input_file, min_silence_len_sec, silence_thresh, progress_queue)
    split_points_sec = find_optimal_split_points_sec(total_length_sec, silence_points_sec, max_chunk_length_sec)

    chunk_files = []
    start_time_sec = 0.0
    split_points_sec.append(total_length_sec)

    # Check input extension to decide encoding strategy
    # Gemini supports: WAV, MP3, AIFF, AAC, OGG, FLAC
    _, ext = os.path.splitext(input_file)
    ext_lower = ext.lower()

    # Map extensions to their output format and whether stream copy is possible
    supported_formats = {
        '.mp3': ('mp3', True),
        '.wav': ('wav', True),
        '.aiff': ('aiff', True),
        '.aif': ('aiff', True),
        '.aac': ('aac', True),
        '.m4a': ('m4a', True),  # AAC in M4A container
        '.ogg': ('ogg', True),
        '.flac': ('flac', True),
    }

    if ext_lower in supported_formats:
        output_ext, can_stream_copy = supported_formats[ext_lower]
    else:
        output_ext, can_stream_copy = 'mp3', False

    for i, end_time_sec in enumerate(split_points_sec):
        if end_time_sec <= start_time_sec + 0.1:
            continue

        chunk_filename = os.path.join(output_dir, f"chunk_{i+1:03d}.{output_ext}")
        duration_sec = end_time_sec - start_time_sec

        msg = f"Exporting chunk {i+1}/{len(split_points_sec)}: {start_time_sec:.2f}s - {end_time_sec:.2f}s -> {os.path.basename(chunk_filename)}"
        if progress_queue: progress_queue.put(msg)
        print(msg)

        # Use stream copy for supported formats, re-encode otherwise
        if can_stream_copy:
            codec_args = ['-c', 'copy']
        else:
            # Re-encode to standard MP3 (compatible with Gemini)
            # -vn (no video), -ar 44100 (sample rate), -ac 2 (stereo), -b:a 192k (bitrate)
            codec_args = ['-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k']

        command_split = [
            'ffmpeg', '-i', input_file, '-ss', str(start_time_sec), '-to', str(end_time_sec)
        ] + codec_args + ['-map_metadata', '-1', '-loglevel', 'error', '-y', chunk_filename]

        try:
            if sys.platform == 'win32':
                 creation_flags = subprocess.CREATE_NO_WINDOW
            else:
                 creation_flags = 0

            subprocess.run(command_split, check=True, capture_output=True, text=True, timeout=300, creationflags=creation_flags)
            
            # Verify file size > 0
            if os.path.getsize(chunk_filename) == 0:
                raise Exception("Generated file is 0 bytes.")
                
            chunk_files.append(chunk_filename)
        except subprocess.CalledProcessError as e:
            error_msg = f"  Error exporting {chunk_filename}: {e.stderr}"
            if progress_queue: progress_queue.put(error_msg)
            print(error_msg)
        except Exception as e:
            error_msg = f"  An unexpected error occurred while exporting {chunk_filename}: {e}"
            if progress_queue: progress_queue.put(error_msg)
            print(error_msg)

        start_time_sec = end_time_sec

    if not chunk_files:
        msg = "Error: No audio chunks were successfully exported."
        if progress_queue: progress_queue.put(msg)
        print(msg)
        return []

    success_msg = f"Splitting complete! {len(chunk_files)} chunks saved in {output_dir}"
    if progress_queue: progress_queue.put(success_msg)
    print(success_msg)
    return chunk_files

def main():
    parser = argparse.ArgumentParser(description="Splits a long audio file into smaller chunks using ffmpeg.")
    parser.add_argument("-i", "--input", required=True, help="Input audio file path.")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR}).")
    parser.add_argument("-m", "--max-length", type=int, default=MAX_CHUNK_LENGTH_SEC, help=f"Max chunk length in seconds (default: {MAX_CHUNK_LENGTH_SEC}).")
    parser.add_argument("-s", "--silence-length", type=int, default=int(MIN_SILENCE_LENGTH_SEC * 1000), help=f"Min silence length in milliseconds (default: {int(MIN_SILENCE_LENGTH_SEC * 1000)}).")
    parser.add_argument("-t", "--silence-threshold", type=int, default=SILENCE_THRESH_DB, help=f"Silence threshold in dB (default: {SILENCE_THRESH_DB}).")
    args = parser.parse_args()

    start_time = time.time()
    split_audio(args.input, args.output_dir,
                max_chunk_length=args.max_length * 1000,
                min_silence_len=args.silence_length,
                silence_thresh=args.silence_threshold)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()