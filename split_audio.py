import os
import subprocess
import re
import pathlib
import math
import time
import argparse
import json
import sys
import concurrent.futures
import threading

# --- Configuration ---
OUTPUT_DIR = "audio_chunks"
MAX_CHUNK_LENGTH = 5 * 60  # 5 minutes
MIN_SILENCE_LENGTH = 0.5
SILENCE_THRESH_DB = -40
SKIP_SILENCE_LENGTH = 5.0
DEFAULT_MAX_WORKERS = 8
METADATA_FILENAME = "chunks_metadata.json"
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

def detect_silence_with_ffmpeg(input_file, min_silence_duration, noise_tolerance_db, progress_queue=None):
    """Detects silence in an audio file using ffmpeg silencedetect."""
    msg = f"Detecting silence with ffmpeg (Threshold: {noise_tolerance_db}dB, Min Duration: {min_silence_duration}s)..."
    if progress_queue: progress_queue.put(msg)
    print(msg)

    command = [
        'ffmpeg', '-i', input_file,
        '-af', f'silencedetect=noise={noise_tolerance_db}dB:d={min_silence_duration}',
        '-f', 'null', '-'
    ]
    silence_points = []
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
                    silence_points.append((current_start, current_end))
                current_start = None

        process.wait(timeout=300)
    except Exception as e:
        print(f"Error: An error occurred during silence detection with ffmpeg: {e}")
        return []

    if progress_queue: progress_queue.put(f"Detected {len(silence_points)} silence periods.")
    return silence_points

def find_chunk_ranges(audio_length, silence_points, max_chunk_length,
                      skip_silence_length=SKIP_SILENCE_LENGTH):
    """Returns list of (source_start, source_end) pairs for audio chunks.

    Silences with duration >= skip_silence_length are excluded from chunks
    entirely (not sent to the transcriber). Within each active (non-silent) span,
    chunks are kept under max_chunk_length, preferring to split at shorter
    silence midpoints when one falls within the window.
    """
    long_silences = sorted(
        (s, e) for (s, e) in silence_points if (e - s) >= skip_silence_length
    )
    short_silence_midpoints = sorted(
        (s + e) / 2.0 for (s, e) in silence_points if (e - s) < skip_silence_length
    )

    active_intervals = []
    cursor = 0.0
    for (s, e) in long_silences:
        if s > cursor:
            active_intervals.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < audio_length:
        active_intervals.append((cursor, audio_length))

    chunks = []
    for (iv_start, iv_end) in active_intervals:
        start = iv_start
        while iv_end - start > max_chunk_length:
            limit = start + max_chunk_length
            candidates = [m for m in short_silence_midpoints if start < m <= limit]
            split = candidates[-1] if candidates else limit
            chunks.append((start, split))
            start = split
        if iv_end - start > 0.1:
            chunks.append((start, iv_end))

    return chunks

def split_audio(input_file, output_dir, max_chunk_length=MAX_CHUNK_LENGTH,
                min_silence_len=MIN_SILENCE_LENGTH, silence_thresh=SILENCE_THRESH_DB,
                skip_silence_length=SKIP_SILENCE_LENGTH, max_workers=DEFAULT_MAX_WORKERS,
                progress_queue=None):
    """
    Splits an audio file into chunks using ffmpeg.
    automatically handles re-encoding if input is not mp3.
    """
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    if progress_queue: progress_queue.put(f"Loading audio info: {input_file}")
    total_length = get_audio_duration_ffmpeg(input_file)
    if total_length is None:
        msg = f"Error: Could not get duration for {input_file}. Aborting split."
        if progress_queue: progress_queue.put(msg)
        print(msg)
        return []

    if progress_queue: progress_queue.put(f"Total audio duration: {total_length:.2f} seconds")

    silence_points = detect_silence_with_ffmpeg(input_file, min_silence_len, silence_thresh, progress_queue)
    chunk_ranges = find_chunk_ranges(total_length, silence_points,
                                      max_chunk_length, skip_silence_length)
    skipped_total = total_length - sum(e - s for s, e in chunk_ranges)
    if progress_queue and skipped_total > 0:
        progress_queue.put(
            f"Skipping {skipped_total:.1f}s of silence (>= {skip_silence_length}s gaps)"
        )

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

    # Use stream copy for supported formats, re-encode otherwise
    if can_stream_copy:
        codec_args = ['-c', 'copy']
    else:
        # Re-encode to standard MP3 (compatible with Gemini)
        # -vn (no video), -ar 44100 (sample rate), -ac 2 (stereo), -b:a 192k (bitrate)
        codec_args = ['-vn', '-ar', '44100', '-ac', '2', '-b:a', '192k']

    if sys.platform == 'win32':
        creation_flags = subprocess.CREATE_NO_WINDOW
    else:
        creation_flags = 0

    results = [None] * len(chunk_ranges)
    completed = [0]
    completed_lock = threading.Lock()

    def export_chunk(i, start_time, end_time):
        chunk_filename = os.path.join(output_dir, f"chunk_{i+1:03d}.{output_ext}")
        command_split = [
            'ffmpeg', '-i', input_file, '-ss', str(start_time), '-to', str(end_time)
        ] + codec_args + ['-map_metadata', '-1', '-loglevel', 'error', '-y', chunk_filename]
        try:
            subprocess.run(command_split, check=True, capture_output=True, text=True,
                           timeout=300, creationflags=creation_flags)
            if os.path.getsize(chunk_filename) == 0:
                raise Exception("Generated file is 0 bytes.")
        except subprocess.CalledProcessError as e:
            return None, f"  Error exporting {chunk_filename}: {e.stderr}"
        except Exception as e:
            return None, f"  An unexpected error occurred while exporting {chunk_filename}: {e}"
        return chunk_filename, None

    actual_workers = max(1, min(max_workers, len(chunk_ranges)))
    total = len(chunk_ranges)
    if progress_queue:
        progress_queue.put(f"Exporting {total} chunks with {actual_workers} workers...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        future_to_index = {
            executor.submit(export_chunk, i, s, e): i
            for i, (s, e) in enumerate(chunk_ranges)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            i = future_to_index[future]
            chunk_filename, error = future.result()
            with completed_lock:
                completed[0] += 1
                done = completed[0]
            if error:
                if progress_queue: progress_queue.put(error)
                print(error)
                continue
            start_time, end_time = chunk_ranges[i]
            results[i] = (chunk_filename, {
                "file": os.path.basename(chunk_filename),
                "source_start": start_time,
                "source_end": end_time,
            })
            msg = f"Exported {done}/{total}: {os.path.basename(chunk_filename)} ({start_time:.2f}s - {end_time:.2f}s)"
            if progress_queue: progress_queue.put(msg)
            print(msg)

    chunk_files = [r[0] for r in results if r is not None]
    chunk_metadata = [r[1] for r in results if r is not None]

    if not chunk_files:
        msg = "Error: No audio chunks were successfully exported."
        if progress_queue: progress_queue.put(msg)
        print(msg)
        return []

    metadata_path = os.path.join(output_dir, METADATA_FILENAME)
    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump({
                "version": 1,
                "source_file": os.path.basename(input_file),
                "source_duration": total_length,
                "skip_silence_length": skip_silence_length,
                "chunks": chunk_metadata,
            }, f, indent=2)
    except Exception as e:
        warn_msg = f"  Warning: Failed to write chunk metadata: {e}"
        if progress_queue: progress_queue.put(warn_msg)
        print(warn_msg)

    success_msg = f"Splitting complete! {len(chunk_files)} chunks saved in {output_dir}"
    if progress_queue: progress_queue.put(success_msg)
    print(success_msg)
    return chunk_files

def main():
    parser = argparse.ArgumentParser(description="Splits a long audio file into smaller chunks using ffmpeg.")
    parser.add_argument("-i", "--input", required=True, help="Input audio file path.")
    parser.add_argument("-o", "--output-dir", default=OUTPUT_DIR, help=f"Output directory (default: {OUTPUT_DIR}).")
    parser.add_argument("-m", "--max-chunk-length", type=int, default=MAX_CHUNK_LENGTH, help=f"Max chunk length in seconds (default: {MAX_CHUNK_LENGTH}).")
    parser.add_argument("-s", "--silence-length", type=float, default=MIN_SILENCE_LENGTH, help=f"Min silence length in seconds (default: {MIN_SILENCE_LENGTH}).")
    parser.add_argument("-t", "--silence-threshold", type=int, default=SILENCE_THRESH_DB, help=f"Silence threshold in dB (default: {SILENCE_THRESH_DB}).")
    parser.add_argument("--skip-silence-length", type=float, default=SKIP_SILENCE_LENGTH,
                        help=f"Skip silences longer than this many seconds (default: {SKIP_SILENCE_LENGTH}).")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS,
                        help=f"Parallel ffmpeg workers for chunk export (default: {DEFAULT_MAX_WORKERS}).")
    args = parser.parse_args()

    start_time = time.time()
    split_audio(args.input, args.output_dir,
                max_chunk_length=args.max_chunk_length,
                min_silence_len=args.silence_length,
                silence_thresh=args.silence_threshold,
                skip_silence_length=args.skip_silence_length,
                max_workers=args.max_workers)
    end_time = time.time()
    print(f"Total processing time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()