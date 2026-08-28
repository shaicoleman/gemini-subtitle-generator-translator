import os
import subprocess
import argparse
import sys
import time

def get_audio_duration_ffmpeg(input_file):
    """Gets the audio duration in seconds using ffprobe."""
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        stdout, stderr = process.communicate(timeout=60)
        if process.returncode != 0:
            print(f"  Error: ffprobe failed for {os.path.basename(input_file)}. Return code: {process.returncode}")
            return None
        duration_str = stdout.strip()
        if not duration_str:
            print(f"  Error: ffprobe returned empty output for {os.path.basename(input_file)}.")
            return None
        return float(duration_str)
    except FileNotFoundError:
        print("Error: ffprobe command not found. Ensure ffmpeg is installed and in your system's PATH.")
        return None
    except subprocess.TimeoutExpired:
        print(f"Error: ffprobe timed out for {os.path.basename(input_file)}")
        if process: process.kill()
        return None
    except ValueError:
        print(f"Error: Could not convert ffprobe output '{stdout.strip()}' to a float for {os.path.basename(input_file)}.")
        return None
    except Exception as e:
        print(f"Error: An unknown error occurred with ffprobe for {os.path.basename(input_file)}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="Compares the duration of an original audio file with the sum of its chunks.")
    parser.add_argument("--original-file", required=True, help="Path to the original audio file.")
    parser.add_argument("--chunk-dir", required=True, help="Directory containing the audio chunks.")
    parser.add_argument("--chunk-prefix", default="chunk_", help="Prefix for chunk filenames (default: chunk_).")
    parser.add_argument("--chunk-ext", default=".flac", help="Extension for chunk files (default: .flac).")
    args = parser.parse_args()

    print(f"Getting duration of original file: {args.original_file}")
    original_duration = get_audio_duration_ffmpeg(args.original_file)
    if original_duration is None:
        print("Could not get duration of the original file. Aborting.")
        sys.exit(1)
    print(f"Original file duration: {original_duration:.6f} seconds")

    print(f"\nScanning chunk directory: {args.chunk_dir}")
    total_chunk_duration = 0.0
    chunk_count = 0
    failed_chunks = 0

    try:
        files_in_dir = os.listdir(args.chunk_dir)
    except FileNotFoundError:
        print(f"Error: Chunk directory '{args.chunk_dir}' not found.")
        sys.exit(1)

    chunk_files = sorted([f for f in files_in_dir if f.startswith(args.chunk_prefix) and f.endswith(args.chunk_ext)])

    if not chunk_files:
        print("Error: No matching chunk files found in the specified directory.")
        sys.exit(1)

    print(f"Found {len(chunk_files)} chunk files. Calculating total duration...")

    start_time = time.time()
    for i, filename in enumerate(chunk_files):
        filepath = os.path.join(args.chunk_dir, filename)
        duration = get_audio_duration_ffmpeg(filepath)
        if duration is not None and duration > 0:
            total_chunk_duration += duration
            chunk_count += 1
        else:
            print(f"  Warning: Could not get a valid duration for chunk: {filename}. Skipping.")
            failed_chunks += 1

        if (i + 1) % 50 == 0 or (i + 1) == len(chunk_files):
            elapsed = time.time() - start_time
            print(f"  Processed {i + 1}/{len(chunk_files)} chunks... (Time elapsed: {elapsed:.2f}s)")

    print("\n--- Results ---")
    print(f"Original file ({os.path.basename(args.original_file)}) duration: {original_duration:.6f} seconds")
    print(f"Successfully processed chunks: {chunk_count}")
    if failed_chunks > 0:
        print(f"Chunks that failed duration check: {failed_chunks}")
    print(f"Total duration of all successful chunks: {total_chunk_duration:.6f} seconds")

    difference = total_chunk_duration - original_duration
    print(f"\nDifference (Total chunk duration - Original duration): {difference:+.6f} seconds")

    if abs(difference) < 0.1:
        print("Conclusion: The total duration of the chunks is consistent with the original file.")
    elif difference > 0:
        print("Warning: The total duration of the chunks is significantly greater than the original, which may indicate overlap or calculation errors.")
    else:
        print("Warning: The total duration of the chunks is significantly less than the original, which may lead to timing errors in the final subtitles.")

if __name__ == "__main__":
    main()
