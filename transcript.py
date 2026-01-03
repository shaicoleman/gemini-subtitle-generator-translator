import os
import re
from google import genai
from google.genai import types
import time
import pathlib
import random
import concurrent.futures
import threading

# --- Configuration ---
API_KEY = "YOUR_API_KEY_HERE" # Will be overwritten by the GUI/Main script
AUDIO_DIR = "audio_chunks"
INTERMEDIATE_DIR = "intermediate_transcripts"
MAX_RETRIES = 5
INITIAL_DELAY = 2
DEFAULT_MODEL = "gemini-3-flash-preview"
DEFAULT_MAX_WORKERS = 5
SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.aiff', '.aif', '.aac', '.m4a', '.ogg', '.flac')
# ---------------------

def get_system_instruction(target_language="English"):
    return f"""You are an expert transcription engine powered by Gemini 3.0.
Task:
1.  **Transcribe** the audio verbatim in its original language.
2.  **Translate** the transcript into {target_language} (if the audio is not already in that language).
3.  **Timestamp** every single sentence precisely.

**CRITICAL FORMATTING RULES:**
* You MUST provide timestamps at the start of every segment in the format **[MM:SS.ms]** (Minutes:Seconds.Milliseconds).
* Example: [00:05.123] This is the first sentence.
* Keep each timestamped line under 80 characters. If a sentence is longer, split it at natural pauses (commas, conjunctions, clauses) into multiple timestamped lines.
* Do NOT use (MM:SS) or [MM:SS]. You MUST include milliseconds.
* Output exactly four sections separated by blank lines.
* Output the transcript and translation as is.
* Do not include filler pauses (umm, uhh, err, ahh).
* Extract the text/subtitle that appear on the video as it is.
* Do not censor explicit language.

**Strict Output Template:**
Transcript:
[Full transcript text without timestamps]

Translation:
[Full translation text without timestamps]

Timestamped Transcript:
[00:00.000] First sentence of the transcript.
[00:05.500] Second sentence.

Timestamped Translation:
[00:00.000] First sentence of the translation.
[00:05.500] Second sentence of the translation.
"""

def initialize_genai_client(api_key):
    """Initializes the GenAI client."""
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        print(f"Error initializing GenAI client: {e}")
        return None

def process_audio_file(filepath, intermediate_dir, system_instruction, model_name, client):
    """Processes a single audio file: uploads, transcribes, and saves the result."""
    filename = os.path.basename(filepath)
    transcript_filename = pathlib.Path(filename).stem + ".txt"
    intermediate_filepath = os.path.join(intermediate_dir, transcript_filename)

    for attempt in range(MAX_RETRIES):
        try:
            print(f"  Uploading {filename} (Attempt {attempt + 1}/{MAX_RETRIES})...")
            # Upload file using the new client API
            audio_file = client.files.upload(file=filepath)

            # Wait for processing to complete (important for large files)
            while audio_file.state.name == "PROCESSING":
                time.sleep(1)
                audio_file = client.files.get(name=audio_file.name)

            if audio_file.state.name == "FAILED":
                raise ValueError("Audio file upload failed processing.")

            print(f"  Transcribing {filename} with model {model_name}...")

            # Low temperature for deterministic timestamps
            generation_config = types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.2
            )

            response = client.models.generate_content(
                model=model_name,
                contents=["Please process this audio file strictly according to the system instructions.", audio_file],
                config=generation_config
            )

            # Handle potential "thinking" blocks in Gemini 3.0 if present, though .text usually extracts the final answer
            transcript = response.text

            with open(intermediate_filepath, "w", encoding="utf-8") as f:
                f.write(transcript)

            print(f"  Successfully processed and saved: {intermediate_filepath}")
            client.files.delete(name=audio_file.name)
            return transcript

        except Exception as e:
            print(f"  Error processing {filename} (Attempt {attempt + 1}/{MAX_RETRIES}): {e}")
            if attempt < MAX_RETRIES - 1:
                delay = (INITIAL_DELAY * (2 ** attempt)) + random.uniform(0, 1)
                print(f"  Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                error_message = f"Error processing {filename} after {MAX_RETRIES} attempts: {e}\n"
                with open(intermediate_filepath, "w", encoding="utf-8") as f:
                    f.write(error_message)
                print(f"  Failed to process {filename}. Error saved to {intermediate_filepath}")
                return ""
    return ""

def run_transcription(api_key, audio_dir, intermediate_dir, system_instruction, model_name=DEFAULT_MODEL, progress_queue=None, max_workers=DEFAULT_MAX_WORKERS, skip_existing=True):
    """Transcribes all audio files in a directory in parallel."""
    client = initialize_genai_client(api_key)
    if not client:
        if progress_queue: progress_queue.put("Failed to initialize GenAI client. Check API key.")
        return False

    pathlib.Path(intermediate_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        audio_files = sorted(
            [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)],
            key=lambda x: int(re.findall(r'\d+', pathlib.Path(x).stem)[-1]) if re.findall(r'\d+', pathlib.Path(x).stem) else 0
        )
    except FileNotFoundError:
        if progress_queue: progress_queue.put(f"Error: Audio directory not found at {audio_dir}")
        return False

    if not audio_files:
        if progress_queue: progress_queue.put(f"No audio files found in {audio_dir}")
        return False

    processed_count = 0
    success_count = 0
    skipped_count = 0
    count_lock = threading.Lock()

    def update_progress(message):
        if progress_queue: progress_queue.put(message)
        print(message)

    def is_valid_transcript(filepath):
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            return False
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().lower()
            return "error" not in content and "timestamped transcript:" in content

    def process_file_wrapper(filepath):
        nonlocal processed_count, success_count, skipped_count
        filename = os.path.basename(filepath)
        intermediate_filepath = os.path.join(intermediate_dir, f"{pathlib.Path(filename).stem}.txt")

        if skip_existing and is_valid_transcript(intermediate_filepath):
            with count_lock:
                processed_count += 1
                skipped_count += 1
                success_count += 1
            update_progress(f"({processed_count}/{len(audio_files)}) Skipping existing valid transcript: {filename}")
            return "SKIPPED"

        result = process_audio_file(filepath, intermediate_dir, system_instruction, model_name, client)
        
        with count_lock:
            processed_count += 1
            if result and "error" not in result.lower():
                success_count += 1
                status = "Success"
            else:
                status = "Failed"
        update_progress(f"({processed_count}/{len(audio_files)}) {status}: {filename}")
        return result

    actual_workers = min(max_workers, len(audio_files))
    update_progress(f"Starting transcription of {len(audio_files)} files with {actual_workers} parallel workers using {model_name}...")

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=actual_workers) as executor:
        executor.map(process_file_wrapper, audio_files)

    total_time = time.time() - start_time
    summary_msg = (f"\nTranscription complete. "
                   f"Success: {success_count}/{len(audio_files)} "
                   f"(Skipped: {skipped_count}, Newly Processed: {success_count - skipped_count}). "
                   f"Total time: {total_time:.2f} seconds.")
    update_progress(summary_msg)
    
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Transcribe audio files.")
    parser.add_argument("--api-key", default=os.environ.get("GEMINI_API_KEY"), help="Google AI API key. Defaults to GEMINI_API_KEY environment variable.")
    parser.add_argument("--audio-dir", default=AUDIO_DIR)
    parser.add_argument("--intermediate-dir", default=INTERMEDIATE_DIR)
    parser.add_argument("--target-language", default="Simplified Chinese")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    args = parser.parse_args()

    system_instruction = get_system_instruction(args.target_language)
    run_transcription(args.api_key, args.audio_dir, args.intermediate_dir, system_instruction, args.model_name, max_workers=args.max_workers)
