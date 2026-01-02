import os
import re
import datetime
import pathlib
import sys
import subprocess

# Try to import Mutagen, but allow fallback if not installed
try:
    from mutagen.mp3 import MP3
    from mutagen.wave import WAVE
    HAS_MUTAGEN = True
except ImportError:
    HAS_MUTAGEN = False

# --- Configuration ---
# Rules for subtitle duration
MIN_DURATION_SEC = 1.5   
MAX_DURATION_SEC = 7.0   
CHARS_PER_SECOND = 14    
GAP_BETWEEN_SUBS = 0.05  
# ---------------------

SUPPORTED_AUDIO_EXTENSIONS = ('.mp3', '.wav', '.aiff', '.aif', '.aac', '.m4a', '.ogg', '.flac')

def get_audio_duration(filepath):
    """Robust duration checker."""
    if HAS_MUTAGEN:
        try:
            if filepath.lower().endswith('.mp3'):
                return MP3(filepath).info.length
            elif filepath.lower().endswith('.wav'):
                return WAVE(filepath).info.length
            # For other formats, fall through to ffprobe
        except Exception:
            pass 

    try:
        if sys.platform == 'win32':
             creation_flags = subprocess.CREATE_NO_WINDOW
        else:
             creation_flags = 0
             
        command = [
            'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   universal_newlines=True, creationflags=creation_flags)
        stdout, _ = process.communicate(timeout=10)
        return float(stdout.strip())
    except Exception as e:
        return 0.0

def format_timestamp_srt(seconds):
    if seconds < 0: seconds = 0
    td = datetime.timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int(td.microseconds / 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def parse_timestamp_sec(timestamp_str):
    clean_str = re.sub(r'[^\d:.]', '', timestamp_str)
    try:
        parts = clean_str.split(':')
        if len(parts) == 3: # HH:MM:SS
            h, m, s = parts
            return int(h) * 3600 + int(m) * 60 + float(s)
        elif len(parts) == 2: # MM:SS
            m, s = parts
            return int(m) * 60 + float(s)
    except ValueError:
        return None
    return None

def extract_segments(transcript_text, content_choice, filename=""):
    """
    STRICT Block-Based Parser.
    Extracts text ONLY from the requested section to prevent merging.
    """
    segments = []
    lines = transcript_text.splitlines()
    
    # 1. Normalize lines for reliable detection (remove Markdown formatting)
    # We keep the original lines for data extraction, but use clean lines for detection
    clean_lines_map = []
    for line in lines:
        # Remove bold, italics, headers markers, extra spaces, lower case for comparison
        clean = line.strip().lower().replace('*', '').replace('#', '').replace('_', '').replace(':', '')
        clean_lines_map.append(clean)

    # 2. Determine which section we are looking for
    # Priority list based on user choice
    priority_sections = []
    if content_choice == 'translation':
        priority_sections = ['timestamped translation']
    elif content_choice == 'transcript':
        priority_sections = ['timestamped transcript']
    else: # 'both' - Defaults to Translation, falls back to Transcript
        priority_sections = ['timestamped translation', 'timestamped transcript']

    found_start = -1
    found_end = len(lines)
    
    # 3. Find the Start of the Target Section
    # We iterate through priorities. If we find the first one, we stop.
    for target in priority_sections:
        for i, clean_line in enumerate(clean_lines_map):
            # Check if line contains the target header (e.g. "timestamped translation")
            # We use 'in' to handle cases like "## Timestamped Translation:"
            if target in clean_line and len(clean_line) < 50:
                found_start = i
                break
        if found_start != -1:
            break
    
    # If we didn't find the requested section, try the fallback (if content_choice was strictly one, we might return empty)
    if found_start == -1:
        # If user wanted Translation but it's missing, try Transcript as a last resort?
        # Better to return empty so we don't confuse the user, unless it's 'both'
        if content_choice == 'both':
             # Try finding transcript if translation failed
             for i, clean_line in enumerate(clean_lines_map):
                if 'timestamped transcript' in clean_line and len(clean_line) < 50:
                    found_start = i
                    break
    
    if found_start == -1:
        print(f"Warning: Could not find '{content_choice}' section in {filename}")
        return []

    # 4. Find the End of the Section (The next header)
    # Start scanning from the line AFTER the header
    for i in range(found_start + 1, len(lines)):
        clean_line = clean_lines_map[i]
        # If we hit a line that looks like ANY header (Transcript, Translation, etc), stop.
        if ('transcript' in clean_line or 'translation' in clean_line) and len(clean_line) < 50:
            # Make sure it's not just a sentence containing the word "transcript"
            # Headers usually don't have timestamps or long text
            if 'timestamped' in clean_line or clean_line == 'transcript' or clean_line == 'translation':
                found_end = i
                break

    # 5. Extract Timestamps from the identified block ONLY
    target_block = lines[found_start+1 : found_end]
    
    for line in target_block:
        line_clean = line.strip().replace('**', '').replace('__', '')
        # Matches [00:00], (00:00), 00:00.000 at start of line
        match = re.search(r'^[\[\(]?\s*(\d{1,2}:\d{2}(?:[:.]\d{1,3})?)[\]\)]?:?\s*(.*)', line_clean)
        
        if match:
            ts_str = match.group(1)
            text = match.group(2).strip()
            
            # Avoid capturing empty lines or lines that are just timestamps
            if text: 
                sec = parse_timestamp_sec(ts_str)
                if sec is not None:
                    segments.append((sec, text))

    return segments

def generate_srt(transcript_dir, audio_dir, output_srt_file, content_choice='both', first_chunk_offset=0.0, progress_queue=None):
    if progress_queue: progress_queue.put(f"Generating SRT from {audio_dir}...")

    # 1. Get Audio Files
    try:
        audio_files = sorted(
            [f for f in os.listdir(audio_dir) if f.lower().endswith(SUPPORTED_AUDIO_EXTENSIONS)],
            key=lambda x: int(re.findall(r'\d+', x)[-1]) if re.findall(r'\d+', x) else 0
        )
    except Exception as e:
        if progress_queue: progress_queue.put(f"Error listing audio files: {e}")
        return False

    if not audio_files:
        if progress_queue: progress_queue.put("No audio chunks found.")
        return False

    global_offset = first_chunk_offset
    all_srt_entries = []
    
    # 2. Iterate
    for i, audio_filename in enumerate(audio_files):
        if progress_queue and i % 5 == 0:
            progress_queue.put(f"Merging chunk {i+1}/{len(audio_files)}...")
            
        audio_path = os.path.join(audio_dir, audio_filename)
        duration = get_audio_duration(audio_path)
        
        transcript_filename = pathlib.Path(audio_filename).stem + ".txt"
        transcript_path = os.path.join(transcript_dir, transcript_filename)
        
        if os.path.exists(transcript_path):
            try:
                with open(transcript_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text_content = f.read()
                
                # Pass user choice to extract ONLY that section
                local_segments = extract_segments(text_content, content_choice, filename=transcript_filename)
                
                for j, (start_sec, text) in enumerate(local_segments):
                    global_start = global_offset + start_sec
                    
                    # Duration Logic
                    ideal_duration = (len(text) / CHARS_PER_SECOND) + 1.0
                    ideal_duration = max(MIN_DURATION_SEC, min(ideal_duration, MAX_DURATION_SEC))
                    ideal_end = global_start + ideal_duration
                    
                    # Constraint Logic
                    constraint_time = None
                    if j < len(local_segments) - 1:
                        next_sec = local_segments[j+1][0]
                        constraint_time = (global_offset + next_sec) - GAP_BETWEEN_SUBS
                    else:
                        constraint_time = global_offset + duration

                    if ideal_end < constraint_time:
                        global_end = ideal_end
                    else:
                        global_end = constraint_time

                    if global_end <= global_start:
                        global_end = global_start + MIN_DURATION_SEC

                    all_srt_entries.append({'start': global_start, 'end': global_end, 'text': text})
                    
            except Exception as e:
                print(f"Error parsing {transcript_filename}: {e}")
        
        global_offset += duration

    # 3. Write
    if not all_srt_entries:
        if progress_queue: progress_queue.put("Error: No subtitles could be extracted. Please check the 'intermediate_transcripts' folder to ensure the AI generated valid text.")
        return False 

    try:
        with open(output_srt_file, 'w', encoding='utf-8') as f:
            for idx, entry in enumerate(all_srt_entries):
                f.write(f"{idx + 1}\n")
                f.write(f"{format_timestamp_srt(entry['start'])} --> {format_timestamp_srt(entry['end'])}\n")
                f.write(f"{entry['text']}\n\n")
        
        success_msg = f"SRT generated successfully: {output_srt_file}"
        if progress_queue: progress_queue.put(success_msg)
        print(success_msg)
        return True
    except Exception as e:
        if progress_queue: progress_queue.put(f"Error writing SRT: {e}")
        return False