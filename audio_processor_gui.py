#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import pathlib
import time
import multiprocessing
import psutil
import subprocess
from datetime import datetime

# Import processing functions and video detection from the main script
try:
    from process_audio import run_pipeline, is_video_file, DEFAULT_MAX_WORKERS
except ImportError as e:
    print(f"Error: Unable to import the process_audio.py module. Ensure the file is in the same directory. Details: {e}")
    sys.exit(1)

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)

# Define the translation dictionary
translations = {
    "en_US": {
        "title": "Audio/Video Transcription (Gemini 3.0)",
        "basic_settings": "Basic Settings",
        "input_file": "Input File (Audio/Video):",
        "browse": "Browse...",
        "api_key": "Google AI API Key:",
        "show_hide": "Show/Hide",
        "output_dir": "Output Directory (Optional):",
        "process_params": "Processing Parameters",
        "content_type": "Subtitle Content:",
        "content_desc": "(transcript: Transcription only, translation: Translation only, both: Both)",
        "target_language": "Target Translation Language:",
        "max_length": "Max Segment Length (sec):",
        "silence_length": "Silence Detection Length (ms):",
        "silence_threshold": "Silence Threshold (dB):",
        "first_chunk_offset": "First Segment Offset (sec):",
        "cleanup": "Delete intermediate files after processing",
        "start_process": "Start Processing",
        "stop_process": "Stop Processing",
        "progress": "Progress",
        "ready": "Ready",
        "processing": "Processing...",
        "stopped": "Stopped",
        "file_not_selected": "No file selected",
        "video_file": "Video file (will be converted to MP3)",
        "audio_file": "Audio file",
        "error_no_input": "Error: Please select an input file (audio or video)",
        "error_no_api_key": "Error: Please enter a Google AI API Key",
        "confirm_start": "Confirm",
        "confirm_start_message": "Are you sure you want to start processing?\nThis may take some time depending on the file length.",
        "confirm_stop": "Confirm",
        "confirm_stop_message": "Are you sure you want to stop processing?\nCurrent progress will be lost.",
        "confirm_close": "Confirm",
        "confirm_close_message": "Processing is in progress. Are you sure you want to exit?",
        "complete": "Complete",
        "complete_message": "Processing completed!\nOutput directory: {output_dir}",
        "error": "Error",
        "process_failed": "Processing failed. Please check the detailed log.",
        "unexpected_error": "An error occurred during processing:\n{error}",
        "language": "Language",
        "select_file": "Select Audio or Video File",
        "select_output_dir": "Select Output Directory",
        "user_stop": "User manually stopped processing.",
        "model": "Model:",
        "retry_combine": "Retry Combine",
        "open_error_file": "Open Error File",
        "timestamp_error": "Timestamp Parse Error",
        "timestamp_error_message": "Timestamp parsing errors detected. Please edit the error files and click the 'Retry Combine' button.",
        "select_error_file": "Please select an error file to open",
        "error_file_not_found": "Error file not found or has been moved",
        "waiting_for_fix": "Waiting for fix...",
        "parallel_requests": "Parallel Requests:",
        "resume_from_breakpoint": "Resume from breakpoint (Skip existing transcription files)",
        "skip_audio_segmentation": "Skip audio segmentation",
        "audio_chunks_dir_label": "Audio Chunks Directory:",
        "use_default_dir": "Use Default Directory",
        "error_skip_segmentation_no_dir": "Error: Skip audio segmentation is selected, but no audio chunks directory is specified.",
        "error_chunks_dir_not_found": "Error: The specified audio chunks directory '{dir}' does not exist.",
        "error_select_input_first": "Please select an input file first to use the default audio chunks directory.",
        "dir_not_exist_prompt": "The default audio chunks directory '{dir}' does not exist.\nDo you want to use this path anyway?"
    },
    "zh_CN": {
        "title": "音频/视频转录与字幕生成工具 (Gemini 3.0)",
        "basic_settings": "基本设置",
        "input_file": "输入文件(音频/视频):",
        "browse": "浏览...",
        "api_key": "Google AI API密钥:",
        "show_hide": "显示/隐藏",
        "output_dir": "输出目录 (可选):",
        "process_params": "处理参数",
        "content_type": "字幕内容:",
        "content_desc": "(transcript:仅转录, translation:仅翻译, both:两者)",
        "target_language": "翻译目标语言:",
        "max_length": "最大片段长度(秒):",
        "silence_length": "静音检测长度(毫秒):",
        "silence_threshold": "静音阈值(dB):",
        "first_chunk_offset": "首个片段偏移(秒):",
        "cleanup": "处理完成后删除中间文件",
        "start_process": "开始处理",
        "stop_process": "停止处理",
        "progress": "处理进度",
        "ready": "就绪",
        "processing": "处理中...",
        "stopped": "已停止",
        "file_not_selected": "未选择文件",
        "video_file": "视频文件 (将自动转换为MP3)",
        "audio_file": "音频文件",
        "error_no_input": "错误: 请选择输入文件(音频或视频)",
        "error_no_api_key": "错误: 请输入Google AI API密钥",
        "confirm_start": "确认",
        "confirm_start_message": "确定要开始处理吗？\n这可能需要一段时间，具体取决于文件长度。",
        "confirm_stop": "确认",
        "confirm_stop_message": "确定要停止处理吗？\n当前进度将丢失。",
        "confirm_close": "确认",
        "confirm_close_message": "处理正在进行中。确定要退出吗？",
        "complete": "完成",
        "complete_message": "处理已完成！\n输出目录: {output_dir}",
        "error": "错误",
        "process_failed": "处理失败。请查看详细日志。",
        "unexpected_error": "处理过程中发生错误:\n{error}",
        "language": "语言",
        "select_file": "选择音频或视频文件",
        "select_output_dir": "选择输出目录",
        "user_stop": "用户手动停止处理。",
        "model": "模型:",
        "retry_combine": "重试合并",
        "open_error_file": "打开错误文件",
        "timestamp_error": "时间戳解析错误",
        "timestamp_error_message": "检测到时间戳解析错误。请修改出错的文件后点击“重试合并”按钮。",
        "select_error_file": "请选择要打开的错误文件",
        "error_file_not_found": "错误文件未找到或已被移动",
        "waiting_for_fix": "等待修复...",
        "parallel_requests": "并行请求数:",
        "resume_from_breakpoint": "断点续传 (跳过已存在的转录文件)",
        "skip_audio_segmentation": "跳过音频切分",
        "audio_chunks_dir_label": "音频切片目录:",
        "use_default_dir": "使用默认目录",
        "error_skip_segmentation_no_dir": "错误: 已选择跳过切分音频，但未指定音频切片目录。",
        "error_chunks_dir_not_found": "错误: 指定的音频切片目录 '{dir}' 不存在。",
        "error_select_input_first": "请先选择输入文件，才能使用默认音频切片目录。",
        "dir_not_exist_prompt": "默认音频切片目录 '{dir}' 不存在。\n是否仍要使用此路径？"
    }
}

class AudioProcessorGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.current_language = tk.StringVar(value="en_US")
        
        self.title(translations[self.current_language.get()]["title"])
        self.geometry("800x700")
        self.minsize(700, 600)
        
        self.input_file_path = tk.StringVar()
        self.output_dir_path = tk.StringVar()
        self.api_key = tk.StringVar()
        self.content_choice = tk.StringVar(value="both")
        self.target_language = tk.StringVar(value="Simplified Chinese")
        self.max_length = tk.IntVar(value=300)
        self.silence_length = tk.IntVar(value=500)
        self.silence_threshold = tk.IntVar(value=-40)
        self.first_chunk_offset = tk.DoubleVar(value=0.0)
        self.cleanup = tk.BooleanVar(value=False)
        self.model_name = tk.StringVar(value="gemini-3-flash-preview")
        self.skip_split = tk.BooleanVar(value=False)
        self.audio_chunks_dir = tk.StringVar()
        self.max_workers = tk.IntVar(value=DEFAULT_MAX_WORKERS)
        self.skip_existing = tk.BooleanVar(value=True)

        self.processing = False
        self.process_thread = None
        self.process = None
        self.process_pid = None
        self.waiting_for_user_fix = False
        self.error_files = []
        
        self.progress_queue = multiprocessing.Queue()
        self.control_queue = multiprocessing.Queue()
        
        self.ui_elements = {}
        
        self.create_widgets()
        self.change_language()
        self.check_queue()
        
        self.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def create_widgets(self):
        """Create all GUI components."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Language selection
        lang_frame = ttk.Frame(main_frame)
        lang_frame.pack(fill=tk.X, pady=5)
        
        self.ui_elements["language_label"] = ttk.Label(lang_frame)
        self.ui_elements["language_label"].pack(side=tk.LEFT, padx=5)
        lang_combo = ttk.Combobox(lang_frame, textvariable=self.current_language, width=10)
        lang_combo['values'] = ('en_US', 'zh_CN')
        lang_combo.current(0)
        lang_combo.pack(side=tk.LEFT, padx=5)
        lang_combo.bind("<<ComboboxSelected>>", self.change_language)
        
        # Input frame
        input_frame = ttk.LabelFrame(main_frame, padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        self.ui_elements["basic_settings_frame"] = input_frame
        
        # Input file
        self.ui_elements["input_file_label"] = ttk.Label(input_frame)
        self.ui_elements["input_file_label"].grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.input_file_path, width=50).grid(row=0, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.ui_elements["browse_input_btn"] = ttk.Button(input_frame, command=self.browse_input_file)
        self.ui_elements["browse_input_btn"].grid(row=0, column=2, sticky=tk.W, pady=5)
        
        self.file_type_var = tk.StringVar()
        self.ui_elements["file_type_label"] = ttk.Label(input_frame, textvariable=self.file_type_var, foreground="blue")
        self.ui_elements["file_type_label"].grid(row=0, column=3, sticky=tk.W, pady=5, padx=5)
        
        # API Key
        self.ui_elements["api_key_label"] = ttk.Label(input_frame)
        self.ui_elements["api_key_label"].grid(row=1, column=0, sticky=tk.W, pady=5)
        api_key_entry = ttk.Entry(input_frame, textvariable=self.api_key, width=50, show="*")
        api_key_entry.grid(row=1, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.ui_elements["show_hide_btn"] = ttk.Button(input_frame, command=lambda: self.toggle_api_key_visibility(api_key_entry))
        self.ui_elements["show_hide_btn"].grid(row=1, column=2, sticky=tk.W, pady=5)
        
        env_api_key = os.environ.get("GOOGLE_API_KEY")
        if env_api_key:
            self.api_key.set(env_api_key)
        
        # Output directory
        self.ui_elements["output_dir_label"] = ttk.Label(input_frame)
        self.ui_elements["output_dir_label"].grid(row=2, column=0, sticky=tk.W, pady=5)
        ttk.Entry(input_frame, textvariable=self.output_dir_path, width=50).grid(row=2, column=1, sticky=tk.W+tk.E, padx=5, pady=5)
        self.ui_elements["browse_output_btn"] = ttk.Button(input_frame, command=self.browse_output_dir)
        self.ui_elements["browse_output_btn"].grid(row=2, column=2, sticky=tk.W, pady=5)
        
        # Parameters frame
        params_frame = ttk.LabelFrame(main_frame, padding="10")
        params_frame.pack(fill=tk.X, pady=5)
        self.ui_elements["params_frame"] = params_frame
        
        # Content type
        self.ui_elements["content_label"] = ttk.Label(params_frame)
        self.ui_elements["content_label"].grid(row=0, column=0, sticky=tk.W, pady=5)
        content_combo = ttk.Combobox(params_frame, textvariable=self.content_choice, width=15)
        content_combo['values'] = ('transcript', 'translation', 'both')
        content_combo.current(2)
        content_combo.grid(row=0, column=1, sticky=tk.W, pady=5)
        self.ui_elements["content_desc_label"] = ttk.Label(params_frame)
        self.ui_elements["content_desc_label"].grid(row=0, column=2, sticky=tk.W, pady=5)
        
        # Target language
        self.ui_elements["target_lang_label"] = ttk.Label(params_frame)
        self.ui_elements["target_lang_label"].grid(row=1, column=0, sticky=tk.W, pady=5)
        target_lang_combo = ttk.Combobox(params_frame, textvariable=self.target_language, width=15)
        target_lang_combo['values'] = ('Simplified Chinese', 'Traditional Chinese', 'English', 'Japanese', 'Korean', 'Russian', 'Spanish', 'French', 'German')
        target_lang_combo.current(0)
        target_lang_combo.grid(row=1, column=1, sticky=tk.W, pady=5)

        # Model selection
        self.ui_elements["model_label"] = ttk.Label(params_frame)
        self.ui_elements["model_label"].grid(row=1, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        model_combo = ttk.Combobox(params_frame, textvariable=self.model_name, width=22)
        
        # Updated Model List for Gemini 3.0
        model_combo['values'] = ('gemini-3-flash-preview', 'gemini-3-pro-preview', 'gemini-2.5-flash')
        model_combo.current(0) # Default to gemini-3-flash-preview
        
        model_combo.grid(row=1, column=3, sticky=tk.W, pady=5)
        self.ui_elements["model_combo"] = model_combo
        
        # Parallel processing
        self.ui_elements["parallel_label"] = ttk.Label(params_frame)
        self.ui_elements["parallel_label"].grid(row=2, column=2, sticky=tk.W, pady=5, padx=(10, 0))
        self.ui_elements["parallel_spinbox"] = ttk.Spinbox(params_frame, from_=1, to=20, increment=1, textvariable=self.max_workers, width=5)
        self.ui_elements["parallel_spinbox"].grid(row=2, column=3, sticky=tk.W, pady=5)

        # Audio splitting parameters
        self.ui_elements["max_length_label"] = ttk.Label(params_frame)
        self.ui_elements["max_length_label"].grid(row=3, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=60, to=900, increment=30, textvariable=self.max_length, width=5).grid(row=3, column=1, sticky=tk.W, pady=5)
        
        self.ui_elements["silence_length_label"] = ttk.Label(params_frame)
        self.ui_elements["silence_length_label"].grid(row=4, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=100, to=2000, increment=100, textvariable=self.silence_length, width=5).grid(row=4, column=1, sticky=tk.W, pady=5)
        
        self.ui_elements["silence_threshold_label"] = ttk.Label(params_frame)
        self.ui_elements["silence_threshold_label"].grid(row=5, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=-60, to=-20, increment=5, textvariable=self.silence_threshold, width=5).grid(row=5, column=1, sticky=tk.W, pady=5)
        
        self.ui_elements["first_chunk_offset_label"] = ttk.Label(params_frame)
        self.ui_elements["first_chunk_offset_label"].grid(row=6, column=0, sticky=tk.W, pady=5)
        ttk.Spinbox(params_frame, from_=-5.0, to=5.0, increment=0.1, textvariable=self.first_chunk_offset, width=5).grid(row=6, column=1, sticky=tk.W, pady=5)
        
        # Resume from breakpoint
        self.ui_elements["skip_existing_checkbox"] = ttk.Checkbutton(params_frame, variable=self.skip_existing)
        self.ui_elements["skip_existing_checkbox"].grid(row=7, column=2, columnspan=2, sticky=tk.W, pady=5)
        
        # Cleanup
        self.ui_elements["cleanup_checkbox"] = ttk.Checkbutton(params_frame, variable=self.cleanup)
        self.ui_elements["cleanup_checkbox"].grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        # Skip audio segmentation
        skip_split_frame = ttk.Frame(params_frame)
        skip_split_frame.grid(row=8, column=0, columnspan=4, sticky=tk.W, pady=5)
        
        self.ui_elements["skip_split_checkbox"] = ttk.Checkbutton(skip_split_frame, variable=self.skip_split, command=self.toggle_audio_chunks_controls)
        self.ui_elements["skip_split_checkbox"].grid(row=0, column=0, sticky=tk.W)
        
        self.ui_elements["audio_chunks_dir_label"] = ttk.Label(skip_split_frame)
        self.ui_elements["audio_chunks_dir_label"].grid(row=0, column=1, sticky=tk.W, padx=(20, 5))
        self.ui_elements["audio_chunks_dir_entry"] = ttk.Entry(skip_split_frame, textvariable=self.audio_chunks_dir, width=30)
        self.ui_elements["audio_chunks_dir_entry"].grid(row=0, column=2, sticky=tk.W, padx=5)
        self.ui_elements["browse_chunks_btn"] = ttk.Button(skip_split_frame, command=self.browse_audio_chunks_dir)
        self.ui_elements["browse_chunks_btn"].grid(row=0, column=3, sticky=tk.W)
        self.ui_elements["use_default_chunks_btn"] = ttk.Button(skip_split_frame, command=self.use_default_audio_chunks_dir)
        self.ui_elements["use_default_chunks_btn"].grid(row=0, column=4, sticky=tk.W, padx=5)
        
        self.toggle_audio_chunks_controls()
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_button = ttk.Button(button_frame, command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        self.retry_button = ttk.Button(button_frame, command=self.retry_combine, state=tk.DISABLED)
        self.retry_button.pack(side=tk.LEFT, padx=5)
        self.ui_elements["retry_button"] = self.retry_button
        
        self.open_error_file_button = ttk.Button(button_frame, command=self.open_error_file, state=tk.DISABLED)
        self.open_error_file_button.pack(side=tk.LEFT, padx=5)
        self.ui_elements["open_error_file_button"] = self.open_error_file_button
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, padding="10")
        progress_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        self.ui_elements["progress_frame"] = progress_frame
        
        self.progress_text = scrolledtext.ScrolledText(progress_frame, wrap=tk.WORD, height=15)
        self.progress_text.pack(fill=tk.BOTH, expand=True)
        self.progress_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        status_bar = ttk.Label(self, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def toggle_api_key_visibility(self, entry):
        """Toggle the visibility of the API key."""
        if entry.cget('show') == '*':
            entry.config(show='')
        else:
            entry.config(show='*')
    
    def browse_input_file(self):
        """Open a file browser to select an input file."""
        lang = self.current_language.get()
        filetypes = (
            ("All supported files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg *.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v"),
            ("Audio files", "*.mp3 *.wav *.flac *.m4a *.aac *.ogg"),
            ("Video files", "*.mp4 *.avi *.mkv *.mov *.wmv *.flv *.webm *.m4v"),
            ("All files", "*.*")
        )
        filepath = filedialog.askopenfilename(
            title=translations[lang]["select_file"],
            filetypes=filetypes
        )
        
        if filepath:
            self.input_file_path.set(filepath)
            if not self.output_dir_path.get():
                input_path = pathlib.Path(filepath)
                self.output_dir_path.set(os.path.join(input_path.parent, input_path.stem))
            
            if is_video_file(filepath):
                self.file_type_var.set(translations[lang]["video_file"])
            else:
                self.file_type_var.set(translations[lang]["audio_file"])
    
    def browse_output_dir(self):
        """Open a file browser to select an output directory."""
        dirpath = filedialog.askdirectory(
            title=translations[self.current_language.get()]["select_output_dir"]
        )
        if dirpath:
            self.output_dir_path.set(dirpath)
    
    def start_processing(self):
        """Start the audio processing."""
        lang = self.current_language.get()
        if not self.input_file_path.get():
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_no_input"])
            return
        
        if not self.api_key.get():
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_no_api_key"])
            return
        
        if self.skip_split.get() and not self.audio_chunks_dir.get():
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_skip_segmentation_no_dir"])
            return
        
        if self.skip_split.get() and not os.path.isdir(self.audio_chunks_dir.get()):
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_chunks_dir_not_found"].format(dir=self.audio_chunks_dir.get()))
            return
        
        if not messagebox.askyesno(translations[lang]["confirm_start"], translations[lang]["confirm_start_message"]):
            return
        
        self.error_files = []
        self.waiting_for_user_fix = False
        self.retry_button.config(state=tk.DISABLED)
        self.open_error_file_button.config(state=tk.DISABLED)
        
        params = {
            'input_audio': self.input_file_path.get(),
            'output_dir': self.output_dir_path.get(),
            'api_key': self.api_key.get(),
            'content': self.content_choice.get(),
            'target_language': self.target_language.get(),
            'max_length': self.max_length.get(),
            'silence_length': self.silence_length.get(),
            'silence_threshold': self.silence_threshold.get(),
            'first_chunk_offset': self.first_chunk_offset.get(),
            'cleanup': self.cleanup.get(),
            'model_name': self.model_name.get(),
            'skip_split': self.skip_split.get(),
            'audio_chunks_dir': self.audio_chunks_dir.get() if self.skip_split.get() else None,
            'max_workers': self.max_workers.get(),
            'skip_existing': self.skip_existing.get()
        }
        
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set(translations[lang]["processing"])
        
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.delete(1.0, tk.END)
        self.progress_text.config(state=tk.DISABLED)
        
        self.start_time = time.time()
        
        self.process = multiprocessing.Process(
            target=run_pipeline,
            args=(params, self.progress_queue, self.control_queue)
        )
        self.process.start()
        self.process_pid = self.process.pid
        
        self.process_thread = threading.Thread(
            target=self.monitor_process,
            daemon=True
        )
        self.process_thread.start()
    
    def retry_combine(self):
        """Send a retry signal after the user fixes a timestamp error."""
        if not self.waiting_for_user_fix:
            return
        
        lang = self.current_language.get()
        self.status_var.set(translations[lang]["processing"])
        self.add_progress("Sending retry signal, continuing with transcript combination...")
        
        self.control_queue.put('RETRY_COMBINE')
        
        self.waiting_for_user_fix = False
        self.retry_button.config(state=tk.DISABLED)
        self.open_error_file_button.config(state=tk.DISABLED)
    
    def open_error_file(self):
        """Open the file with the timestamp error for the user to edit."""
        lang = self.current_language.get()
        if not self.error_files:
            messagebox.showinfo(
                translations[lang]["timestamp_error"],
                "No file information to fix."
            )
            return
            
        if len(self.error_files) == 1:
            self.open_file_with_default_editor(self.error_files[0])
            return
            
        error_file_options = []
        for i, error_info in enumerate(self.error_files):
            filename = error_info.get("file", "Unknown File")
            section = error_info.get("section", "Unknown Section")
            timestamp = error_info.get("timestamp_str", "Unknown Timestamp")
            error_file_options.append(f"{i+1}. {filename} - {section} - {timestamp}")
        
        select_dialog = tk.Toplevel(self)
        select_dialog.title(translations[lang]["select_error_file"])
        select_dialog.geometry("500x300")
        select_dialog.resizable(False, False)
        select_dialog.transient(self)
        select_dialog.grab_set()
        
        ttk.Label(select_dialog, text="Please select the error file to open:").pack(pady=10)
        
        listbox = tk.Listbox(select_dialog, width=70, height=10)
        for option in error_file_options:
            listbox.insert(tk.END, option)
        listbox.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
        
        def on_select():
            selected_idx = listbox.curselection()
            if selected_idx:
                self.open_file_with_default_editor(self.error_files[selected_idx[0]])
                select_dialog.destroy()
            else:
                messagebox.showinfo("Hint", "Please select a file")
        
        ttk.Button(select_dialog, text="Open Selected File", command=on_select).pack(pady=10)
        ttk.Button(select_dialog, text="Cancel", command=select_dialog.destroy).pack(pady=5)
    
    def open_file_with_default_editor(self, error_info):
        """Open the file containing the error with the system's default editor."""
        lang = self.current_language.get()
        if not error_info or "file" not in error_info:
            return
            
        filename = error_info.get("file")
        output_dir = self.output_dir_path.get()
        
        if not output_dir:
            messagebox.showerror(translations[lang]["error"], "Output directory is not set, cannot locate the error file.")
            return
            
        intermediate_dir = os.path.join(output_dir, "intermediate_transcripts")
        file_path = os.path.join(intermediate_dir, filename)
        
        if not os.path.exists(file_path):
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_file_not_found"])
            return
            
        line_num = error_info.get("line_num", 1)
        section = error_info.get("section", "Unknown Section")
        timestamp = error_info.get("timestamp_str", "Unknown Timestamp")
        
        self.add_progress(f"Opening error file: {file_path}")
        self.add_progress(f"Location of issue: Line {line_num}, Section: {section}, Timestamp: {timestamp}")
        self.add_progress("Please correct the timestamp format, then click the 'Retry Combine' button.")
        
        try:
            if sys.platform == 'win32':
                os.startfile(file_path)
            elif sys.platform == 'darwin':
                subprocess.call(['open', file_path])
            else:
                subprocess.call(['xdg-open', file_path])
        except Exception as e:
            messagebox.showerror(translations[lang]["error"], f"Unable to open file: {str(e)}")
    
    def stop_processing(self):
        """Forcefully stop the processing and all its child processes."""
        if not self.processing or not self.process_pid:
            return
        
        lang = self.current_language.get()
        if messagebox.askyesno(translations[lang]["confirm_stop"], translations[lang]["confirm_stop_message"]):
            self.add_progress(f"\n{translations[lang]['user_stop']}")
            self.status_var.set(translations[lang]["stopped"])
            
            if self.waiting_for_user_fix:
                self.control_queue.put('STOP_PROCESSING')
                self.waiting_for_user_fix = False
                self.retry_button.config(state=tk.DISABLED)
                self.open_error_file_button.config(state=tk.DISABLED)
            
            try:
                parent = psutil.Process(self.process_pid)
                children = parent.children(recursive=True)
                for child in children:
                    try:
                        child.terminate()
                    except psutil.NoSuchProcess:
                        pass
                
                try:
                    parent.terminate()
                except psutil.NoSuchProcess:
                    pass

                gone, still_alive = psutil.wait_procs(children + [parent], timeout=3)
                
                for p in still_alive:
                    try:
                        p.kill()
                    except psutil.NoSuchProcess:
                        pass
                
                self.add_progress("All related processes have been terminated.")
            except Exception as e:
                self.add_progress(f"An error occurred while terminating processes: {str(e)}")
            
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.process = None
            self.process_pid = None
    
    def check_queue(self):
        """Check the progress queue and update the display."""
        try:
            while True:
                message = self.progress_queue.get_nowait()
                if isinstance(message, dict) and message.get('type') == 'PARSE_ERROR':
                    self.handle_parse_error(message)
                else:
                    self.add_progress(message)
        except queue.Empty:
            pass
        self.after(100, self.check_queue)
    
    def handle_parse_error(self, error_data):
        """Handle special messages for timestamp parsing errors."""
        lang = self.current_language.get()
        self.error_files = error_data.get('errors', [])
        self.waiting_for_user_fix = True
        
        self.status_var.set(translations[lang]["waiting_for_fix"])
        self.retry_button.config(state=tk.NORMAL)
        self.open_error_file_button.config(state=tk.NORMAL)
        
        self.add_progress("\n" + "-" * 50)
        self.add_progress(f"Detected {len(self.error_files)} timestamp parsing errors!")
        self.add_progress(translations[lang]["timestamp_error_message"])
        self.add_progress("List of erroneous files:")
        
        for i, error in enumerate(self.error_files):
            error_detail = (f"  {i+1}. File: {error.get('file', 'Unknown')}, "
                           f"Section: {error.get('section', 'Unknown')}, "
                           f"Line: {error.get('line_num', 'Unknown')}, "
                           f"Timestamp: '{error.get('timestamp_str', 'Unknown')}'")
            self.add_progress(error_detail)
        
        self.add_progress("-" * 50)
        
        messagebox.showinfo(
            translations[lang]["timestamp_error"],
            translations[lang]["timestamp_error_message"]
        )
    
    def add_progress(self, message):
        """Add a message to the progress display."""
        self.progress_text.config(state=tk.NORMAL)
        self.progress_text.insert(tk.END, f"{message}\n")
        self.progress_text.see(tk.END)
        self.progress_text.config(state=tk.DISABLED)
        self.update_idletasks()
    
    def on_closing(self):
        """Handle window closing."""
        if self.processing:
            if not messagebox.askyesno(translations[self.current_language.get()]["confirm_close"], translations[self.current_language.get()]["confirm_close_message"]):
                return
        self.destroy()
    
    def change_language(self, event=None):
        """Change the language of the GUI."""
        lang = self.current_language.get()
        
        self.title(translations[lang]["title"])
        self.ui_elements["language_label"].config(text=translations[lang]["language"])
        self.ui_elements["basic_settings_frame"].config(text=translations[lang]["basic_settings"])
        self.ui_elements["input_file_label"].config(text=translations[lang]["input_file"])
        self.ui_elements["browse_input_btn"].config(text=translations[lang]["browse"])
        self.ui_elements["api_key_label"].config(text=translations[lang]["api_key"])
        self.ui_elements["show_hide_btn"].config(text=translations[lang]["show_hide"])
        self.ui_elements["output_dir_label"].config(text=translations[lang]["output_dir"])
        self.ui_elements["browse_output_btn"].config(text=translations[lang]["browse"])
        self.ui_elements["params_frame"].config(text=translations[lang]["process_params"])
        self.ui_elements["content_label"].config(text=translations[lang]["content_type"])
        self.ui_elements["content_desc_label"].config(text=translations[lang]["content_desc"])
        self.ui_elements["target_lang_label"].config(text=translations[lang]["target_language"])
        self.ui_elements["max_length_label"].config(text=translations[lang]["max_length"])
        self.ui_elements["silence_length_label"].config(text=translations[lang]["silence_length"])
        self.ui_elements["silence_threshold_label"].config(text=translations[lang]["silence_threshold"])
        self.ui_elements["first_chunk_offset_label"].config(text=translations[lang]["first_chunk_offset"])
        self.ui_elements["cleanup_checkbox"].config(text=translations[lang]["cleanup"])
        self.start_button.config(text=translations[lang]["start_process"])
        self.stop_button.config(text=translations[lang]["stop_process"])
        self.ui_elements["progress_frame"].config(text=translations[lang]["progress"])
        self.ui_elements["retry_button"].config(text=translations[lang]["retry_combine"])
        self.ui_elements["open_error_file_button"].config(text=translations[lang]["open_error_file"])
        self.ui_elements["model_label"].config(text=translations[lang]["model"])
        self.ui_elements["parallel_label"].config(text=translations[lang]["parallel_requests"])
        self.ui_elements["skip_existing_checkbox"].config(text=translations[lang]["resume_from_breakpoint"])
        self.ui_elements["skip_split_checkbox"].config(text=translations[lang]["skip_audio_segmentation"])
        self.ui_elements["audio_chunks_dir_label"].config(text=translations[lang]["audio_chunks_dir_label"])
        self.ui_elements["browse_chunks_btn"].config(text=translations[lang]["browse"])
        self.ui_elements["use_default_chunks_btn"].config(text=translations[lang]["use_default_dir"])

        if self.waiting_for_user_fix:
            self.status_var.set(translations[lang]["waiting_for_fix"])
        else:
            self.status_var.set(translations[lang]["ready"])
        
        if self.input_file_path.get():
            if is_video_file(self.input_file_path.get()):
                self.file_type_var.set(translations[lang]["video_file"])
            else:
                self.file_type_var.set(translations[lang]["audio_file"])
        else:
            self.file_type_var.set(translations[lang]["file_not_selected"])

    def monitor_process(self):
        """Monitor the processing and update the UI when it's complete."""
        if not self.process:
            return
            
        self.process.join()
        
        if not self.processing or self.waiting_for_user_fix:
            return
            
        elapsed = time.time() - self.start_time
        self.add_progress(f"\nProcessing complete! Total time: {elapsed:.2f} seconds")
        
        self.processing = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set(translations[self.current_language.get()]["ready"])
        
        if self.output_dir_path.get():
            messagebox.showinfo(
                translations[self.current_language.get()]["complete"], 
                translations[self.current_language.get()]["complete_message"].format(output_dir=self.output_dir_path.get())
            )
    
    def toggle_audio_chunks_controls(self):
        """Enable or disable the audio chunks directory controls."""
        state = "normal" if self.skip_split.get() else "disabled"
        self.ui_elements["audio_chunks_dir_entry"].config(state=state)
        self.ui_elements["browse_chunks_btn"].config(state=state)
        self.ui_elements["audio_chunks_dir_label"].config(state=state)
        self.ui_elements["use_default_chunks_btn"].config(state=state)
            
    def browse_audio_chunks_dir(self):
        """Open a file browser to select an existing audio chunks directory."""
        dirpath = filedialog.askdirectory(title="Select Existing Audio Chunks Directory")
        if dirpath:
            self.audio_chunks_dir.set(dirpath)
    
    def use_default_audio_chunks_dir(self):
        """Use the default audio chunks directory."""
        lang = self.current_language.get()
        if not self.input_file_path.get():
            messagebox.showerror(translations[lang]["error"], translations[lang]["error_select_input_first"])
            return
            
        input_path = pathlib.Path(self.input_file_path.get())
        default_output_dir = os.path.join(input_path.parent, input_path.stem)
        default_chunks_dir = os.path.join(default_output_dir, "audio_chunks")
        
        if not os.path.exists(default_chunks_dir):
            if not messagebox.askyesno("Directory Not Found", translations[lang]["dir_not_exist_prompt"].format(dir=default_chunks_dir)):
                return
        
        self.audio_chunks_dir.set(default_chunks_dir)
        self.add_progress(f"Set default audio chunks directory: {default_chunks_dir}")

if __name__ == "__main__":
    app = AudioProcessorGUI()
    app.mainloop()