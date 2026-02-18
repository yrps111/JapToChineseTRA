# -*- coding: utf-8 -*-
"""
日语视频自动翻译字幕工具 v2.1
作者: Yrps
功能: MP4日语视频 → 语音识别 → AI翻译 → SRT/ASS字幕
v2.1 更新:
  - 修复批量处理完成后程序崩溃/卡死的问题
  - 新增"路径设置"选项卡，所有外部程序路径可通过GUI配置
  - 支持自定义 Whisper模型/CUDA/cuDNN/FFmpeg 路径
v2.0 更新:
  - 修复处理完成后卡死的问题
  - 支持批量处理多个视频文件
  - 一键清理临时文件
  - 字幕与临时文件分离存放
"""

import os
import sys
import gc
import json
import time
import math
import shutil
import threading
import traceback as tb_module
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import timedelta
from queue import Queue, Empty

# ============================================================
#  第三方依赖
# ============================================================
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("请先安装 faster-whisper: pip install faster-whisper")
    sys.exit(1)

try:
    from openai import OpenAI
except ImportError:
    print("请先安装 openai: pip install openai")
    sys.exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("请先安装 pydub: pip install pydub")
    sys.exit(1)


# ============================================================
#  配置管理（最先加载，供后续 setup 函数使用）
# ============================================================
class ConfigManager:
    """持久化配置管理器"""

    def __init__(self):
        self.config_dir = Path.home() / ".jp_subtitle_tool"
        self.config_file = self.config_dir / "config.json"
        self.default_config = {
            "api_base_url": "",
            "api_key": "",
            "model_name": "",
            "available_models": [],
            "whisper_model_size": "large-v3",
            "whisper_device": "cuda",
            "whisper_compute_type": "float16",
            "segment_duration": 30,
            "translation_workers": 4,
            "last_input_dir": "",
            "last_output_dir": "",
            "target_language": "中文",
            # [v2.1] 外部程序路径 —— 默认值为你当前的路径
            "whisper_model_path": r"F:\KKTV\KaiF\model",
            "cuda_bin_path": r"F:\1A111BIGDESIGN\CUDA\v12.6\bin",
            "cudnn_bin_path": r"F:\1A111BIGDESIGN\cudnn\v9.6\bin",
            "ffmpeg_bin_path": r"E:\FFmpeg\ffmpeg-8.0.1-essentials_build\bin",
        }
        self.config = self._load()

    def _load(self) -> dict:
        if self.config_file.exists():
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    saved = json.load(f)
                merged = {**self.default_config, **saved}
                return merged
            except Exception:
                return dict(self.default_config)
        return dict(self.default_config)

    def save(self):
        self.config_dir.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def set(self, key, value):
        self.config[key] = value

    def update(self, d: dict):
        self.config.update(d)


# 提前加载配置，供 setup 函数使用
_early_config = ConfigManager()


# ============================================================
#  CUDA 路径设置 [v2.1] 参数化
# ============================================================
def _setup_cuda(config=None):
    """确保 CTranslate2 能找到 CUDA 的 cuBLAS 等库"""
    if config is None:
        config = _early_config
    cuda_paths = []
    for key in ("cuda_bin_path", "cudnn_bin_path"):
        p = config.get(key, "")
        if p and os.path.isdir(p):
            cuda_paths.append(p)

    current_path = os.environ.get("PATH", "")
    added = [p for p in cuda_paths if p not in current_path]
    if added:
        os.environ["PATH"] = os.pathsep.join(added) + os.pathsep + current_path

_setup_cuda()


# ============================================================
#  FFmpeg 路径设置 [v2.1] 参数化
# ============================================================
def _setup_ffmpeg(config=None):
    """确保 pydub 能找到 ffmpeg"""
    if config is None:
        config = _early_config
    ffmpeg_bin_dir = config.get("ffmpeg_bin_path", "")

    if ffmpeg_bin_dir:
        ffmpeg_exe = os.path.join(ffmpeg_bin_dir, "ffmpeg.exe")
        ffprobe_exe = os.path.join(ffmpeg_bin_dir, "ffprobe.exe")

        if os.path.isfile(ffmpeg_exe):
            AudioSegment.converter = ffmpeg_exe
            if os.path.isfile(ffprobe_exe):
                AudioSegment.ffprobe = ffprobe_exe

            current_path = os.environ.get("PATH", "")
            if ffmpeg_bin_dir not in current_path:
                os.environ["PATH"] = ffmpeg_bin_dir + os.pathsep + current_path
            return

    # 回退：从系统 PATH 中查找
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        AudioSegment.converter = ffmpeg_path
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path:
            AudioSegment.ffprobe = ffprobe_path

_setup_ffmpeg()


# ============================================================
#  音频提取与分段
# ============================================================
class AudioProcessor:
    """从视频提取音频并按指定时长分段"""

    def __init__(self, segment_duration: int = 30):
        self.segment_duration = segment_duration

    def extract_audio(self, video_path: str, output_dir: str,
                      progress_callback=None, stop_flag=None) -> str:
        audio_path = os.path.join(output_dir, "full_audio.wav")
        if progress_callback:
            progress_callback("正在提取音频...")

        audio = AudioSegment.from_file(video_path)
        audio = audio.set_frame_rate(16000).set_channels(1)
        audio.export(audio_path, format="wav")
        del audio

        if progress_callback:
            progress_callback(f"音频提取完成: {audio_path}")
        return audio_path

    def split_audio(self, audio_path: str, output_dir: str,
                    progress_callback=None, stop_flag=None) -> list:
        if progress_callback:
            progress_callback("正在分段音频...")

        audio = AudioSegment.from_wav(audio_path)
        total_ms = len(audio)
        seg_ms = self.segment_duration * 1000
        segments = []
        num_segments = math.ceil(total_ms / seg_ms)

        seg_dir = os.path.join(output_dir, "segments")
        os.makedirs(seg_dir, exist_ok=True)

        for i in range(num_segments):
            if stop_flag and not stop_flag():
                if progress_callback:
                    progress_callback("分段已取消")
                del audio
                return segments

            start_ms = i * seg_ms
            end_ms = min((i + 1) * seg_ms, total_ms)
            chunk = audio[start_ms:end_ms]

            seg_path = os.path.join(seg_dir, f"seg_{i:05d}.wav")
            chunk.export(seg_path, format="wav")
            del chunk

            start_sec = start_ms / 1000.0
            end_sec = end_ms / 1000.0
            segments.append((seg_path, start_sec, end_sec))

            if progress_callback:
                progress_callback(
                    f"分段进度: {i + 1}/{num_segments} "
                    f"({start_sec:.1f}s - {end_sec:.1f}s)"
                )

        del audio

        if progress_callback:
            progress_callback(f"音频分段完成，共 {num_segments} 段")
        return segments


# ============================================================
#  语音识别 (faster-whisper) [v2.1] model_path 参数化
# ============================================================
class SpeechRecognizer:
    """使用 faster-whisper 进行日语语音识别"""

    def __init__(self, model_path: str,
                 model_size: str = "large-v3",
                 device: str = "cuda",
                 compute_type: str = "float16"):
        self.model_path = model_path
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.model = None

    def load_model(self, progress_callback=None):
        if progress_callback:
            progress_callback(
                f"正在加载 Whisper 模型: {self.model_path} "
                f"(设备: {self.device}, 精度: {self.compute_type})..."
            )
        self.model = WhisperModel(
            self.model_path,
            device=self.device,
            compute_type=self.compute_type,
        )
        if progress_callback:
            progress_callback("Whisper 模型加载完成！")

    def unload_model(self, progress_callback=None):
        """[v2.1] 安全卸载模型，每一步都 try/except"""
        if self.model is not None:
            try:
                del self.model
            except Exception:
                pass
            self.model = None

            try:
                gc.collect()
            except Exception:
                pass

            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            if progress_callback:
                progress_callback("Whisper 模型已卸载，GPU显存已释放")

    def transcribe_segment(self, audio_path: str, offset: float = 0.0) -> list:
        if self.model is None:
            raise RuntimeError("Whisper 模型尚未加载")

        segments_iter, info = self.model.transcribe(
            audio_path,
            language="ja",
            beam_size=5,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500),
        )

        results = []
        for seg in segments_iter:
            results.append({
                "start": round(seg.start + offset, 3),
                "end": round(seg.end + offset, 3),
                "text": seg.text.strip(),
            })
        return results

    def transcribe_all(self, segments: list,
                       progress_callback=None,
                       stop_flag=None) -> list:
        all_entries = []
        total = len(segments)

        for idx, (seg_path, start_sec, end_sec) in enumerate(segments):
            if stop_flag and not stop_flag():
                if progress_callback:
                    progress_callback("语音识别已取消")
                return all_entries

            if progress_callback:
                progress_callback(
                    f"语音识别: {idx + 1}/{total} "
                    f"({start_sec:.1f}s - {end_sec:.1f}s)"
                )

            entries = self.transcribe_segment(seg_path, offset=start_sec)
            all_entries.extend(entries)

        all_entries = [e for e in all_entries if e["text"]]
        all_entries.sort(key=lambda x: x["start"])

        if progress_callback:
            progress_callback(f"语音识别完成，共 {len(all_entries)} 条字幕")
        return all_entries


# ============================================================
#  AI 翻译 (OpenAI兼容API)
# ============================================================
class Translator:
    """使用 OpenAI 兼容 API 进行批量翻译"""

    def __init__(self, api_base_url: str, api_key: str,
                 model_name: str, target_lang: str = "中文"):
        self.client = OpenAI(
            base_url=api_base_url,
            api_key=api_key,
        )
        self.model_name = model_name
        self.target_lang = target_lang

    @staticmethod
    def fetch_models(api_base_url: str, api_key: str) -> list:
        try:
            client = OpenAI(base_url=api_base_url, api_key=api_key)
            models = client.models.list()
            return [m.id for m in models.data]
        except Exception:
            return []

    def translate_single(self, text: str) -> str:
        if not text.strip():
            return ""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"你是专业且私人的日语翻译。请将用户提供的日语文本准确翻译为{self.target_lang}。"
                            f"只输出翻译结果，不要添加任何解释、注释或额外内容。"
                            f"保持语句自然流畅。如果是专有名词请保留原文并在括号中标注翻译。"
                        ),
                    },
                    {"role": "user", "content": text},
                ],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            return f"[翻译失败: {e}]"

    def translate_batch(self, entries: list, max_workers: int = 4,
                        progress_callback=None, stop_flag=None) -> list:
        results = [None] * len(entries)
        total = len(entries)
        completed = 0
        lock = threading.Lock()

        def _task(idx, entry):
            nonlocal completed
            if stop_flag and not stop_flag():
                return idx, {
                    "start": entry["start"],
                    "end": entry["end"],
                    "original": entry["text"],
                    "translated": "[已取消]",
                }

            translated = self.translate_single(entry["text"])
            result = {
                "start": entry["start"],
                "end": entry["end"],
                "original": entry["text"],
                "translated": translated,
            }
            with lock:
                completed += 1
                if progress_callback:
                    progress_callback(f"翻译进度: {completed}/{total}")
            return idx, result

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = [
                pool.submit(_task, i, e)
                for i, e in enumerate(entries)
            ]
            for future in as_completed(futures):
                idx, result = future.result()
                results[idx] = result

        return results


# ============================================================
#  SRT / ASS 字幕生成
# ============================================================
class SubtitleWriter:
    """生成 SRT / ASS 格式字幕文件"""

    @staticmethod
    def seconds_to_srt_time(seconds: float) -> str:
        total_seconds = int(seconds)
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        secs = total_seconds % 60
        millis = int((seconds - int(seconds)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

    @staticmethod
    def write_srt(entries: list, output_path: str,
                  include_original: bool = True):
        with open(output_path, "w", encoding="utf-8") as f:
            for i, entry in enumerate(entries, 1):
                start_t = SubtitleWriter.seconds_to_srt_time(entry["start"])
                end_t = SubtitleWriter.seconds_to_srt_time(entry["end"])
                f.write(f"{i}\n")
                f.write(f"{start_t} --> {end_t}\n")
                if include_original:
                    f.write(f"{entry['translated']}\n")
                    f.write(f"{entry['original']}\n")
                else:
                    f.write(f"{entry['translated']}\n")
                f.write("\n")

    @staticmethod
    def write_ass(entries: list, output_path: str,
                  include_original: bool = True):
        with open(output_path, "w", encoding="utf-8-sig") as f:
            f.write("[Script Info]\n")
            f.write("Title: JP Subtitle Tool Output\n")
            f.write("ScriptType: v4.00+\n")
            f.write("WrapStyle: 0\n")
            f.write("PlayResX: 1920\n")
            f.write("PlayResY: 1080\n")
            f.write("ScaledBorderAndShadow: yes\n\n")

            f.write("[V4+ Styles]\n")
            f.write("Format: Name, Fontname, Fontsize, PrimaryColour, "
                    "SecondaryColour, OutlineColour, BackColour, Bold, "
                    "Italic, Underline, StrikeOut, ScaleX, ScaleY, "
                    "Spacing, Angle, BorderStyle, Outline, Shadow, "
                    "Alignment, MarginL, MarginR, MarginV, Encoding\n")
            f.write("Style: CN,Microsoft YaHei,48,&H00FFFFFF,"
                    "&H000000FF,&H00000000,&H80000000,"
                    "0,0,0,0,100,100,0,0,1,2,1,2,10,10,30,1\n")
            f.write("Style: JP,Microsoft YaHei,36,&H0000FFFF,"
                    "&H000000FF,&H00000000,&H80000000,"
                    "0,0,0,0,100,100,0,0,1,2,1,8,10,10,80,1\n\n")

            f.write("[Events]\n")
            f.write("Format: Layer, Start, End, Style, Name, "
                    "MarginL, MarginR, MarginV, Effect, Text\n")

            for entry in entries:
                start_t = SubtitleWriter._seconds_to_ass_time(entry["start"])
                end_t = SubtitleWriter._seconds_to_ass_time(entry["end"])
                f.write(f"Dialogue: 0,{start_t},{end_t},CN,,0,0,0,,"
                        f"{entry['translated']}\n")
                if include_original:
                    f.write(f"Dialogue: 0,{start_t},{end_t},JP,,0,0,0,,"
                            f"{entry['original']}\n")

    @staticmethod
    def _seconds_to_ass_time(seconds: float) -> str:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        cs = int((seconds - int(seconds)) * 100)
        return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


# ============================================================
#  主 GUI 应用  [v2.1] 大幅重构
# ============================================================
class MainApp:
    def __init__(self):
        self.config = _early_config  # [v2.1] 复用已加载的配置
        self.root = tk.Tk()
        self.root.title("日语视频翻译字幕工具 v2.1 - by Yrps")
        self.root.geometry("960x800")
        self.root.resizable(True, True)

        self._log_queue = Queue()

        self._build_ui()
        self._load_saved_config()

        self.is_running = False
        self._poll_log_queue()

    # -------------------- 日志系统 (线程安全) --------------------

    def _poll_log_queue(self):
        try:
            while True:
                msg = self._log_queue.get_nowait()
                self._write_log(msg)
        except Empty:
            pass
        self.root.after(100, self._poll_log_queue)

    def _write_log(self, message: str):
        """仅在主线程调用"""
        self.log_text.configure(state=tk.NORMAL)
        timestamp = time.strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _log(self, message: str):
        """线程安全：往队列丢消息"""
        self._log_queue.put(message)

    # -------------------- UI 构建 --------------------

    def _build_ui(self):
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # ---- Tab 1: 主操作 ----
        main_frame = ttk.Frame(notebook)
        notebook.add(main_frame, text="  主操作  ")
        self._build_main_tab(main_frame)

        # ---- Tab 2: API 设置 ----
        api_frame = ttk.Frame(notebook)
        notebook.add(api_frame, text="  API 设置  ")
        self._build_api_tab(api_frame)

        # ---- Tab 3: Whisper 设置 ----
        whisper_frame = ttk.Frame(notebook)
        notebook.add(whisper_frame, text="  Whisper 设置  ")
        self._build_whisper_tab(whisper_frame)

        # ---- Tab 4: 路径设置 [v2.1] 新增 ----
        path_frame = ttk.Frame(notebook)
        notebook.add(path_frame, text="  路径设置  ")
        self._build_path_tab(path_frame)

    def _build_main_tab(self, parent):
        """主操作选项卡"""
        # 文件选择
        file_frame = ttk.LabelFrame(parent, text="文件设置")
        file_frame.pack(fill=tk.X, padx=10, pady=5)

        ttk.Label(file_frame, text="输入视频:").grid(
            row=0, column=0, sticky=tk.NW, padx=5, pady=3)

        list_frame = ttk.Frame(file_frame)
        list_frame.grid(row=0, column=1, padx=5, pady=3, sticky=tk.EW)

        self.file_listbox = tk.Listbox(
            list_frame, height=4, selectmode=tk.EXTENDED,
            font=("Consolas", 9))
        file_scrollbar = ttk.Scrollbar(
            list_frame, orient=tk.VERTICAL,
            command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scrollbar.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        file_frame.columnconfigure(1, weight=1)

        btn_col = ttk.Frame(file_frame)
        btn_col.grid(row=0, column=2, padx=5, pady=3, sticky=tk.N)

        ttk.Button(btn_col, text="添加文件",
                   command=self._browse_input).pack(fill=tk.X, pady=2)
        ttk.Button(btn_col, text="移除选中",
                   command=self._remove_selected_files).pack(
            fill=tk.X, pady=2)
        ttk.Button(btn_col, text="清空列表",
                   command=self._clear_file_list).pack(fill=tk.X, pady=2)

        ttk.Label(file_frame, text="输出目录:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.output_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.output_var, width=60).grid(
            row=1, column=1, padx=5, pady=3, sticky=tk.EW)
        ttk.Button(file_frame, text="浏览",
                   command=self._browse_output).grid(
            row=1, column=2, padx=5, pady=3)

        # 字幕选项
        opt_frame = ttk.LabelFrame(parent, text="输出选项")
        opt_frame.pack(fill=tk.X, padx=10, pady=5)

        self.bilingual_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(opt_frame, text="双语字幕（翻译+日语原文）",
                        variable=self.bilingual_var).grid(
            row=0, column=0, padx=5, pady=3, sticky=tk.W)

        self.format_var = tk.StringVar(value="SRT")
        ttk.Label(opt_frame, text="字幕格式:").grid(
            row=0, column=1, padx=5, pady=3)
        ttk.Combobox(opt_frame, textvariable=self.format_var,
                     values=["SRT", "ASS"], state="readonly",
                     width=8).grid(row=0, column=2, padx=5, pady=3)

        # 执行按钮
        btn_frame = ttk.Frame(parent)
        btn_frame.pack(fill=tk.X, padx=10, pady=5)

        self.run_btn = ttk.Button(btn_frame, text="▶ 开始处理",
                                  command=self._start_processing)
        self.run_btn.pack(side=tk.LEFT, padx=5)

        self.stop_btn = ttk.Button(btn_frame, text="■ 停止",
                                   command=self._stop_processing,
                                   state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.clean_btn = ttk.Button(
            btn_frame, text="🗑 清理临时文件",
            command=self._clean_temp_files)
        self.clean_btn.pack(side=tk.LEFT, padx=5)

        # 进度
        progress_frame = ttk.Frame(parent)
        progress_frame.pack(fill=tk.X, padx=10, pady=3)

        self.batch_label_var = tk.StringVar(value="就绪")
        ttk.Label(progress_frame,
                  textvariable=self.batch_label_var).pack(
            side=tk.LEFT, padx=5)

        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(
            parent, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=10, pady=3)

        # 日志
        log_frame = ttk.LabelFrame(parent, text="处理日志")
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=15, state=tk.DISABLED,
            font=("Consolas", 9))
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def _build_api_tab(self, parent):
        """API 设置选项卡"""
        api_inner = ttk.LabelFrame(parent, text="翻译 API 配置")
        api_inner.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(api_inner, text="API 地址:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_url_var = tk.StringVar()
        ttk.Entry(api_inner, textvariable=self.api_url_var, width=55).grid(
            row=0, column=1, padx=5, pady=5)

        ttk.Label(api_inner, text="API Key:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.api_key_var = tk.StringVar()
        key_entry = ttk.Entry(api_inner, textvariable=self.api_key_var,
                              width=55, show="*")
        key_entry.grid(row=1, column=1, padx=5, pady=5)

        self.show_key_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            api_inner, text="显示Key", variable=self.show_key_var,
            command=lambda: key_entry.configure(
                show="" if self.show_key_var.get() else "*")
        ).grid(row=1, column=2, padx=5, pady=5)

        ttk.Label(api_inner, text="模型:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.model_var = tk.StringVar()
        self.model_combo = ttk.Combobox(
            api_inner, textvariable=self.model_var, width=52)
        self.model_combo.grid(row=2, column=1, padx=5, pady=5)

        btn_row = ttk.Frame(api_inner)
        btn_row.grid(row=3, column=0, columnspan=3, pady=10)

        ttk.Button(btn_row, text="获取模型列表",
                   command=self._fetch_models).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="保存配置",
                   command=self._save_api_config).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_row, text="测试连接",
                   command=self._test_connection).pack(side=tk.LEFT, padx=5)

    def _build_whisper_tab(self, parent):
        """Whisper 设置选项卡"""
        w_inner = ttk.LabelFrame(parent, text="语音识别设置")
        w_inner.pack(fill=tk.X, padx=10, pady=10)

        ttk.Label(w_inner, text="模型大小:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.whisper_size_var = tk.StringVar(value="large-v3")
        ttk.Combobox(
            w_inner, textvariable=self.whisper_size_var,
            values=["tiny", "base", "small", "medium",
                    "large-v2", "large-v3"],
            state="readonly", width=20
        ).grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(w_inner, text="运算设备:").grid(
            row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.whisper_device_var = tk.StringVar(value="cuda")
        ttk.Combobox(
            w_inner, textvariable=self.whisper_device_var,
            values=["cuda", "cpu"], state="readonly", width=20
        ).grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(w_inner, text="计算精度:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.whisper_compute_var = tk.StringVar(value="float16")
        ttk.Combobox(
            w_inner, textvariable=self.whisper_compute_var,
            values=["float16", "int8", "float32"],
            state="readonly", width=20
        ).grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(w_inner, text="分段时长(秒):").grid(
            row=3, column=0, sticky=tk.W, padx=5, pady=5)
        self.seg_dur_var = tk.IntVar(value=30)
        ttk.Spinbox(w_inner, from_=10, to=120,
                    textvariable=self.seg_dur_var, width=10).grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Label(w_inner, text="翻译并发数:").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.workers_var = tk.IntVar(value=4)
        ttk.Spinbox(w_inner, from_=1, to=16,
                    textvariable=self.workers_var, width=10).grid(
            row=4, column=1, sticky=tk.W, padx=5, pady=5)

        ttk.Button(w_inner, text="保存设置",
                   command=self._save_whisper_config).grid(
            row=5, column=0, columnspan=2, pady=10)

        ttk.Label(
            w_inner,
            text="提示: 3060 Laptop (6GB VRAM) 建议使用 large-v3 + float16\n"
                 "      如果显存不足，可切换至 medium 或使用 int8 精度\n"
                 "      分段时长30秒是较好的平衡点",
            foreground="gray"
        ).grid(row=6, column=0, columnspan=2, padx=10, pady=5)

    def _build_path_tab(self, parent):
        """[v2.1] 路径设置选项卡 —— 所有外部程序路径"""
        p_inner = ttk.LabelFrame(parent, text="外部程序路径配置")
        p_inner.pack(fill=tk.X, padx=10, pady=10)

        p_inner.columnconfigure(1, weight=1)

        # ---- Whisper 模型目录 ----
        ttk.Label(p_inner, text="Whisper 模型目录:").grid(
            row=0, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_whisper_var = tk.StringVar()
        ttk.Entry(p_inner, textvariable=self.path_whisper_var,
                  width=60).grid(row=0, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(p_inner, text="浏览",
                   command=lambda: self._browse_path(
                       self.path_whisper_var, "选择 Whisper 模型目录")
                   ).grid(row=0, column=2, padx=5, pady=5)

        ttk.Label(p_inner, text="包含 model.bin 的文件夹",
                  foreground="gray").grid(
            row=1, column=1, sticky=tk.W, padx=5, pady=0)

        # ---- CUDA bin ----
        ttk.Label(p_inner, text="CUDA bin 目录:").grid(
            row=2, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_cuda_var = tk.StringVar()
        ttk.Entry(p_inner, textvariable=self.path_cuda_var,
                  width=60).grid(row=2, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(p_inner, text="浏览",
                   command=lambda: self._browse_path(
                       self.path_cuda_var, "选择 CUDA bin 目录")
                   ).grid(row=2, column=2, padx=5, pady=5)

        ttk.Label(p_inner, text="通常为 CUDA 安装目录下的 bin 文件夹",
                  foreground="gray").grid(
            row=3, column=1, sticky=tk.W, padx=5, pady=0)

        # ---- cuDNN bin ----
        ttk.Label(p_inner, text="cuDNN bin 目录:").grid(
            row=4, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_cudnn_var = tk.StringVar()
        ttk.Entry(p_inner, textvariable=self.path_cudnn_var,
                  width=60).grid(row=4, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(p_inner, text="浏览",
                   command=lambda: self._browse_path(
                       self.path_cudnn_var, "选择 cuDNN bin 目录")
                   ).grid(row=4, column=2, padx=5, pady=5)

        ttk.Label(p_inner, text="通常为 cuDNN 解压目录下的 bin 文件夹",
                  foreground="gray").grid(
            row=5, column=1, sticky=tk.W, padx=5, pady=0)

        # ---- FFmpeg bin ----
        ttk.Label(p_inner, text="FFmpeg bin 目录:").grid(
            row=6, column=0, sticky=tk.W, padx=5, pady=5)
        self.path_ffmpeg_var = tk.StringVar()
        ttk.Entry(p_inner, textvariable=self.path_ffmpeg_var,
                  width=60).grid(row=6, column=1, padx=5, pady=5, sticky=tk.EW)
        ttk.Button(p_inner, text="浏览",
                   command=lambda: self._browse_path(
                       self.path_ffmpeg_var, "选择 FFmpeg bin 目录")
                   ).grid(row=6, column=2, padx=5, pady=5)

        ttk.Label(p_inner,
                  text="包含 ffmpeg.exe / ffprobe.exe 的文件夹",
                  foreground="gray").grid(
            row=7, column=1, sticky=tk.W, padx=5, pady=0)

        # ---- 按钮行 ----
        btn_row = ttk.Frame(p_inner)
        btn_row.grid(row=8, column=0, columnspan=3, pady=15)

        ttk.Button(btn_row, text="🔍 验证所有路径",
                   command=self._validate_paths).pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_row, text="💾 保存并应用",
                   command=self._save_path_config).pack(side=tk.LEFT, padx=10)

        # ---- 验证结果 ----
        self.path_result_var = tk.StringVar(value="")
        result_label = ttk.Label(p_inner,
                                 textvariable=self.path_result_var,
                                 wraplength=700, justify=tk.LEFT)
        result_label.grid(row=9, column=0, columnspan=3,
                          padx=10, pady=5, sticky=tk.W)

        # ---- 提示 ----
        ttk.Label(
            p_inner,
            text="提示: 首次使用请确保所有路径正确设置\n"
                 "      修改路径后需点击「保存并应用」才会生效\n"
                 "      如果使用 CPU 模式，CUDA/cuDNN 路径可留空",
            foreground="gray"
        ).grid(row=10, column=0, columnspan=3, padx=10, pady=10)

    def _browse_path(self, var: tk.StringVar, title: str):
        """通用目录浏览"""
        current = var.get().strip()
        path = filedialog.askdirectory(
            title=title,
            initialdir=current if current and os.path.isdir(current) else None
        )
        if path:
            var.set(path)

    # -------------------- 路径管理 [v2.1] --------------------

    def _validate_paths(self):
        """[v2.1] 验证所有路径是否存在"""
        results = []
        all_ok = True

        checks = [
            ("Whisper 模型", self.path_whisper_var.get().strip()),
            ("CUDA bin", self.path_cuda_var.get().strip()),
            ("cuDNN bin", self.path_cudnn_var.get().strip()),
            ("FFmpeg bin", self.path_ffmpeg_var.get().strip()),
        ]

        for name, path in checks:
            if not path:
                results.append(f"⚪ {name}: 未设置（留空）")
            elif os.path.isdir(path):
                results.append(f"✅ {name}: {path}")
            else:
                results.append(f"❌ {name}: 路径不存在 → {path}")
                all_ok = False

        # 额外检查 FFmpeg 可执行文件
        ffmpeg_dir = self.path_ffmpeg_var.get().strip()
        if ffmpeg_dir and os.path.isdir(ffmpeg_dir):
            ffmpeg_exe = os.path.join(ffmpeg_dir, "ffmpeg.exe")
            if not os.path.isfile(ffmpeg_exe):
                # 可能是 Linux/Mac
                ffmpeg_exe_unix = os.path.join(ffmpeg_dir, "ffmpeg")
                if not os.path.isfile(ffmpeg_exe_unix):
                    results.append(
                        f"⚠ 警告: FFmpeg 目录下未找到 ffmpeg 可执行文件")
                    all_ok = False

        result_text = "\n".join(results)
        if all_ok:
            result_text += "\n\n✅ 所有路径验证通过！"
        else:
            result_text += "\n\n⚠ 部分路径有问题，请检查"

        self.path_result_var.set(result_text)

    def _save_path_config(self):
        """[v2.1] 保存路径并立即应用"""
        self.config.update({
            "whisper_model_path": self.path_whisper_var.get().strip(),
            "cuda_bin_path": self.path_cuda_var.get().strip(),
            "cudnn_bin_path": self.path_cudnn_var.get().strip(),
            "ffmpeg_bin_path": self.path_ffmpeg_var.get().strip(),
        })
        self.config.save()

        # 重新应用路径
        _setup_cuda(self.config)
        _setup_ffmpeg(self.config)

        self._log("路径配置已保存并应用")
        messagebox.showinfo("保存成功",
                            "路径配置已保存！\n"
                            "CUDA / FFmpeg 路径已立即生效\n"
                            "Whisper 模型路径将在下次加载模型时使用")

    # -------------------- 配置恢复 --------------------

    def _load_saved_config(self):
        self.api_url_var.set(self.config.get("api_base_url", ""))
        self.api_key_var.set(self.config.get("api_key", ""))
        self.model_var.set(self.config.get("model_name", ""))
        self.whisper_size_var.set(
            self.config.get("whisper_model_size", "large-v3"))
        self.whisper_device_var.set(
            self.config.get("whisper_device", "cuda"))
        self.whisper_compute_var.set(
            self.config.get("whisper_compute_type", "float16"))
        self.seg_dur_var.set(self.config.get("segment_duration", 30))
        self.workers_var.set(self.config.get("translation_workers", 4))

        models = self.config.get("available_models", [])
        if models:
            self.model_combo["values"] = models

        # [v2.1] 路径设置恢复
        self.path_whisper_var.set(
            self.config.get("whisper_model_path", ""))
        self.path_cuda_var.set(
            self.config.get("cuda_bin_path", ""))
        self.path_cudnn_var.set(
            self.config.get("cudnn_bin_path", ""))
        self.path_ffmpeg_var.set(
            self.config.get("ffmpeg_bin_path", ""))

    # -------------------- 文件列表操作 --------------------

    def _browse_input(self):
        last_dir = self.config.get("last_input_dir", "")
        paths = filedialog.askopenfilenames(
            title="选择视频文件（可多选）",
            initialdir=last_dir if last_dir else None,
            filetypes=[("视频文件",
                        "*.mp4;*.mkv;*.avi;*.flv;*.webm;*.mov;*.wmv"),
                       ("所有文件", "*.*")]
        )
        if paths:
            existing = list(self.file_listbox.get(0, tk.END))
            for p in paths:
                if p not in existing:
                    self.file_listbox.insert(tk.END, p)
            self.config.set("last_input_dir",
                            os.path.dirname(paths[0]))
            self.config.save()

    def _remove_selected_files(self):
        selected = list(self.file_listbox.curselection())
        for idx in reversed(selected):
            self.file_listbox.delete(idx)

    def _clear_file_list(self):
        self.file_listbox.delete(0, tk.END)

    def _browse_output(self):
        last_dir = self.config.get("last_output_dir", "")
        path = filedialog.askdirectory(
            title="选择输出目录",
            initialdir=last_dir if last_dir else None
        )
        if path:
            self.output_var.set(path)
            self.config.set("last_output_dir", path)
            self.config.save()

    # -------------------- API 操作 --------------------

    def _fetch_models(self):
        url = self.api_url_var.get().strip()
        key = self.api_key_var.get().strip()
        if not url or not key:
            messagebox.showwarning("提示", "请先填写 API 地址和 Key")
            return
        self._log("正在获取模型列表...")
        models = Translator.fetch_models(url, key)
        if models:
            self.model_combo["values"] = sorted(models)
            self.config.set("available_models", sorted(models))
            self.config.save()
            self._log(f"获取到 {len(models)} 个模型")
            messagebox.showinfo("成功",
                                f"获取到 {len(models)} 个可用模型")
        else:
            messagebox.showerror("失败",
                                 "无法获取模型列表，请检查API地址和Key")

    def _save_api_config(self):
        self.config.update({
            "api_base_url": self.api_url_var.get().strip(),
            "api_key": self.api_key_var.get().strip(),
            "model_name": self.model_var.get().strip(),
        })
        self.config.save()
        messagebox.showinfo("保存成功",
                            "API配置已保存，下次打开无需重新填写")

    def _save_whisper_config(self):
        self.config.update({
            "whisper_model_size": self.whisper_size_var.get(),
            "whisper_device": self.whisper_device_var.get(),
            "whisper_compute_type": self.whisper_compute_var.get(),
            "segment_duration": self.seg_dur_var.get(),
            "translation_workers": self.workers_var.get(),
        })
        self.config.save()
        messagebox.showinfo("保存成功", "Whisper设置已保存")

    def _test_connection(self):
        url = self.api_url_var.get().strip()
        key = self.api_key_var.get().strip()
        model = self.model_var.get().strip()
        if not all([url, key, model]):
            messagebox.showwarning("提示", "请填写完整的API信息")
            return
        self._log("测试API连接...")
        try:
            translator = Translator(url, key, model)
            result = translator.translate_single("テスト")
            self._log(f"测试翻译 'テスト' → '{result}'")
            messagebox.showinfo("连接成功",
                                f"API连接正常！\n测试翻译: テスト → {result}")
        except Exception as e:
            messagebox.showerror("连接失败", f"错误: {e}")

    # -------------------- 一键清理 --------------------

    def _clean_temp_files(self):
        output_dir = self.output_var.get().strip()
        if not output_dir or not os.path.isdir(output_dir):
            messagebox.showwarning("提示", "请先设置有效的输出目录")
            return

        temp_dirs = []
        total_size = 0

        for root_dir, dirs, files in os.walk(output_dir):
            for d in dirs:
                if d == "temp":
                    temp_path = os.path.join(root_dir, d)
                    temp_dirs.append(temp_path)
                    for dp, dn, fns in os.walk(temp_path):
                        for fn in fns:
                            fp = os.path.join(dp, fn)
                            try:
                                total_size += os.path.getsize(fp)
                            except OSError:
                                pass

        if not temp_dirs:
            messagebox.showinfo("提示", "没有找到需要清理的临时文件")
            return

        size_mb = total_size / (1024 * 1024)
        confirm = messagebox.askyesno(
            "确认清理",
            f"找到 {len(temp_dirs)} 个临时文件夹\n"
            f"总计约 {size_mb:.1f} MB\n\n"
            f"确定要删除这些临时文件吗？\n"
            f"（字幕文件不会被删除）"
        )

        if confirm:
            deleted_count = 0
            for temp_path in temp_dirs:
                try:
                    shutil.rmtree(temp_path)
                    deleted_count += 1
                    self._log(f"已清理: {temp_path}")
                except Exception as e:
                    self._log(f"清理失败: {temp_path} - {e}")

            self._log(f"清理完成！共删除 {deleted_count} 个临时文件夹，"
                      f"释放约 {size_mb:.1f} MB")
            messagebox.showinfo("清理完成",
                                f"已删除 {deleted_count} 个临时文件夹\n"
                                f"释放约 {size_mb:.1f} MB 空间")

    # -------------------- 处理流程 [v2.1 核心修复] --------------------

    def _start_processing(self):
        video_paths = list(self.file_listbox.get(0, tk.END))
        output_dir = self.output_var.get().strip()

        if not video_paths:
            messagebox.showwarning("提示", "请先添加视频文件")
            return
        if not output_dir:
            messagebox.showwarning("提示", "请选择输出目录")
            return

        missing = [p for p in video_paths if not os.path.isfile(p)]
        if missing:
            messagebox.showwarning(
                "文件不存在",
                "以下文件不存在:\n" +
                "\n".join(missing[:5]) +
                ("\n..." if len(missing) > 5 else "")
            )
            return

        # [v2.1] 检查 Whisper 模型路径
        whisper_path = self.config.get("whisper_model_path", "")
        if not whisper_path or not os.path.isdir(whisper_path):
            messagebox.showwarning(
                "提示",
                "Whisper 模型路径无效！\n"
                "请在「路径设置」选项卡中设置正确的模型目录")
            return

        api_url = (self.api_url_var.get().strip()
                   or self.config.get("api_base_url"))
        api_key = (self.api_key_var.get().strip()
                   or self.config.get("api_key"))
        model = (self.model_var.get().strip()
                 or self.config.get("model_name"))

        if not all([api_url, api_key, model]):
            messagebox.showwarning("提示",
                                   "请先在 'API设置' 中配置翻译API")
            return

        self.is_running = True
        self.run_btn.configure(state=tk.DISABLED)
        self.stop_btn.configure(state=tk.NORMAL)
        self.clean_btn.configure(state=tk.DISABLED)
        self.progress_var.set(0)

        thread = threading.Thread(
            target=self._batch_pipeline,
            args=(video_paths, output_dir, api_url, api_key, model),
            daemon=True,
        )
        thread.start()

    def _stop_processing(self):
        self.is_running = False
        self._log("⚠ 用户请求停止，正在等待当前步骤完成...")
        self.stop_btn.configure(state=tk.DISABLED)

    def _set_progress(self, value):
        self.root.after(0, lambda v=value: self.progress_var.set(v))

    def _set_batch_label(self, text):
        self.root.after(0, lambda t=text: self.batch_label_var.set(t))

    def _batch_pipeline(self, video_paths, output_dir,
                        api_url, api_key, model):
        """
        [v2.1] 批量处理流水线
        关键修复: finally 中只做资源释放，UI 更新统一交给 _on_batch_finished
        """
        total_videos = len(video_paths)
        subtitle_dir = os.path.join(output_dir, "subtitles")
        os.makedirs(subtitle_dir, exist_ok=True)

        recognizer_holder = [None]
        success_count = 0
        fail_count = 0

        try:
            for vid_idx, video_path in enumerate(video_paths):
                if not self.is_running:
                    self._log("批量处理已取消")
                    break

                video_name = Path(video_path).stem
                self._set_batch_label(
                    f"正在处理: {vid_idx + 1}/{total_videos} - {video_name}")
                self._log("=" * 60)
                self._log(
                    f"📁 开始处理第 {vid_idx + 1}/{total_videos} "
                    f"个视频: {video_name}")
                self._log("=" * 60)

                temp_dir = os.path.join(output_dir, "temp", video_name)
                os.makedirs(temp_dir, exist_ok=True)

                try:
                    self._process_single_video(
                        video_path=video_path,
                        video_name=video_name,
                        temp_dir=temp_dir,
                        subtitle_dir=subtitle_dir,
                        api_url=api_url,
                        api_key=api_key,
                        model=model,
                        recognizer_holder=recognizer_holder,
                        vid_idx=vid_idx,
                        total_videos=total_videos,
                    )
                    success_count += 1
                except Exception as e:
                    self._log(f"❌ 视频处理失败: {video_name} - {e}")
                    self._log(tb_module.format_exc())
                    fail_count += 1
                    continue

        except Exception as e:
            self._log(f"❌ 批量处理异常: {e}")
            self._log(tb_module.format_exc())

        finally:
            # ========================================================
            # [v2.1] 安全释放资源 —— 每步 try/except，绝不让异常传播
            # ========================================================
            if recognizer_holder[0] is not None:
                try:
                    recognizer_holder[0].unload_model(self._log)
                except Exception as e:
                    self._log(f"模型卸载警告: {e}")
                finally:
                    recognizer_holder[0] = None

            # 给 CUDA 一点时间完成异步释放
            time.sleep(0.5)

            try:
                gc.collect()
            except Exception:
                pass

            # ========================================================
            # [v2.1] 核心修复：用 **一个** root.after 把所有 UI 操作
            #        打包到主线程执行，避免跨线程竞争和模态弹窗阻塞
            # ========================================================
            self.root.after(200, lambda: self._on_batch_finished(
                success_count, fail_count, subtitle_dir))

    def _on_batch_finished(self, success_count, fail_count, subtitle_dir):
        """
        [v2.1] 在主线程上安全执行所有批处理完成后的 UI 操作
        这是修复卡死问题的关键 —— 所有 UI 操作在同一个主线程回调中完成
        """
        self.is_running = False
        self.progress_var.set(100)
        self.batch_label_var.set("处理完成")
        self.run_btn.configure(state=tk.NORMAL)
        self.stop_btn.configure(state=tk.DISABLED)
        self.clean_btn.configure(state=tk.NORMAL)

        summary = (
            f"✅ 批量处理完成！\n"
            f"成功: {success_count} 个\n"
            f"失败: {fail_count} 个\n"
            f"字幕保存在: {subtitle_dir}"
        )

        # 直接写日志（已在主线程，不走队列）
        self._write_log("=" * 60)
        self._write_log(summary)
        self._write_log("=" * 60)

        # messagebox 放最后，因为它是模态阻塞的
        messagebox.showinfo("完成", summary)

    def _process_single_video(self, video_path, video_name, temp_dir,
                               subtitle_dir, api_url, api_key, model,
                               recognizer_holder, vid_idx, total_videos):
        """处理单个视频的完整流程"""
        stop_flag = lambda: self.is_running

        base_progress = (vid_idx / total_videos) * 100
        video_weight = 100.0 / total_videos

        def vid_progress(pct):
            self._set_progress(base_progress + pct * video_weight / 100)

        # ---- 阶段1: 提取音频 ----
        self._log("阶段 1/4: 提取音频")
        vid_progress(5)

        audio_proc = AudioProcessor(
            segment_duration=self.seg_dur_var.get())
        audio_path = audio_proc.extract_audio(
            video_path, temp_dir, self._log, stop_flag)

        if not self.is_running:
            return

        # ---- 阶段2: 音频分段 ----
        self._log("阶段 2/4: 音频分段")
        vid_progress(15)

        segments = audio_proc.split_audio(
            audio_path, temp_dir, self._log, stop_flag)

        if not self.is_running:
            return

        # ---- 阶段3: 语音识别 ----
        self._log("阶段 3/4: 语音识别（Whisper）")
        vid_progress(25)

        # [v2.1] 从配置读取模型路径
        recognizer = recognizer_holder[0]
        if recognizer is None:
            whisper_path = self.config.get("whisper_model_path", "")
            recognizer = SpeechRecognizer(
                model_path=whisper_path,
                model_size=self.whisper_size_var.get(),
                device=self.whisper_device_var.get(),
                compute_type=self.whisper_compute_var.get(),
            )
            recognizer.load_model(self._log)
            recognizer_holder[0] = recognizer

        entries = recognizer.transcribe_all(
            segments, self._log, stop_flag)
        vid_progress(60)

        if not self.is_running:
            return

        # 保存日语原文字幕
        ja_srt_path = os.path.join(subtitle_dir, f"{video_name}_ja.srt")
        ja_entries_for_srt = [
            {"start": e["start"], "end": e["end"],
             "original": e["text"], "translated": e["text"]}
            for e in entries
        ]
        SubtitleWriter.write_srt(
            ja_entries_for_srt, ja_srt_path, include_original=False)
        self._log(f"日语原文字幕已保存: {ja_srt_path}")

        # ---- 阶段4: AI翻译 ----
        self._log("阶段 4/4: AI翻译")
        vid_progress(65)

        translator = Translator(api_url, api_key, model)
        translated = translator.translate_batch(
            entries,
            max_workers=self.workers_var.get(),
            progress_callback=self._log,
            stop_flag=stop_flag,
        )
        vid_progress(90)

        if not self.is_running:
            return

        # ---- 输出字幕 ----
        fmt = self.format_var.get()
        bilingual = self.bilingual_var.get()

        if fmt == "SRT":
            out_path = os.path.join(
                subtitle_dir, f"{video_name}_translated.srt")
            SubtitleWriter.write_srt(translated, out_path, bilingual)
        else:
            out_path = os.path.join(
                subtitle_dir, f"{video_name}_translated.ass")
            SubtitleWriter.write_ass(translated, out_path, bilingual)

        vid_progress(100)
        self._log(f"✅ {video_name} 处理完成！字幕: {out_path}")

    def run(self):
        self.root.mainloop()


# ============================================================
#  入口
# ============================================================
if __name__ == "__main__":
    app = MainApp()
    app.run()
