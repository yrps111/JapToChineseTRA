# 🎬 Japanese Video Subtitle Tool / 日语视频自动翻译字幕工具

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-Windows-lightgrey.svg)]()

一款基于 **Faster-Whisper** 语音识别 + **AI大模型翻译** 的日语视频字幕生成工具。

输入日语视频，自动输出中日双语 SRT/ASS 字幕文件。支持批量处理，适合挂机批量生成。

---

## ✨ 功能特性

- 🎙️ **日语语音识别** — 基于 Faster-Whisper (CTranslate2)，支持 GPU 加速
- 🌐 **AI 智能翻译** — 兼容任意 OpenAI API 格式的大模型（ChatGPT / Claude / 本地模型等）
- 📁 **批量处理** — 选择多个视频文件，一键挂机，逐个自动处理
- 📝 **双语字幕** — 支持 SRT / ASS 格式，可选中日双语或纯翻译
- 🗑️ **一键清理** — 处理完成后一键删除临时音频文件，只保留字幕
- ⚙️ **GUI 配置** — 所有外部程序路径、API设置均可通过图形界面配置
- 💾 **配置持久化** — 所有设置自动保存，下次打开无需重新配置

## 📸 界面预览

![image](https://files.catbox.moe/3aemnr.png)

## 🔧 环境要求

| 项目        | 要求                   |
| ----------- | ---------------------- |
| 操作系统    | Windows 10/11          |
| Python      | 3.8+                   |
| GPU（推荐） | NVIDIA 显卡，6GB+ VRAM |
| CUDA        | 12.x（使用GPU时需要）  |
| FFmpeg      | 必须安装               |

## 📦 安装步骤
### 1. 克隆仓库

```bash
git clone https://github.com/yrps111/JapToChineseTRA.git
cd JapToChineseTRA
```
### 2. 安装 Python 依赖

```bash
pip install -r requirements.txt
```
### 3. 安装外部程序
#### FFmpeg
- 下载地址: https://ffmpeg.org/download.html
- 解压后记下 `bin` 目录路径
#### CUDA + cuDNN（使用 GPU 时需要）
- CUDA Toolkit: https://developer.nvidia.com/cuda-toolkit
- cuDNN: https://developer.nvidia.com/cudnn
#### Faster-Whisper 模型
- 推荐使用 `large-v3` 模型
- 下载地址: https://huggingface.co/Systran/faster-whisper-large-v3
- 下载整个模型文件夹到本地
### 4. 运行程序

```bash
python main.py
```
### 5. 首次配置

1. 进入 **「路径设置」** 选项卡，设置以下路径：
   - Whisper 模型目录（包含 model.bin 的文件夹）
   - CUDA bin 目录
   - cuDNN bin 目录
   - FFmpeg bin 目录
2. 进入 **「API 设置」** 选项卡，填写翻译 API 信息
3. 点击「保存」即可，下次打开会自动加载

## 📂 输出目录结构

```txt
输出目录/
├── subtitles/                ← 所有字幕文件
│   ├── video1_ja.srt         ← 日语原文
│   ├── video1_translated.srt ← 翻译字幕
│   └── ...
└── temp/                     ← 临时文件（可一键清理）
    └── video1/
        ├── full_audio.wav
        └── segments/
```

## ⚙️ 推荐配置

| 显卡            | 模型大小 | 精度       | 分段时长 |
| --------------- | -------- | ---------- | -------- |
| RTX 3060 (6GB)  | large-v3 | float16    | 30s      |
| RTX 3060 (12GB) | large-v3 | float16    | 30s      |
| GTX 1660 (6GB)  | medium   | int8       | 30s      |
| 无显卡          | small    | int8 (CPU) | 20s      |

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 License

[MIT License](LICENSE)

## 🙏 致谢

- [Faster-Whisper](https://github.com/SYSTRAN/faster-whisper) — 高性能语音识别
- [OpenAI API](https://platform.openai.com/) — 翻译接口标准
- [FFmpeg](https://ffmpeg.org/) — 音视频处理
- [pydub](https://github.com/jiaaro/pydub) — 音频处理

