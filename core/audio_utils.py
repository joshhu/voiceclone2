"""音訊工具 — 波形圖、頻譜圖、音訊預處理"""

import tempfile
from pathlib import Path

import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

# 非互動式後端，避免 Gradio 環境下衝突
matplotlib.use("Agg")

# 中文字型支援（matplotlib）
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK TC", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# soundfile 不支援的格式（需要透過 librosa/ffmpeg 轉檔）
_SOUNDFILE_SUPPORTED = {".wav", ".flac", ".ogg", ".oga"}


def preprocess_audio(audio_path: str) -> str:
    """預處理音訊：將不支援的格式（AAC、MP3 等）轉換為 WAV

    如果已經是 WAV 等 soundfile 支援的格式，直接回傳原路徑。
    否則用 librosa（ffmpeg 後端）讀取後轉存為 WAV 臨時檔。
    """
    ext = Path(audio_path).suffix.lower()
    if ext in _SOUNDFILE_SUPPORTED:
        return audio_path

    # 用 librosa 讀取（支援幾乎所有格式，透過 ffmpeg）
    y, sr = librosa.load(audio_path, sr=None, mono=False)

    # 寫出為 WAV 臨時檔
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, y.T if y.ndim > 1 else y, sr)
    return tmp.name


def get_audio_info(audio_path: str) -> dict:
    """取得音訊資訊（時長、取樣率、聲道數）

    使用 librosa 取得基本資訊，避免 soundfile 的格式限制。
    """
    y, sr = librosa.load(audio_path, sr=None, mono=False)
    if y.ndim == 1:
        channels = 1
    else:
        channels = y.shape[0]
    duration = librosa.get_duration(y=y, sr=sr)
    ext = Path(audio_path).suffix.lstrip(".").upper()
    return {
        "duration": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "format": ext,
        "subtype": "PCM_16" if ext == "WAV" else ext,
    }


def plot_waveform(audio_path: str) -> plt.Figure:
    """繪製波形圖，回傳 matplotlib Figure"""
    y, sr = librosa.load(audio_path, sr=None)
    duration = len(y) / sr

    fig, ax = plt.subplots(figsize=(10, 3))
    time_axis = np.linspace(0, duration, len(y))
    ax.plot(time_axis, y, linewidth=0.5, color="#2563eb")
    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("振幅")
    ax.set_title("波形圖")
    ax.set_xlim(0, duration)
    fig.tight_layout()
    return fig


def plot_spectrogram(audio_path: str) -> plt.Figure:
    """繪製梅爾頻譜圖，回傳 matplotlib Figure"""
    y, sr = librosa.load(audio_path, sr=None)

    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_dB = librosa.power_to_db(S, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(S_dB, sr=sr, x_axis="time", y_axis="mel", ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set_title("梅爾頻譜圖")
    ax.set_xlabel("時間 (秒)")
    ax.set_ylabel("頻率 (Hz)")
    fig.tight_layout()
    return fig


def format_audio_info(info: dict) -> str:
    """將音訊資訊格式化為可讀字串"""
    return (
        f"⏱ 時長：{info['duration']} 秒\n"
        f"🎵 取樣率：{info['sample_rate']} Hz\n"
        f"📢 聲道數：{info['channels']}\n"
        f"📁 格式：{info['format']} ({info['subtype']})"
    )
