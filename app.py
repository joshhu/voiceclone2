# coding=utf-8
# Voice Clone 語音克隆系統 — 基於 Qwen3-TTS Base 模型
import numpy as np
import torch
import gradio as gr
from huggingface_hub import snapshot_download
from qwen_tts import Qwen3TTSModel
import librosa
from faster_whisper import WhisperModel

# 偵測 flash-attn 是否可用
try:
    import flash_attn  # noqa: F401
    _ATTN_IMPL = "flash_attention_2"
except ImportError:
    _ATTN_IMPL = "sdpa"

# 支援的語言
LANGUAGES = [
    "Auto", "Chinese", "English", "Japanese", "Korean",
    "French", "German", "Spanish", "Portuguese", "Russian",
]
MODEL_SIZES = ["0.6B", "1.7B"]


def get_model_path(model_size: str) -> str:
    """根據模型大小取得 HuggingFace 模型路徑。"""
    return snapshot_download(f"Qwen/Qwen3-TTS-12Hz-{model_size}-Base")


# ============================================================================
# 啟動時載入模型
# ============================================================================
print("正在載入 Voice Clone 模型...")

print("  載入 Base 0.6B ...")
base_model_0_6b = Qwen3TTSModel.from_pretrained(
    get_model_path("0.6B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation=_ATTN_IMPL,
)

print("  載入 Base 1.7B ...")
base_model_1_7b = Qwen3TTSModel.from_pretrained(
    get_model_path("1.7B"),
    device_map="cuda",
    dtype=torch.bfloat16,
    attn_implementation=_ATTN_IMPL,
)

BASE_MODELS = {"0.6B": base_model_0_6b, "1.7B": base_model_1_7b}
print("  載入 Whisper large-v3-turbo ...")
WHISPER_MODEL = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="int8",
)
print("Whisper 載入完成！")
print("模型載入完成！")


# ============================================================================
# 音訊工具函式
# ============================================================================
def _normalize_audio(wav, eps=1e-12):
    """正規化音訊至 float32 [-1, 1] 範圍。"""
    x = np.asarray(wav)

    if np.issubdtype(x.dtype, np.integer):
        info = np.iinfo(x.dtype)
        if info.min < 0:
            y = x.astype(np.float32) / max(abs(info.min), info.max)
        else:
            mid = (info.max + 1) / 2.0
            y = (x.astype(np.float32) - mid) / mid
    elif np.issubdtype(x.dtype, np.floating):
        y = x.astype(np.float32)
        m = np.max(np.abs(y)) if y.size else 0.0
        if m > 1.0 + 1e-6:
            y = y / (m + eps)
    else:
        raise TypeError(f"不支援的 dtype: {x.dtype}")

    y = np.clip(y, -1.0, 1.0)
    if y.ndim > 1:
        y = np.mean(y, axis=-1).astype(np.float32)
    return y


def _audio_to_tuple(audio):
    """將 Gradio 音訊輸入轉換為 (wav, sr) tuple。"""
    if audio is None:
        return None
    # Gradio 預設格式：(sample_rate, numpy_array)
    if isinstance(audio, tuple) and len(audio) == 2 and isinstance(audio[0], int):
        sr, wav = audio
        return _normalize_audio(wav), int(sr)
    # dict 格式
    if isinstance(audio, dict) and "sampling_rate" in audio and "data" in audio:
        return _normalize_audio(audio["data"]), int(audio["sampling_rate"])
    return None


def transcribe_audio(audio):
    """將參考音訊自動轉錄為文字。"""
    audio_tuple = _audio_to_tuple(audio)
    if audio_tuple is None:
        return ""
    wav, sr = audio_tuple
    # Whisper 需要 float32、mono、16kHz
    if sr != 16000:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=16000)
    segments, _ = WHISPER_MODEL.transcribe(wav, beam_size=5)
    return "".join(seg.text for seg in segments).strip()


# ============================================================================
# 語音克隆核心
# ============================================================================
def generate_voice_clone(
    ref_audio, ref_text, target_text, language, use_xvector_only, model_size,
    progress=gr.Progress(track_tqdm=True),
):
    """使用 Base 模型進行語音克隆。"""
    if not target_text or not target_text.strip():
        return None, "錯誤：請輸入目標文字。"

    audio_tuple = _audio_to_tuple(ref_audio)
    if audio_tuple is None:
        return None, "錯誤：請上傳參考音訊。"

    if not use_xvector_only and (not ref_text or not ref_text.strip()):
        return None, "錯誤：未啟用「僅使用 x-vector」時，需要提供參考文字。"

    try:
        tts = BASE_MODELS[model_size]
        wavs, sr = tts.generate_voice_clone(
            text=target_text.strip(),
            language=language,
            ref_audio=audio_tuple,
            ref_text=ref_text.strip() if ref_text else None,
            x_vector_only_mode=use_xvector_only,
            max_new_tokens=2048,
        )
        return (sr, wavs[0]), "語音克隆完成！"
    except Exception as e:
        return None, f"錯誤：{type(e).__name__}: {e}"


# ============================================================================
# Gradio UI
# ============================================================================
def build_ui():
    with gr.Blocks(title="Voice Clone 語音克隆") as demo:
        gr.Markdown(
            """
# Voice Clone 語音克隆系統
上傳一段參考音訊，克隆該聲音來朗讀你指定的文字。
基於 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Base 模型。
"""
        )

        with gr.Row():
            # ---- 左欄：參考音訊輸入 ----
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
                    label="參考音訊（上傳要克隆的聲音樣本）",
                    type="numpy",
                )
                ref_text = gr.Textbox(
                    label="參考文字（參考音訊中說的內容）",
                    lines=2,
                    placeholder="輸入參考音訊中的確切文字...",
                )
                xvector_only = gr.Checkbox(
                    label="僅使用 x-vector（不需要參考文字，但品質較低）",
                    value=False,
                )

            # ---- 右欄：目標文字與設定 ----
            with gr.Column(scale=1):
                target_text = gr.Textbox(
                    label="目標文字（要用克隆聲音朗讀的文字）",
                    lines=4,
                    placeholder="輸入你想讓克隆聲音說的文字...",
                )
                with gr.Row():
                    language = gr.Dropdown(
                        label="語言",
                        choices=LANGUAGES,
                        value="Auto",
                    )
                    model_size = gr.Dropdown(
                        label="模型大小",
                        choices=MODEL_SIZES,
                        value="1.7B",
                    )
                clone_btn = gr.Button("開始克隆", variant="primary")

        with gr.Row():
            audio_out = gr.Audio(label="生成的音訊", type="numpy")
            status = gr.Textbox(label="狀態", lines=2, interactive=False)

        clone_btn.click(
            generate_voice_clone,
            inputs=[ref_audio, ref_text, target_text, language, xvector_only, model_size],
            outputs=[audio_out, status],
        )

        ref_audio.change(
            transcribe_audio,
            inputs=[ref_audio],
            outputs=[ref_text],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(
            font=[gr.themes.GoogleFont("Source Sans Pro"), "Arial", "sans-serif"],
        ),
    )
