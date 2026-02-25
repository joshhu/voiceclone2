"""Gradio Web UI — 語音克隆系統主介面"""

import gradio as gr

from core.audio_utils import (
    format_audio_info,
    get_audio_info,
    plot_spectrogram,
    plot_waveform,
    preprocess_audio,
)
from core.cloner import analyze_voice, synthesize
from core.model import get_model_name, load_model

# 支援的語言選項
LANGUAGES = ["Chinese", "English", "Japanese", "Korean", "French", "German"]


def on_analyze(audio_path: str, ref_text: str):
    """處理「分析聲音」按鈕事件"""
    if audio_path is None:
        gr.Warning("請先上傳參考音檔")
        return None, None, "", None, "❌ 請先上傳參考音檔"

    if not ref_text.strip():
        gr.Warning("請輸入參考音檔的文字內容")
        return None, None, "", None, "❌ 請輸入參考音檔的文字內容"

    # 預處理：將 AAC/MP3 等格式轉為 WAV
    wav_path = preprocess_audio(audio_path)

    # 載入模型（首次會觸發下載）
    load_model()

    # 音訊分析
    info = get_audio_info(wav_path)
    info_text = format_audio_info(info)
    waveform_fig = plot_waveform(wav_path)
    spec_fig = plot_spectrogram(wav_path)

    # 建立 voice clone prompt
    prompt = analyze_voice(wav_path, ref_text)

    status = f"✅ 聲紋特徵已擷取（模型：{get_model_name()}）"
    return waveform_fig, spec_fig, info_text, prompt, status


def on_synthesize(text: str, language: str, voice_prompt):
    """處理「生成語音」按鈕事件"""
    if voice_prompt is None:
        gr.Warning("請先完成聲音分析")
        return None, "❌ 請先上傳參考音檔並完成聲音分析"

    if not text.strip():
        gr.Warning("請輸入要合成的文字")
        return None, "❌ 請輸入要合成的文字"

    audio_array, sr, output_path = synthesize(text, voice_prompt, language)

    status = f"✅ 語音合成完成：{output_path}"
    return output_path, status


def build_ui() -> gr.Blocks:
    """建立 Gradio 介面"""
    with gr.Blocks(title="Voice Clone — Qwen3-TTS 語音克隆") as app:
        gr.Markdown("# 🎙️ Voice Clone — Qwen3-TTS 語音克隆系統")
        gr.Markdown("上傳一段參考語音，系統分析聲音特徵後，即可用克隆的聲音合成任意文字。")

        # 跨區塊共享的 voice clone prompt
        voice_prompt_state = gr.State(value=None)

        # ── 區塊一：聲音分析 ──
        with gr.Group():
            gr.Markdown("## 📎 第一步：聲音分析")

            with gr.Row():
                with gr.Column(scale=1):
                    ref_audio = gr.Audio(
                        label="上傳參考音檔（wav / mp3 / flac）",
                        type="filepath",
                    )
                    ref_text = gr.Textbox(
                        label="參考音檔的文字內容",
                        placeholder="請輸入參考音檔中說的話...",
                        lines=3,
                    )
                    analyze_btn = gr.Button("🔍 分析聲音", variant="primary")

                with gr.Column(scale=1):
                    audio_info = gr.Textbox(label="音訊資訊", interactive=False, lines=4)
                    analyze_status = gr.Textbox(label="分析狀態", interactive=False)

            with gr.Row():
                waveform_plot = gr.Plot(label="波形圖")
                spectrogram_plot = gr.Plot(label="梅爾頻譜圖")

        # ── 區塊二：語音合成 ──
        with gr.Group():
            gr.Markdown("## 🗣️ 第二步：語音合成")

            with gr.Row():
                with gr.Column(scale=2):
                    synth_text = gr.Textbox(
                        label="要合成的文字",
                        placeholder="輸入任意中文文字...",
                        lines=4,
                    )
                with gr.Column(scale=1):
                    language = gr.Dropdown(
                        choices=LANGUAGES,
                        value="Chinese",
                        label="語言",
                    )
                    synth_btn = gr.Button("🎵 生成語音", variant="primary")

            synth_status = gr.Textbox(label="合成狀態", interactive=False)
            synth_audio = gr.Audio(label="合成結果", type="filepath")

        # ── 事件綁定 ──
        analyze_btn.click(
            fn=on_analyze,
            inputs=[ref_audio, ref_text],
            outputs=[waveform_plot, spectrogram_plot, audio_info, voice_prompt_state, analyze_status],
        )

        synth_btn.click(
            fn=on_synthesize,
            inputs=[synth_text, language, voice_prompt_state],
            outputs=[synth_audio, synth_status],
        )

    return app


if __name__ == "__main__":
    app = build_ui()
    app.launch(server_name="0.0.0.0", server_port=7860, theme=gr.themes.Soft())
