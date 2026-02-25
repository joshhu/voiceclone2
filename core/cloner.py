"""語音克隆邏輯 — 分析參考音檔 + 合成語音"""

from pathlib import Path

import numpy as np
import soundfile as sf

from core.model import get_model

# 合成語音輸出目錄
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)


def analyze_voice(audio_path: str, ref_text: str) -> dict:
    """分析參考音檔，建立可重複使用的 voice clone prompt

    Args:
        audio_path: 參考音檔路徑
        ref_text: 參考音檔對應的文字內容

    Returns:
        voice clone prompt 物件（用於後續合成）
    """
    model = get_model()
    prompt = model.create_voice_clone_prompt(
        ref_audio=audio_path,
        ref_text=ref_text,
    )
    return prompt


def synthesize(
    text: str,
    voice_clone_prompt: dict,
    language: str = "Chinese",
) -> tuple[np.ndarray, int, str]:
    """用克隆的聲音合成語音

    Args:
        text: 要合成的文字
        voice_clone_prompt: analyze_voice 回傳的 prompt 物件
        language: 語言（預設中文）

    Returns:
        (音訊 numpy array, sample rate, 輸出檔案路徑)
    """
    model = get_model()
    wavs, sr = model.generate_voice_clone(
        text=text,
        language=language,
        voice_clone_prompt=voice_clone_prompt,
    )

    # 儲存到 outputs 目錄
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = OUTPUT_DIR / f"clone_{timestamp}.wav"
    sf.write(str(output_path), wavs[0], sr)

    return wavs[0], sr, str(output_path)
