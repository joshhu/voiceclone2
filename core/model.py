"""模型載入與管理 — 支援 Qwen3-TTS 語音克隆模型"""

import torch
from qwen_tts import Qwen3TTSModel

# 模型單例
_model: Qwen3TTSModel | None = None
_current_model_name: str | None = None

# 模型優先順序：先嘗試 1.7B，VRAM 不足再降級 0.6B
PRIMARY_MODEL = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
FALLBACK_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def _detect_attn_implementation() -> str:
    """偵測是否有 flash-attn，沒有就用 sdpa"""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def load_model(
    model_name: str = PRIMARY_MODEL,
    device: str = "cuda:0",
) -> Qwen3TTSModel:
    """載入 Qwen3-TTS 模型（單例模式）

    如果已載入相同模型，直接回傳。
    如果 VRAM 不足，自動降級到 0.6B 模型。
    """
    global _model, _current_model_name

    if _model is not None and _current_model_name == model_name:
        return _model

    # 釋放舊模型
    if _model is not None:
        del _model
        _model = None
        _current_model_name = None
        torch.cuda.empty_cache()

    attn_impl = _detect_attn_implementation()
    print(f"[模型] 正在載入 {model_name}（attention: {attn_impl}）...")

    try:
        _model = Qwen3TTSModel.from_pretrained(
            model_name,
            device_map=device,
            dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        _current_model_name = model_name
        print(f"[模型] {model_name} 載入完成")
        return _model

    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        if "out of memory" in str(e).lower() and model_name == PRIMARY_MODEL:
            print(f"[模型] VRAM 不足，自動降級到 {FALLBACK_MODEL}")
            torch.cuda.empty_cache()
            return load_model(model_name=FALLBACK_MODEL, device=device)
        raise


def get_model() -> Qwen3TTSModel:
    """取得已載入的模型，若尚未載入則自動載入"""
    if _model is None:
        return load_model()
    return _model


def get_model_name() -> str:
    """取得目前載入的模型名稱"""
    return _current_model_name or "尚未載入"
