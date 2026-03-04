# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 環境設置

此專案使用 **uv** 管理 Python 環境（Python 3.12），**禁止**使用 base pip 或 conda。

```bash
# 安裝依賴
uv sync

# 啟動應用程式
uv run python app.py
```

應用程式啟動後會在 `http://0.0.0.0:7860` 開放 Gradio UI。

## 架構概述

**單檔架構**：整個系統集中在 `app.py`，包含三個主要區塊：

1. **模型載入**（啟動時執行）：在程式啟動時同時將兩個 Base 模型（0.6B、1.7B）載入至 GPU，儲存在全域字典 `BASE_MODELS`。這是刻意設計，推論時零等待但啟動慢且佔 GPU 記憶體。

2. **音訊工具函式**：`_normalize_audio()` 將任意格式音訊正規化為 float32 [-1, 1]；`_audio_to_tuple()` 處理 Gradio 的不同音訊輸入格式（tuple 或 dict）。

3. **Gradio UI**：`build_ui()` 建立雙欄介面，左欄為參考音訊輸入，右欄為目標文字與參數設定。

## 模型資訊

- 使用 `Qwen/Qwen3-TTS-12Hz-{0.6B|1.7B}-Base`，透過 HuggingFace `snapshot_download` 下載。
- 需要 CUDA GPU；自動偵測 `flash-attn` 是否安裝，有則使用 `flash_attention_2`，否則退回 `sdpa`。
- 模型以 `bfloat16` 精度載入。

## 語音克隆模式

- **標準模式**：需要參考音訊 + 參考文字（音訊中說的內容），品質較高。
- **x-vector 模式**（`use_xvector_only=True`）：只需參考音訊，無需文字，品質較低但更方便。

## 輸出

合成的語音輸出存放在 `outputs/` 目錄（WAV 格式），此目錄已加入 `.gitignore`。
