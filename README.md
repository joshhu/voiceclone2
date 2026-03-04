# Voice Clone 語音克隆系統

基於 [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) Base 模型的語音克隆系統，支援從 YouTube 截取片段或直接上傳音訊，搭配 Whisper 自動轉錄，一鍵克隆任意聲音。

## 功能特色

- **YouTube 音訊提取**：輸入 YouTube 網址，指定開始／結束時間（HH:MM:SS），自動下載指定片段作為聲音來源
- **自動語音轉文字**：上傳或提取音訊後，由 Whisper `large-v3-turbo` 自動辨識參考文字，無需手動輸入
- **雙模型選擇**：0.6B（快速）與 1.7B（高品質）兩個 Base 模型，啟動時同時載入至 GPU
- **兩種克隆模式**：
  - **標準模式**：參考音訊 + 參考文字，品質最高
  - **x-vector 模式**：僅需參考音訊，無需文字，更便捷

## 系統需求

| 項目 | 需求 |
|------|------|
| Python | 3.12+ |
| GPU | NVIDIA CUDA（建議 ≥ 12GB VRAM） |
| 套件管理 | [uv](https://github.com/astral-sh/uv) |
| 系統工具 | `ffmpeg`（音訊轉換必要） |

### 安裝 ffmpeg

```bash
# Ubuntu / Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

## 安裝

```bash
# 1. clone 專案
git clone https://github.com/joshhu/voiceclone2.git
cd voiceclone2

# 2. 安裝依賴（使用 uv）
uv sync

# 3. 啟動
uv run python app.py
```

啟動後開啟瀏覽器至 `http://localhost:7860`。

> 首次啟動會從 HuggingFace 下載模型（Qwen3-TTS 0.6B + 1.7B + Whisper large-v3-turbo），約需數分鐘。

## 使用方式

### 方法一：從 YouTube 提取聲音

1. 在左欄「YouTube 網址」輸入影片連結
2. 點「載入影片預覽」確認影片內容
3. 輸入「開始時間」與「結束時間」（格式：`0:01:30`）
4. 點「提取音訊片段」→ 自動下載並填入參考音訊與參考文字
5. 在右欄輸入「目標文字」（要讓克隆聲音說的話）
6. 點「開始克隆」

### 方法二：直接上傳音訊

1. 在左欄「參考音訊」上傳音訊檔（WAV / MP3 等）
2. Whisper 自動辨識並填入「參考文字」
3. 在右欄輸入「目標文字」
4. 點「開始克隆」

## 架構說明

整個系統集中在單一 `app.py`，分三個主要區塊：

```
啟動時載入模型
├── Qwen3-TTS Base 0.6B  (bfloat16, CUDA)
├── Qwen3-TTS Base 1.7B  (bfloat16, CUDA)
└── Whisper large-v3-turbo (int8, CUDA)

音訊工具函式
├── _normalize_audio()      — 正規化至 float32 [-1, 1]
├── _audio_to_tuple()       — 統一 Gradio 音訊輸入格式
├── transcribe_audio()      — Whisper 語音轉文字
├── extract_video_id()      — 解析 YouTube video_id
├── parse_time()            — HH:MM:SS → 秒數
├── load_youtube_preview()  — 產生 iframe embed HTML
└── extract_youtube_audio() — yt-dlp 下載指定片段

Gradio UI (build_ui())
├── 左欄：YouTube 提取 + 參考音訊輸入
└── 右欄：目標文字 + 模型設定 + 克隆按鈕
```

## 依賴套件

| 套件 | 用途 |
|------|------|
| `qwen-tts` | Qwen3-TTS 模型推論 |
| `gradio` | Web UI |
| `faster-whisper` | 語音辨識（CTranslate2 後端） |
| `yt-dlp` | YouTube 音訊下載 |
| `soundfile` | WAV 檔讀寫 |
| `librosa` | 音訊重採樣 |
| `numpy` | 數值運算 |

## 注意事項

- **VRAM 使用量**：三個模型合計約佔 6-7GB VRAM（RTX 3080 Ti 12GB 測試通過）
- **flash-attn**：若系統安裝 `flash-attn`，自動啟用 `flash_attention_2`；否則退回 `sdpa`
- **YouTube 下載**：僅供個人研究使用，請遵守 YouTube 服務條款及著作權規定
- **輸出檔案**：克隆音訊存至 `outputs/` 目錄（已加入 `.gitignore`）

## 授權

本專案遵循 [Qwen3-TTS 模型授權](https://github.com/QwenLM/Qwen3-TTS)。
