# YouTube 音訊片段提取功能設計文件

**日期：** 2026-03-04
**狀態：** 已核准

## 目標

在 Gradio 介面中讓使用者輸入 YouTube 網址，預覽影片，指定開始／結束時間（HH:MM:SS），自動提取該段音訊作為語音克隆的參考來源。

## 架構

```
YouTube URL → gr.HTML iframe 顯示影片
             ↓
開始/結束時間（HH:MM:SS）→ yt-dlp download_range_func → 只下載指定片段（WAV）
                                                          ↓
                                              自動填入 ref_audio
                                                          ↓
                                              自動觸發 Whisper 轉錄 → ref_text
```

## 技術選擇

| 功能 | 方案 | 理由 |
|------|------|------|
| YouTube 影片預覽 | `gr.HTML` + YouTube embed iframe | Gradio 無原生 YT 元件；gr.HTML 支援 iframe |
| 音訊提取 | `yt-dlp` Python API + `download_range_func` | 只下載指定片段，速度快；比 pytube 穩定 |
| 格式轉換 | ffmpeg（yt-dlp postprocessor） | 系統已存在；轉 WAV 方便後續處理 |
| 時間格式 | HH:MM:SS 文字輸入 | 直覺，支援長影片 |

## UI 佈局變更

左欄在原有「參考音訊」上方，新增：

1. **YouTube URL 輸入框** + **載入影片** 按鈕
2. **YouTube iframe 播放器**（點載入後顯示）
3. **開始時間** / **結束時間** 兩欄並排文字輸入
4. **提取音訊片段** 按鈕

提取成功後自動填入原有的 `ref_audio` 與 `ref_text`。

## 關鍵實作細節

### video_id 擷取

```python
import re
from urllib.parse import urlparse, parse_qs

def extract_video_id(url: str) -> str | None:
    patterns = [
        r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    return None
```

### yt-dlp 片段下載

```python
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func

def extract_youtube_segment(url, start_sec, end_sec, output_path):
    ydl_opts = {
        'format': 'bestaudio/best',
        'download_ranges': download_range_func(None, [(start_sec, end_sec)]),
        'force_keyframes_at_cuts': True,
        'outtmpl': output_path,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
```

### HH:MM:SS 轉換

```python
def parse_time(t: str) -> float:
    parts = t.strip().split(":")
    parts = [float(p) for p in parts]
    if len(parts) == 3:
        return parts[0]*3600 + parts[1]*60 + parts[2]
    elif len(parts) == 2:
        return parts[0]*60 + parts[1]
    return float(parts[0])
```

## 錯誤處理

- 無效 URL → 顯示錯誤訊息
- 無效時間格式 → 顯示錯誤訊息
- 開始時間 >= 結束時間 → 顯示錯誤訊息
- yt-dlp 下載失敗（版權、私人影片等）→ 顯示錯誤訊息

## 依賴變更

- 新增：`yt-dlp`（Python 套件）
- ffmpeg 已存在
