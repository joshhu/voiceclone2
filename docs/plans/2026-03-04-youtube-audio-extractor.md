# YouTube 音訊片段提取功能實作計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 在 Gradio UI 中讓使用者輸入 YouTube URL、預覽影片、指定 HH:MM:SS 時間範圍，自動提取音訊填入 ref_audio 並觸發 Whisper 轉錄。

**Architecture:** 以 yt-dlp Python API 的 `download_range_func` 只下載指定片段（轉 WAV），`gr.HTML` iframe 嵌入 YouTube 播放器預覽，提取後串接現有的 `transcribe_audio()` 自動轉錄。

**Tech Stack:** `yt-dlp`、`soundfile`（已有）、`gr.HTML` iframe、stdlib `os/re/tempfile/glob`

---

### Task 1：安裝 yt-dlp

**Files:**
- Modify: `pyproject.toml`（由 uv 自動管理）

**Step 1: 安裝套件**

```bash
cd /home/joshhu/workspace/voiceclone2
uv add yt-dlp
```

**Step 2: 驗證可 import**

```bash
uv run python -c "from yt_dlp import YoutubeDL; from yt_dlp.utils import download_range_func; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: 新增 yt-dlp 依賴"
```

---

### Task 2：新增工具函式

**Files:**
- Modify: `app.py`（在 `transcribe_audio()` 之後、`generate_voice_clone()` 之前插入）

**Step 1: 在 app.py 頂部 import 區塊（第 3 行之後）加入以下 stdlib import**

```python
import os
import re
import glob
import tempfile
```

同時加入 yt-dlp import（緊接在 `from faster_whisper import WhisperModel` 之後）：

```python
import soundfile as sf
from yt_dlp import YoutubeDL
from yt_dlp.utils import download_range_func
```

**Step 2: 在 `transcribe_audio()` 結束後（第 115 行之後）插入四個工具函式**

```python
def extract_video_id(url: str) -> str | None:
    """從 YouTube URL 中提取 11 位 video_id。"""
    m = re.search(r"(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})", url)
    return m.group(1) if m else None


def parse_time(t: str) -> float:
    """將 HH:MM:SS、MM:SS 或純秒數字串轉換為浮點秒數。"""
    parts = [float(p) for p in t.strip().split(":")]
    if len(parts) == 3:
        return parts[0] * 3600 + parts[1] * 60 + parts[2]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0]


def load_youtube_preview(url: str) -> str:
    """回傳 YouTube embed iframe HTML；URL 無效時回傳錯誤提示。"""
    vid = extract_video_id(url.strip() if url else "")
    if not vid:
        return '<p style="color:red">⚠ 無效的 YouTube 網址</p>'
    return (
        f'<iframe width="100%" height="280" '
        f'src="https://www.youtube.com/embed/{vid}" '
        f'frameborder="0" allowfullscreen></iframe>'
    )


def extract_youtube_audio(url: str, start_str: str, end_str: str):
    """
    以 yt-dlp 下載 YouTube 指定時間片段並轉為 WAV。
    回傳 (audio_tuple, status_msg)。
    audio_tuple 為 (sample_rate, numpy_float32_array) 或 None（失敗時）。
    """
    if not url or not url.strip():
        return None, "錯誤：請輸入 YouTube 網址。"
    if not extract_video_id(url):
        return None, "錯誤：無效的 YouTube 網址。"
    try:
        start_sec = parse_time(start_str)
        end_sec = parse_time(end_str)
    except Exception:
        return None, "錯誤：時間格式不正確，請使用 HH:MM:SS 或 MM:SS。"
    if start_sec >= end_sec:
        return None, "錯誤：開始時間必須小於結束時間。"

    tmp_dir = tempfile.mkdtemp()
    out_template = os.path.join(tmp_dir, "segment")
    ydl_opts = {
        'format': 'bestaudio/best',
        'download_ranges': download_range_func(None, [(start_sec, end_sec)]),
        'force_keyframes_at_cuts': True,
        'outtmpl': out_template,
        'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'wav'}],
        'quiet': True,
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
    except Exception as e:
        return None, f"錯誤：下載失敗 — {e}"

    # yt-dlp 可能在檔名加上時間段後綴，用 glob 尋找
    candidates = glob.glob(os.path.join(tmp_dir, "*.wav"))
    if not candidates:
        return None, "錯誤：音訊提取失敗，找不到輸出 WAV 檔。"
    wav_path = candidates[0]

    wav_data, sr = sf.read(wav_path)
    wav_data = wav_data.astype(np.float32)
    if wav_data.ndim > 1:
        wav_data = np.mean(wav_data, axis=-1)
    duration = len(wav_data) / sr
    return (sr, wav_data), f"✅ 提取完成！{start_str} → {end_str}（{duration:.1f} 秒）"
```

**Step 3: 驗證語法**

```bash
cd /home/joshhu/workspace/voiceclone2
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text()); print('語法 OK')"
```

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: 新增 YouTube 音訊提取工具函式"
```

---

### Task 3：更新 Gradio UI（左欄新增 YouTube 區塊）

**Files:**
- Modify: `app.py`（`build_ui()` 函式，左欄 Column 內）

**Step 1: 在左欄 Column 的最頂部（`ref_audio = gr.Audio(...)` 之前）插入 YouTube UI 區塊**

找到這一行：
```python
            # ---- 左欄：參考音訊輸入 ----
            with gr.Column(scale=1):
                ref_audio = gr.Audio(
```

在 `ref_audio = gr.Audio(` 之前插入：

```python
                gr.Markdown("### 從 YouTube 提取音訊（選用）")
                yt_url = gr.Textbox(
                    label="YouTube 網址",
                    placeholder="https://www.youtube.com/watch?v=...",
                    lines=1,
                )
                yt_load_btn = gr.Button("載入影片預覽", size="sm")
                yt_preview = gr.HTML(value="")
                with gr.Row():
                    yt_start = gr.Textbox(label="開始時間", value="0:00:00", lines=1, scale=1)
                    yt_end = gr.Textbox(label="結束時間", value="0:00:30", lines=1, scale=1)
                yt_extract_btn = gr.Button("提取音訊片段", variant="secondary")
                yt_status = gr.Textbox(label="提取狀態", lines=1, interactive=False)
                gr.Markdown("---")
```

**Step 2: 驗證語法**

```bash
cd /home/joshhu/workspace/voiceclone2
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text()); print('語法 OK')"
```

**Step 3: Commit**

```bash
git add app.py
git commit -m "feat: 左欄新增 YouTube UI 區塊"
```

---

### Task 4：串接事件處理器

**Files:**
- Modify: `app.py`（`build_ui()` 函式，`ref_audio.change(...)` 之後）

**Step 1: 在 `ref_audio.change(...)` 區塊之後插入事件綁定**

找到：
```python
        ref_audio.change(
            transcribe_audio,
            inputs=[ref_audio],
            outputs=[ref_text],
        )
```

在其後插入：

```python
        yt_load_btn.click(
            load_youtube_preview,
            inputs=[yt_url],
            outputs=[yt_preview],
        )

        def _extract_and_transcribe(url, start, end):
            audio, msg = extract_youtube_audio(url, start, end)
            if audio is None:
                return None, "", msg
            return audio, transcribe_audio(audio), msg

        yt_extract_btn.click(
            _extract_and_transcribe,
            inputs=[yt_url, yt_start, yt_end],
            outputs=[ref_audio, ref_text, yt_status],
        )
```

**Step 2: 驗證語法**

```bash
cd /home/joshhu/workspace/voiceclone2
uv run python -c "import ast, pathlib; ast.parse(pathlib.Path('app.py').read_text()); print('語法 OK')"
```

**Step 3: 確認所有 event handler 都在**

```bash
grep -n "yt_load_btn\|yt_extract_btn\|ref_audio.change" /home/joshhu/workspace/voiceclone2/app.py
```

**Step 4: Commit**

```bash
git add app.py
git commit -m "feat: 串接 YouTube 載入與提取音訊事件"
```

---

### Task 5：最終 push

```bash
cd /home/joshhu/workspace/voiceclone2
git push origin features/version2
```
