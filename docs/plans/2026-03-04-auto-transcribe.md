# 自動語音轉文字（STT）實作計劃

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 上傳參考音訊後自動辨識並填入「參考文字」欄位，省去手動輸入。

**Architecture:** 在啟動時同步載入 faster-whisper large-v3-turbo 模型（與 TTS 模型相同策略），音訊上傳後透過 Gradio `.change()` 事件自動觸發轉錄，結果填入 `ref_text` Textbox。

**Tech Stack:** `faster-whisper`（CTranslate2 後端）、`librosa`（重採樣至 16kHz）、Gradio `.change()` 事件

---

### Task 1：安裝 faster-whisper 依賴

**Files:**
- Modify: `pyproject.toml`

**Step 1: 新增套件**

```bash
uv add faster-whisper
```

**Step 2: 確認安裝**

```bash
uv run python -c "from faster_whisper import WhisperModel; print('OK')"
```

Expected: `OK`

**Step 3: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "chore: 新增 faster-whisper 依賴"
```

---

### Task 2：啟動時載入 Whisper 模型

**Files:**
- Modify: `app.py`（在 TTS 模型載入區塊之後）

**Step 1: 新增 import**

在 `app.py` 頂部加入：

```python
import librosa
from faster_whisper import WhisperModel
```

**Step 2: 在 `BASE_MODELS = {...}` 之後新增 Whisper 載入**

```python
print("  載入 Whisper large-v3-turbo ...")
WHISPER_MODEL = WhisperModel(
    "large-v3-turbo",
    device="cuda",
    compute_type="int8",
)
print("Whisper 載入完成！")
```

**Step 3: 驗證**

```bash
uv run python -c "
import app
print('WHISPER_MODEL:', app.WHISPER_MODEL)
"
```

Expected: 印出模型物件，無 CUDA OOM 錯誤。

---

### Task 3：實作 transcribe_audio() 函式

**Files:**
- Modify: `app.py`（在「音訊工具函式」區塊內）

**Step 1: 新增函式**

在 `_audio_to_tuple()` 之後加入：

```python
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
```

**Step 2: 快速測試（選用）**

若有測試音訊：

```bash
uv run python -c "
import app, soundfile as sf
wav, sr = sf.read('test.wav')
import numpy as np
audio = (sr, wav.astype(np.float32))
print(app.transcribe_audio(audio))
"
```

---

### Task 4：串接 Gradio UI

**Files:**
- Modify: `app.py`（`build_ui()` 函式內）

**Step 1: 在 `clone_btn.click(...)` 之後加入 `.change()` 事件**

```python
ref_audio.change(
    transcribe_audio,
    inputs=[ref_audio],
    outputs=[ref_text],
)
```

**Step 2: 驗證 UI**

```bash
uv run python app.py
```

開啟 http://localhost:7860，上傳任意音訊，確認「參考文字」欄位自動填入。

---

### Task 5：提交與推送

```bash
git add app.py
git commit -m "feat: 上傳參考音訊後自動語音轉文字（faster-whisper large-v3-turbo）"
git push origin features/version2
```
