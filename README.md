
# AI Meeting Summarizer

An AI-powered meeting summarizer built using Hugging Face Transformers, Whisper, and BART.

This project:
- Converts meeting audio into text
- Generates summaries from the transcript
- Runs completely locally using GPU acceleration

---

# Features

- Speech-to-text transcription using Whisper
- Text summarization using BART
- GPU acceleration with PyTorch CUDA
- Chunking support for long transcripts
- Local inference using open-source models

---

# Models Used

## Transcription
- openai/whisper-tiny.en

## Summarization
- facebook/bart-large-cnn

---

# Project Structure

```text
meeting-summarizer/
│
├── main.py
├── transcriber.py
├── summarizer.py
│
├── audio/
├── output/
└── README.md
```

---

# Installation

## Clone Repository

```bash
git clone https://github.com/SamReznov/meeting-summarizer.git
```

---

## Create Virtual Environment

```bash
uv venv
```

---

## Activate Environment

### Windows

```bash
.venv\Scripts\activate
```

---

## Install Dependencies

```bash
uv pip install transformers torch accelerate librosa soundfile
```

---

# Run The Project

Place your audio file inside:

```text
audio/
```

Then run:

```bash
python main.py
```

---

# Workflow

```text
Audio File
    ↓
Whisper Transcription
    ↓
Transcript
    ↓
BART Summarization
    ↓
Final Summary
```

---

# Future Improvements

- Real-time transcription
- Speaker diarization
- Timestamp support
- Markdown formatting
- FastAPI backend
- React frontend

---

# Author

Samar
