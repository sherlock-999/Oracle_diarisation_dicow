# DiCoW Chunk WER Evaluation

Evaluates [DiCoW](https://huggingface.co/BUT-FIT/DiCoW_v3_2) (speaker-conditioned Whisper) on multi-speaker audio using oracle diarization, measuring WER at configurable chunk sizes.

## Pipeline

1. **Reference** — parse speaker transcripts from Praat TextGrid files
2. **Diarization mask** — build a full binary `[num_speakers, total_frames]` mask at 50 fps from RTTM
3. **Transcription** — chunk audio into fixed-length windows; for each chunk slice the mask and run DiCoW
4. **WER** — normalize both hypothesis and reference with `EnglishTextNormalizer` (fillers, contractions, case), then compute macro WER (sum S/D/I/H across all speakers, divide once)

## Dataset layout

```
<dataset_root>/
  audio/       *.wav   (16 kHz mono)
  textgrid/    *.TextGrid
  rttm/        *.rttm
```

## Setup

```bash
pip install -r requirements.txt
# scoring_dicow is included as a subdirectory — no separate install needed

# Place the DiCoW model weights in:
#   model/DiCoW/model.safetensors  (+ config.json, tokenizer files)
# or download from HuggingFace:
#   git clone https://huggingface.co/BUT-FIT/DiCoW_v3_2 model/DiCoW
```

## Configuration

Edit `config.yaml`:

```yaml
audio_dir:     /path/to/dataset/audio/
textgrid_dir:  /path/to/dataset/textgrid/
rttm_dir:      /path/to/dataset/rttm/
chunk_length_s: 5.0          # chunk size in seconds
dicow_model:   ./model/DiCoW # local model path
output_dir:    ./output
```

## Run

```bash
python evaluate_chunk_wer.py --config config.yaml
```

Results are written to `output/<chunk_length>s.txt`, e.g. `output/5.0s.txt`:

```
sdm_MTG_32000_sc_meetup_0-0  WER = 18.34%  (S=... D=... I=... H=...)
...

Overall WER = 0.2103
```

## File overview

| File | Description |
|------|-------------|
| `evaluate_chunk_wer.py` | Main evaluation script |
| `dicow_pipeline.py` | `DiCoW_Pipeline` — HuggingFace pipeline wrapper with diarization mask injection |
| `dicow_inference.py` | `DiCoWTranscriber` and tokenizer utilities |
| `config.yaml` | Runtime configuration |
| `scoring_dicow/` | Text normalization and scoring utilities |
