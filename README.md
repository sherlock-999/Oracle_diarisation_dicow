# DiCoW Chunk WER Evaluation

Evaluates [DiCoW](https://huggingface.co/BUT-FIT/DiCoW_v3_2) with oracle diarization. It writes oracle-labelled hypotheses and scores them using the shared, current `DiCoW_Experiments/scoring_dicow` checkout.

## Pipeline

1. **Oracle mask** — build `[num_speakers, total_frames]` activity masks from reference RTTM at 50 fps.
2. **Transcription** — decode fixed-length chunks with those masks.
3. **Hypotheses** — save one segment per RTTM speaker and chunk to `output/hypothesis_multi.jsonl`.
4. **Scoring** — invoke the latest `scoring_dicow` in `oracle` mapping mode.

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
# Install the shared scoring_dicow checkout separately, including its runtime dependencies.

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
scoring_dicow_root: /path/to/DiCoW_Experiments/scoring_dicow
testset_root: /path/to/dataset_root  # parent of nsf/, ami/, or l2m/
dataset_name: nsf
mapping: nsf
collar: 5
```

## Run

```bash
python evaluate_chunk_wer.py --config config.yaml
```

Results are written to:

```text
output/hypothesis_multi.jsonl
output/scoring/diagnostic_sessions.jsonl
output/scoring/run_summary.json
output/scoring/normalized_eval/oracle_fixed_wer_average.norm.json
output/scoring/normalized_eval/oracle_fixed_tcpwer_average.norm.json
```

`oracle_fixed_*` does not permute speakers. It is the primary oracle TS-ASR metric because it exposes an incorrect target stream.

## File overview

| File | Description |
|------|-------------|
| `evaluate_chunk_wer.py` | Main evaluation script |
| `dicow_pipeline.py` | `DiCoW_Pipeline` — HuggingFace pipeline wrapper with diarization mask injection |
| `dicow_inference.py` | `DiCoWTranscriber` and tokenizer utilities |
| `config.yaml` | Runtime configuration |
| `scoring_dicow/` | Historical copy only. It is not used. Configure `scoring_dicow_root` to the shared current checkout. |
