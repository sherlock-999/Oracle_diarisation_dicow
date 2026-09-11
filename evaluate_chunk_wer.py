"""
Evaluate chunked DiCoW with oracle diarization and `scoring_dicow`.

Pipeline per file:
  1. load_diarization_mask(rttm, duration) → oracle speaker labels and mask
  2. transcribe_audio(audio, full_mask)    → chunk-level hypothesis JSONL rows
  3. score_dataset(...)                    → oracle fixed-label diagnostics
"""

import re
import sys
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import librosa
import torch
import yaml
from tqdm import tqdm

from transformers import AutoFeatureExtractor, AutoTokenizer
from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_pipeline import DiCoW_Pipeline
from dicow_inference import create_lower_uppercase_mapping


_DIAR_FPS = 50  # frames per second used by DiCoW diarization mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Full diarization mask from RTTM
# ─────────────────────────────────────────────────────────────────────────────

def load_diarization_mask(
    rttm_path: str,
    total_duration: float,
) -> Tuple[List[str], torch.Tensor]:
    """
    Parse an RTTM file and build the full binary diarization mask for the recording.
    Returns:
        speakers   — sorted list of speaker names (defines row order in the mask)
        full_mask  — [num_speakers, total_frames] at _DIAR_FPS
    Per chunk, slice as full_mask[:, f_start:f_end].
    """
    segments: Dict[str, List[Tuple[float, float]]] = {}
    with open(rttm_path) as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0] != "SPEAKER":
                continue
            start    = float(parts[3])
            end      = start + float(parts[4])
            speaker  = parts[7].lower()  # match parse_textgrid() which lowercases tier names
            segments.setdefault(speaker, []).append((start, end))

    speakers     = sorted(segments.keys())
    total_frames = max(1, round(total_duration * _DIAR_FPS))
    full_mask    = torch.zeros(len(speakers), total_frames)

    for i, spk in enumerate(speakers):
        for seg_start, seg_end in segments[spk]:
            f0 = max(0, min(round(seg_start * _DIAR_FPS), total_frames))
            f1 = max(0, min(round(seg_end   * _DIAR_FPS), total_frames))
            full_mask[i, f0:f1] = 1.0

    return speakers, full_mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Transcribe chunked audio
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    speakers: List[str],
    full_diar_mask: torch.Tensor,
    pipeline: DiCoW_Pipeline,
    chunk_length_s: float,
) -> List[dict]:
    """
    Split audio into non-overlapping chunks and transcribe with DiCoW.
    For each chunk, slice full_diar_mask[:, f_start:f_end] and set it on the pipeline.
    Returns one JSONL-compatible segment per oracle speaker per chunk. Chunk
    boundaries are used as timing because Whisper timestamps are chunk-relative.
    """
    total_duration = len(audio) / sr
    chunk_samples  = int(chunk_length_s * sr)
    num_chunks     = int(np.ceil(total_duration / chunk_length_s))

    hypothesis_rows: List[dict] = []

    try:
        for idx in range(num_chunks):
            chunk_start = idx * chunk_length_s
            chunk_end   = min(chunk_start + chunk_length_s, total_duration)

            s0 = idx * chunk_samples
            s1 = min(s0 + chunk_samples, len(audio))
            chunk_audio = audio[s0:s1]

            # Slice the full mask to get this chunk's diarization
            f_start = round(chunk_start * _DIAR_FPS)
            f_end   = round(chunk_end   * _DIAR_FPS)
            pipeline.diarization_mask = full_diar_mask[:, f_start:f_end]

            # Pass array directly — no disk write/read
            result          = pipeline({"array": chunk_audio, "sampling_rate": sr}, return_timestamps=True)
            per_spk_outputs = result["per_spk_outputs"]  # list[str] with <|timestamp|> tokens

            for i, spk in enumerate(speakers):
                clean = re.sub(r"<\|\d+\.\d+\|>", " ", per_spk_outputs[i])
                clean = re.sub(r"\s+", " ", clean).strip()
                hypothesis_rows.append(
                    {
                        "speaker": spk,
                        "start_time": chunk_start,
                        "end_time": chunk_end,
                        "words": clean,
                    }
                )

    finally:
        pipeline.diarization_mask = None

    return hypothesis_rows


# ─────────────────────────────────────────────────────────────────────────────
# Decode one file
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_file(
    audio_path: str,
    rttm_path: str,
    pipeline: DiCoW_Pipeline,
    chunk_length_s: float,
) -> List[dict]:
    """
    Decode one recording. The RTTM speaker names are retained as hypothesis
    labels, which makes this a valid oracle speaker mapping for scoring_dicow.
    """
    audio, sr = librosa.load(audio_path, sr=16_000, mono=True)  # resample once here

    speakers, full_diar_mask = load_diarization_mask(rttm_path, len(audio) / sr)
    hypotheses = transcribe_audio(audio, sr, speakers, full_diar_mask, pipeline, chunk_length_s)
    session_id = Path(audio_path).stem
    return [{"session_id": session_id, **row} for row in hypotheses]


# ─────────────────────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────────────────────

def load_pipeline(model_path: str, device: torch.device) -> DiCoW_Pipeline:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = DiCoWForConditionalGeneration.from_pretrained(
        model_path, local_files_only=True, torch_dtype=dtype
    ).to(device)
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path, local_files_only=True)
    tokenizer         = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    create_lower_uppercase_mapping(tokenizer)
    model.set_tokenizer(tokenizer)
    model.eval()

    return DiCoW_Pipeline(
        model,
        speaker_embedding_model=None,
        feature_extractor=feature_extractor,
        tokenizer=tokenizer,
        device=device,
    )


def load_latest_scorer(scorer_root: Path):
    """Load the shared scorer checkout, never this experiment's stale copy."""
    scorer_src = scorer_root / "src"
    if not (scorer_src / "scoring_dicow" / "metrics.py").is_file():
        raise FileNotFoundError(
            f"Latest scoring_dicow checkout not found at {scorer_root}. "
            "Set scoring_dicow_root in config.yaml."
        )
    sys.path.insert(0, str(scorer_src))
    from scoring_dicow.config import DatasetConfig, SpeakerMappingConfig
    from scoring_dicow.metrics import score_dataset

    return DatasetConfig, SpeakerMappingConfig, score_dataset


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main(config_path: str) -> None:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device     = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_dir    = Path(cfg["audio_dir"])
    rttm_dir     = Path(cfg["rttm_dir"])
    tg_dir       = Path(cfg["textgrid_dir"])
    chunk_length = cfg.get("chunk_length_s", 5.0)
    collar = int(cfg.get("collar", 5))
    dataset_name = str(cfg.get("dataset_name", audio_dir.parent.name))
    mapping = str(cfg.get("mapping", dataset_name))
    testset_root = Path(cfg.get("testset_root", audio_dir.parent.parent))
    scorer_root = Path(
        cfg.get(
            "scoring_dicow_root",
            Path(__file__).resolve().parents[1] / "scoring_dicow",
        )
    )
    DatasetConfig, SpeakerMappingConfig, score_dataset = load_latest_scorer(scorer_root)

    files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(files)} files.  Chunk length: {chunk_length}s\n")

    print(f"Loading model: {cfg['dicow_model']}")
    pipeline = load_pipeline(cfg["dicow_model"], device)
    print("Model ready.\n")

    hypothesis_rows: List[dict] = []

    for audio_path in tqdm(files):
        rttm_path = rttm_dir / (audio_path.stem + ".rttm")
        tg_path   = tg_dir   / (audio_path.stem + ".TextGrid")

        print(f"Processing: {audio_path.name}")
        if not rttm_path.is_file() or not tg_path.is_file():
            raise FileNotFoundError(f"Missing RTTM or TextGrid for {audio_path.stem}")
        hypothesis_rows.extend(
            evaluate_file(str(audio_path), str(rttm_path), pipeline, chunk_length)
        )

    hypothesis_path = output_dir / "hypothesis_multi.jsonl"
    with hypothesis_path.open("w", encoding="utf-8") as f:
        for row in hypothesis_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(hypothesis_rows)} oracle-labelled segments to {hypothesis_path}")

    scorer_output = output_dir / "scoring"
    summary = score_dataset(
        testset_root,
        DatasetConfig(
            name=dataset_name,
            predictions=str(hypothesis_path),
            mapping=mapping,
            speaker_mapping=SpeakerMappingConfig(mode="oracle"),
        ),
        scorer_output,
        collar=collar,
    )
    print(json.dumps(summary["normalized_metrics"], indent=2))
    print(f"Diagnostics: {scorer_output / 'diagnostic_sessions.jsonl'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
