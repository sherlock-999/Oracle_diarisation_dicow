"""
Evaluate DiCoW WER on chunked audio using oracle diarization.

Pipeline per file:
  1. load_reference(textgrid)              →  {speaker: transcript}
  2. load_diarization_mask(rttm, duration) →  speakers, full_mask [num_spk, total_frames]
  3. transcribe_audio(audio, full_mask)    →  {speaker: hypothesis}
     - per chunk: pipeline.diarization_mask = full_mask[:, f_start:f_end]
  4. Compute WER(reference, hypothesis) per speaker
"""

import re
import sys
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import soundfile as sf
import librosa
import torch
import yaml
import jiwer
from tqdm import tqdm

# Use scoring_dicow for TextGrid parsing and text normalization
sys.path.insert(0, str(Path(__file__).parent / "scoring_dicow" / "src"))
from scoring_dicow.reference import parse_textgrid, normalize_rows

from transformers import AutoFeatureExtractor, AutoTokenizer
from model.DiCoW.modeling_dicow import DiCoWForConditionalGeneration
from dicow_pipeline import DiCoW_Pipeline
from dicow_inference import create_lower_uppercase_mapping


_DIAR_FPS = 50  # frames per second used by DiCoW diarization mask


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Reference transcriptions from TextGrid
# ─────────────────────────────────────────────────────────────────────────────

def load_reference(textgrid_path: str) -> Dict[str, str]:
    """
    Parse a Praat TextGrid and return {speaker: transcript}.
    Uses scoring_dicow's parse_textgrid() which handles clean_text (lowercase,
    tag removal), then aggregates all intervals per speaker.
    """
    rows = parse_textgrid(Path(textgrid_path))
    by_speaker: Dict[str, List[str]] = defaultdict(list)
    for row in rows:
        by_speaker[row["speaker"]].append(row["words"])
    return {spk: " ".join(words) for spk, words in by_speaker.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Full diarization mask from RTTM
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
# Step 3 — Transcribe chunked audio
# ─────────────────────────────────────────────────────────────────────────────

def transcribe_audio(
    audio: np.ndarray,
    sr: int,
    speakers: List[str],
    full_diar_mask: torch.Tensor,
    pipeline: DiCoW_Pipeline,
    chunk_length_s: float,
) -> Dict[str, str]:
    """
    Split audio into non-overlapping chunks and transcribe with DiCoW.
    For each chunk, slice full_diar_mask[:, f_start:f_end] and set it on the pipeline.
    Returns {speaker: full_hypothesis_text}.
    """
    total_duration = len(audio) / sr
    chunk_samples  = int(chunk_length_s * sr)
    num_chunks     = int(np.ceil(total_duration / chunk_length_s))

    per_spk_hyps: Dict[str, List[str]] = {spk: [] for spk in speakers}

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
                per_spk_hyps[spk].append(clean)

    finally:
        pipeline.diarization_mask = None

    return {spk: " ".join(per_spk_hyps[spk]) for spk in speakers}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate one file
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_file(
    audio_path: str,
    rttm_path: str,
    tg_path: str,
    pipeline: DiCoW_Pipeline,
    chunk_length_s: float,
) -> Dict[str, Tuple[str, str]]:
    """
    Evaluate one recording.  Returns {speaker: (hypothesis, reference)}.
    """
    references = load_reference(tg_path)

    audio, sr = librosa.load(audio_path, sr=16_000, mono=True)  # resample once here

    speakers, full_diar_mask = load_diarization_mask(rttm_path, len(audio) / sr)
    hypotheses = transcribe_audio(audio, sr, speakers, full_diar_mask, pipeline, chunk_length_s)

    return {
        spk: (hypotheses.get(spk, ""), references.get(spk, ""))
        for spk in speakers
    }


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

    files = sorted(audio_dir.glob("*.wav"))
    print(f"Found {len(files)} files.  Chunk length: {chunk_length}s\n")

    print(f"Loading model: {cfg['dicow_model']}")
    pipeline = load_pipeline(cfg["dicow_model"], device)
    print("Model ready.\n")

    file_wer_lines = []   # recorded per-file WER strings
    total_s = total_d = total_i = total_h = 0  # dataset-level accumulators

    for audio_path in tqdm(files):
        rttm_path = rttm_dir / (audio_path.stem + ".rttm")
        tg_path   = tg_dir   / (audio_path.stem + ".TextGrid")

        print(f"Processing: {audio_path.name}")
        results = evaluate_file(str(audio_path), str(rttm_path), str(tg_path),
                                pipeline, chunk_length)

        # Accumulate counts for this file across all its speakers
        file_s = file_d = file_i = file_h = 0
        for spk, (hyp, ref) in results.items():
            if not ref.strip():
                continue
            norm_rows = normalize_rows([{"words": ref}, {"words": hyp}])
            ref_n, hyp_n = norm_rows[0]["words"], norm_rows[1]["words"]
            m = jiwer.process_words(ref_n, hyp_n)
            file_s += m.substitutions
            file_d += m.deletions
            file_i += m.insertions
            file_h += m.hits

        file_wer = (file_s + file_d + file_i) / max(1, file_s + file_d + file_h)
        line = f"{audio_path.stem}  WER = {file_wer:.2%}  (S={file_s} D={file_d} I={file_i} H={file_h})"
        print(f"  {line}")
        file_wer_lines.append(line)

        total_s += file_s
        total_d += file_d
        total_i += file_i
        total_h += file_h

    overall = (total_s + total_d + total_i) / max(1, total_s + total_d + total_h)
    print(f"\n── Dataset WER " + "─" * 50)
    print(f"  Overall WER = {overall:.2%}  (S={total_s} D={total_d} I={total_i} H={total_h})")

    report = "\n".join(file_wer_lines) + f"\n\nOverall WER = {overall:.4f}\n"
    (output_dir / f"{chunk_length}s.txt").write_text(report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    main(args.config)
