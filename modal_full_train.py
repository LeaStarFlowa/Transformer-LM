from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import time
from pathlib import Path

import modal


APP_NAME = "cs336-tinystories-320m"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_ROOT = Path("/data")

TRAIN_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_DIR = VOLUME_ROOT / "tokenizers" / "tinystories_10k"
VOCAB_PATH = TOKENIZER_DIR / "vocab.json"
MERGES_PATH = TOKENIZER_DIR / "merges.txt"

TOKENIZED_DIR = VOLUME_ROOT / "tokenized" / "tinystories_10k"
TRAIN_IDS_PATH = TOKENIZED_DIR / "train.bin"
VALID_IDS_PATH = TOKENIZED_DIR / "valid.bin"
DATASET_METADATA_PATH = TOKENIZED_DIR / "metadata.json"

RUN_NAME = "tinystories_320m_d448_lr1e-3_20260902"
RUN_DIR = VOLUME_ROOT / "training_runs" / RUN_NAME

SPECIAL_TOKEN = "<|endoftext|>"
GPT2_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
TOKEN_DTYPE = "uint16"
PRETOKENIZE_CHUNK_CHARACTERS = 16 * 1024 * 1024

RANDOM_SEED = 42
MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 448,
    "num_layers": 4,
    "num_heads": 14,
    "d_ff": 1_216,
    "rope_theta": 10_000.0,
    "use_rms_norm": True,
    "norm_mode": "pre",
    "ffn_type": "swiglu",
}
BATCH_SIZE = 128
NUM_STEPS = 9_766
TOKENS_PER_STEP = BATCH_SIZE * MODEL_CONFIG["context_length"]
TARGET_TRAINING_TOKENS = NUM_STEPS * TOKENS_PER_STEP
MAX_LEARNING_RATE = 1e-3
MIN_LEARNING_RATE = 1e-4
WARMUP_STEPS = 100
BETAS = (0.9, 0.95)
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
EVAL_INTERVAL = 100
EVAL_ITERS = 20
CHECKPOINT_INTERVAL = 1_000


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .uv_pip_install("tensorboard>=2.20")
    .add_local_python_source("cs336_basics")
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_serialized_vocab() -> dict[int, bytes]:
    from cs336_basics.train_bpe import bytes_to_unicode

    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        serialized_vocab = json.load(file)
    return {
        int(token_id): bytes(byte_decoder[character] for character in serialized_token)
        for token_id, serialized_token in serialized_vocab.items()
    }


def _load_fast_tokenizer():
    import tiktoken

    vocab = _load_serialized_vocab()
    special_bytes = SPECIAL_TOKEN.encode("utf-8")
    special_token_id = next((token_id for token_id, token in vocab.items() if token == special_bytes), None)
    if special_token_id is None:
        raise ValueError(f"{SPECIAL_TOKEN!r} is absent from {VOCAB_PATH}")

    mergeable_ranks = {token: token_id for token_id, token in vocab.items() if token != special_bytes}
    encoding = tiktoken.Encoding(
        name="tinystories_10k",
        pat_str=GPT2_PATTERN,
        mergeable_ranks=mergeable_ranks,
        special_tokens={SPECIAL_TOKEN: special_token_id},
    )
    return encoding, vocab


def _load_reference_tokenizer():
    from cs336_basics.tokenizer import BPETokenizer
    from cs336_basics.train_bpe import bytes_to_unicode

    vocab = _load_serialized_vocab()
    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    merges = []
    with MERGES_PATH.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                left, right = line.split(" ", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Invalid merge at {MERGES_PATH}:{line_number}") from error
            merges.append(
                (
                    bytes(byte_decoder[character] for character in left),
                    bytes(byte_decoder[character] for character in right),
                )
            )
    return BPETokenizer(vocab, merges, [SPECIAL_TOKEN])


def _assert_fast_tokenizer_equivalent(fast_tokenizer) -> None:
    reference = _load_reference_tokenizer()
    samples = (
        "Once upon a time, there was a tiny dragon.",
        f"First story.{SPECIAL_TOKEN}\nSecond story!",
        "Unicode: café, 中文, naïve, 🙂\n\n",
    )
    for sample in samples:
        fast_ids = fast_tokenizer.encode(sample, allowed_special={SPECIAL_TOKEN})
        reference_ids = reference.encode(sample)
        if fast_ids != reference_ids:
            raise AssertionError(f"Fast tokenizer mismatch for sample {sample!r}")


def _dataset_signature() -> dict[str, object]:
    return {
        "format_version": 1,
        "train_source": str(TRAIN_TEXT_PATH),
        "valid_source": str(VALID_TEXT_PATH),
        "train_source_bytes": TRAIN_TEXT_PATH.stat().st_size,
        "valid_source_bytes": VALID_TEXT_PATH.stat().st_size,
        "vocab_sha256": _sha256(VOCAB_PATH),
        "merges_sha256": _sha256(MERGES_PATH),
        "vocab_size": MODEL_CONFIG["vocab_size"],
        "special_token": SPECIAL_TOKEN,
        "token_dtype": TOKEN_DTYPE,
    }


def _encode_file(source: Path, destination: Path, tokenizer) -> int:
    import numpy as np

    partial_path = destination.with_suffix(destination.suffix + ".partial")
    token_count = 0
    input_characters = 0
    buffer = ""
    started_at = time.monotonic()

    with source.open("r", encoding="utf-8") as input_file, partial_path.open("wb") as output_file:
        while True:
            chunk = input_file.read(PRETOKENIZE_CHUNK_CHARACTERS)
            if not chunk:
                text_to_encode = buffer
                buffer = ""
            else:
                input_characters += len(chunk)
                buffer += chunk
                boundary = buffer.rfind(SPECIAL_TOKEN)
                if boundary == -1:
                    continue
                boundary += len(SPECIAL_TOKEN)
                text_to_encode = buffer[:boundary]
                buffer = buffer[boundary:]

            if text_to_encode:
                token_ids = tokenizer.encode(text_to_encode, allowed_special={SPECIAL_TOKEN})
                if token_ids:
                    maximum_id = max(token_ids)
                    if maximum_id >= MODEL_CONFIG["vocab_size"]:
                        raise ValueError(f"Token id {maximum_id} exceeds configured vocabulary")
                    np.asarray(token_ids, dtype=np.uint16).tofile(output_file)
                    token_count += len(token_ids)

            elapsed = max(time.monotonic() - started_at, 1e-9)
            print(
                f"[{source.name}] chars={input_characters:,}, tokens={token_count:,}, "
                f"rate={input_characters / elapsed / 1e6:.2f}M chars/s"
            )
            if not chunk:
                break

        output_file.flush()
        os.fsync(output_file.fileno())

    os.replace(partial_path, destination)
    return token_count


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    cpu=16.0,
    memory=32_768,
    timeout=3 * 60 * 60,
)
def prepare_full_dataset(force: bool = False) -> dict:
    for path in (TRAIN_TEXT_PATH, VALID_TEXT_PATH, VOCAB_PATH, MERGES_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input not found: {path}")

    signature = _dataset_signature()
    if not force and TRAIN_IDS_PATH.is_file() and VALID_IDS_PATH.is_file() and DATASET_METADATA_PATH.is_file():
        with DATASET_METADATA_PATH.open("r", encoding="utf-8") as file:
            existing = json.load(file)
        if all(existing.get(key) == value for key, value in signature.items()):
            expected_train_bytes = int(existing["train_tokens"]) * 2
            expected_valid_bytes = int(existing["valid_tokens"]) * 2
            if TRAIN_IDS_PATH.stat().st_size == expected_train_bytes and VALID_IDS_PATH.stat().st_size == expected_valid_bytes:
                print("Reusing verified full token dataset")
                return existing

    tokenizer, vocab = _load_fast_tokenizer()
    if len(vocab) != MODEL_CONFIG["vocab_size"]:
        raise AssertionError(f"Expected vocab size {MODEL_CONFIG['vocab_size']}, got {len(vocab)}")
    _assert_fast_tokenizer_equivalent(tokenizer)
    print("Fast tiktoken encoding exactly matches the assignment tokenizer on validation samples")

    TOKENIZED_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.monotonic()
    train_tokens = _encode_file(TRAIN_TEXT_PATH, TRAIN_IDS_PATH, tokenizer)
    valid_tokens = _encode_file(VALID_TEXT_PATH, VALID_IDS_PATH, tokenizer)
    minimum_tokens = MODEL_CONFIG["context_length"] + 1
    if train_tokens < minimum_tokens or valid_tokens < minimum_tokens:
        raise RuntimeError("Tokenized dataset is too short")

    metadata = {
        **signature,
        "train_tokens": train_tokens,
        "valid_tokens": valid_tokens,
        "train_binary_bytes": TRAIN_IDS_PATH.stat().st_size,
        "valid_binary_bytes": VALID_IDS_PATH.stat().st_size,
        "elapsed_seconds": time.monotonic() - started_at,
    }
    with DATASET_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    data_volume.commit()
    print(f"Prepared full dataset: train={train_tokens:,} tokens, valid={valid_tokens:,} tokens")
    return metadata


def _model_kwargs(device: str) -> dict[str, object]:
    return {**MODEL_CONFIG, "device": device}


def _sample_batch(dataset, batch_size: int, context_length: int, device: str):
    import numpy as np
    import torch

    max_start = len(dataset) - context_length - 1
    if max_start < 0:
        raise ValueError("Dataset is shorter than one context window")
    starts = torch.randint(0, max_start + 1, (batch_size,)).tolist()
    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return torch.from_numpy(inputs).to(device).long(), torch.from_numpy(targets).to(device).long()


def _mean_loss(model, dataset, device: str) -> float:
    import torch

    from cs336_basics.losses import cross_entropy

    cpu_rng_state = torch.random.get_rng_state()
    torch.manual_seed(RANDOM_SEED + 10_000)
    model.eval()
    total = 0.0
    try:
        with torch.no_grad():
            for _ in range(EVAL_ITERS):
                inputs, targets = _sample_batch(
                    dataset,
                    BATCH_SIZE,
                    MODEL_CONFIG["context_length"],
                    device,
                )
                total += cross_entropy(model(inputs), targets).item()
    finally:
        torch.random.set_rng_state(cpu_rng_state)
    return total / EVAL_ITERS


def _save_resume_checkpoint(path: Path, model, optimizer, step: int, elapsed_seconds: float) -> None:
    import numpy as np
    import torch

    temporary = path.with_suffix(path.suffix + ".partial")
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": step,
        "elapsed_seconds": elapsed_seconds,
        "torch_rng_state": torch.random.get_rng_state(),
        "numpy_rng_state": np.random.get_state(),
        "config": _run_config(),
    }
    torch.save(checkpoint, temporary)
    os.replace(temporary, path)


def _load_resume_checkpoint(path: Path, model, optimizer) -> tuple[int, float]:
    import numpy as np
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    torch.random.set_rng_state(checkpoint["torch_rng_state"])
    np.random.set_state(checkpoint["numpy_rng_state"])
    return int(checkpoint["iteration"]), float(checkpoint.get("elapsed_seconds", 0.0))


def _run_config() -> dict[str, object]:
    return {
        "run_name": RUN_NAME,
        "random_seed": RANDOM_SEED,
        "model": MODEL_CONFIG,
        "optimizer": {
            "name": "AdamW",
            "max_learning_rate": MAX_LEARNING_RATE,
            "min_learning_rate": MIN_LEARNING_RATE,
            "warmup_steps": WARMUP_STEPS,
            "betas": list(BETAS),
            "weight_decay": WEIGHT_DECAY,
            "max_grad_norm": MAX_GRAD_NORM,
        },
        "training": {
            "batch_size": BATCH_SIZE,
            "num_steps": NUM_STEPS,
            "tokens_per_step": TOKENS_PER_STEP,
            "target_training_tokens": TARGET_TRAINING_TOKENS,
            "eval_interval": EVAL_INTERVAL,
            "eval_iters": EVAL_ITERS,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "precision": "float32 with TF32 matmul enabled",
        },
        "data": {
            "train_ids": str(TRAIN_IDS_PATH),
            "valid_ids": str(VALID_IDS_PATH),
            "dtype": TOKEN_DTYPE,
        },
    }


METRIC_FIELDS = (
    "step",
    "tokens_seen",
    "elapsed_seconds",
    "split",
    "loss",
    "learning_rate",
    "gradient_norm",
    "tokens_per_second",
    "peak_gpu_memory_mib",
)


def _truncate_metrics_for_resume(path: Path, start_step: int) -> None:
    if not path.is_file():
        return
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = [row for row in csv.DictReader(file) if int(row["step"]) <= start_step]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read_loss_history(path: Path) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    train_points = []
    valid_points = []
    with path.open("r", newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            point = (int(row["step"]), float(row["loss"]))
            if row["split"] == "train":
                train_points.append(point)
            elif row["split"] == "valid":
                valid_points.append(point)
    return train_points, valid_points


def _moving_average(points: list[tuple[int, float]], window: int = 50) -> list[tuple[int, float]]:
    smoothed = []
    values = []
    running_sum = 0.0
    for step, value in points:
        values.append(value)
        running_sum += value
        if len(values) > window:
            running_sum -= values[-window - 1]
        smoothed.append((step, running_sum / min(len(values), window)))
    return smoothed


def _write_loss_svg(metrics_path: Path, output_path: Path) -> None:
    train_points, valid_points = _read_loss_history(metrics_path)
    train_points = _moving_average(train_points)
    all_points = train_points + valid_points
    if not all_points:
        return

    width, height = 1_200, 700
    left, right, top, bottom = 90, 40, 55, 80
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(step for step, _ in all_points)
    min_loss = min(value for _, value in all_points)
    max_loss = max(value for _, value in all_points)
    loss_padding = max((max_loss - min_loss) * 0.05, 0.1)
    y_min = max(0.0, min_loss - loss_padding)
    y_max = max_loss + loss_padding

    def xy(point: tuple[int, float]) -> tuple[float, float]:
        step, loss = point
        x = left + plot_width * step / max(max_step, 1)
        y = top + plot_height * (y_max - loss) / max(y_max - y_min, 1e-9)
        return x, y

    def polyline(points: list[tuple[int, float]]) -> str:
        if len(points) > 2_000:
            stride = math.ceil(len(points) / 2_000)
            points = points[::stride] + ([points[-1]] if points[-1] != points[::stride][-1] else [])
        return " ".join(f"{x:.2f},{y:.2f}" for x, y in map(xy, points))

    grid = []
    labels = []
    for index in range(6):
        fraction = index / 5
        x = left + plot_width * fraction
        step = round(max_step * fraction)
        grid.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>')
        labels.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle">{step:,}</text>')
        y = top + plot_height * fraction
        loss = y_max - (y_max - y_min) * fraction
        grid.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>')
        labels.append(f'<text x="{left - 14}" y="{y + 5:.1f}" text-anchor="end">{loss:.2f}</text>')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#ffffff"/>
<style>text {{ font-family: Inter, Arial, sans-serif; fill: #263238; font-size: 15px }} .grid {{ stroke: #dfe5eb; stroke-width: 1 }} .axis {{ stroke: #455a64; stroke-width: 1.5 }}</style>
<text x="{width / 2}" y="32" text-anchor="middle" style="font-size:24px;font-weight:600">TinyStories 320M-token training</text>
<g class="grid">{''.join(grid)}</g>
<line class="axis" x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}"/>
<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>
{''.join(labels)}
<polyline points="{polyline(train_points)}" fill="none" stroke="#1976d2" stroke-width="2" opacity="0.9"/>
<polyline points="{polyline(valid_points)}" fill="none" stroke="#e65100" stroke-width="3"/>
<g><line x1="{width - 330}" y1="74" x2="{width - 290}" y2="74" stroke="#1976d2" stroke-width="3"/><text x="{width - 280}" y="80">train (50-step mean)</text>
<line x1="{width - 330}" y1="103" x2="{width - 290}" y2="103" stroke="#e65100" stroke-width="3"/><text x="{width - 280}" y="109">validation</text></g>
<text x="{left + plot_width / 2}" y="{height - 10}" text-anchor="middle">optimizer step</text>
<text transform="translate(24 {top + plot_height / 2}) rotate(-90)" text-anchor="middle">cross-entropy loss</text>
</svg>'''
    output_path.write_text(svg, encoding="utf-8")


def _log(file, message: str) -> None:
    timestamped = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(timestamped, flush=True)
    file.write(timestamped + "\n")
    file.flush()


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    gpu="B200",
    cpu=8.0,
    memory=32_768,
    timeout=6 * 60 * 60,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=10.0),
)
def train_320m() -> dict:
    import numpy as np
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from cs336_basics.losses import cross_entropy
    from cs336_basics.nn import TransformerLM
    from cs336_basics.optimizer import AdamW
    from cs336_basics.scheduler import get_lr_cosine_schedule

    data_volume.reload()
    for path in (TRAIN_IDS_PATH, VALID_IDS_PATH, DATASET_METADATA_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset is missing: {path}; run prepare_full_dataset first")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = RUN_DIR / "summary.json"
    if summary_path.is_file():
        with summary_path.open("r", encoding="utf-8") as file:
            existing_summary = json.load(file)
        if existing_summary.get("status") == "complete":
            print(f"Run is already complete at {RUN_DIR}; reusing artifacts")
            return existing_summary

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA GPU")
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)

    train_data = np.memmap(TRAIN_IDS_PATH, dtype=np.uint16, mode="r")
    valid_data = np.memmap(VALID_IDS_PATH, dtype=np.uint16, mode="r")
    if len(train_data) <= MODEL_CONFIG["context_length"] or len(valid_data) <= MODEL_CONFIG["context_length"]:
        raise RuntimeError("Prepared token dataset is invalid")

    model = TransformerLM(**_model_kwargs(device))
    optimizer = AdamW(
        model.parameters(),
        lr=MAX_LEARNING_RATE,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    config = {
        **_run_config(),
        "parameter_count": parameter_count,
        "gpu": gpu_name,
        "train_dataset_tokens": int(len(train_data)),
        "valid_dataset_tokens": int(len(valid_data)),
    }
    with (RUN_DIR / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, ensure_ascii=False, indent=2)

    latest_checkpoint_path = RUN_DIR / "checkpoint_latest.pt"
    final_checkpoint_path = RUN_DIR / "checkpoint_final.pt"
    final_model_path = RUN_DIR / "final_model.pt"
    best_model_path = RUN_DIR / "best_model.pt"
    metrics_path = RUN_DIR / "metrics.csv"
    train_log_path = RUN_DIR / "train.log"
    start_step = 0
    previous_elapsed = 0.0

    if latest_checkpoint_path.is_file():
        start_step, previous_elapsed = _load_resume_checkpoint(latest_checkpoint_path, model, optimizer)
        _truncate_metrics_for_resume(metrics_path, start_step)

    metrics_exists = metrics_path.is_file() and metrics_path.stat().st_size > 0
    metrics_file = metrics_path.open("a", newline="", encoding="utf-8")
    metrics_writer = csv.DictWriter(metrics_file, fieldnames=METRIC_FIELDS)
    if not metrics_exists:
        metrics_writer.writeheader()
        metrics_file.flush()

    log_file = train_log_path.open("a", encoding="utf-8")
    tensorboard_writer = SummaryWriter(log_dir=str(RUN_DIR / "tensorboard"), purge_step=start_step or None)
    _log(
        log_file,
        f"device={gpu_name}; parameters={parameter_count:,}; start_step={start_step}; "
        f"target_steps={NUM_STEPS:,}; target_tokens={TARGET_TRAINING_TOKENS:,}",
    )

    train_points, valid_points = _read_loss_history(metrics_path)
    best_valid_loss = min((loss for _, loss in valid_points), default=float("inf"))
    best_step = min(valid_points, key=lambda point: point[1])[0] if valid_points else 0
    run_started_at = time.monotonic()
    interval_started_at = run_started_at
    interval_tokens = 0
    completed_step = start_step

    try:
        if start_step == 0 and not valid_points:
            initial_valid_loss = _mean_loss(model, valid_data, device)
            metrics_writer.writerow(
                {
                    "step": 0,
                    "tokens_seen": 0,
                    "elapsed_seconds": previous_elapsed,
                    "split": "valid",
                    "loss": f"{initial_valid_loss:.8f}",
                    "learning_rate": "0.0",
                    "gradient_norm": "",
                    "tokens_per_second": "",
                    "peak_gpu_memory_mib": f"{torch.cuda.max_memory_allocated() / 2**20:.2f}",
                }
            )
            metrics_file.flush()
            tensorboard_writer.add_scalar("loss/valid", initial_valid_loss, 0)
            best_valid_loss = initial_valid_loss
            best_step = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "iteration": 0,
                    "valid_loss": initial_valid_loss,
                    "config": config,
                },
                best_model_path,
            )
            _log(log_file, f"step=0 valid_loss={initial_valid_loss:.4f}")

        for step in range(start_step + 1, NUM_STEPS + 1):
            learning_rate = get_lr_cosine_schedule(
                step,
                max_learning_rate=MAX_LEARNING_RATE,
                min_learning_rate=MIN_LEARNING_RATE,
                warmup_iters=WARMUP_STEPS,
                cosine_cycle_iters=NUM_STEPS,
            )
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = learning_rate

            model.train()
            inputs, targets = _sample_batch(
                train_data,
                BATCH_SIZE,
                MODEL_CONFIG["context_length"],
                device,
            )
            optimizer.zero_grad()
            loss = cross_entropy(model(inputs), targets)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=MAX_GRAD_NORM,
                foreach=True,
            ).item()
            if not math.isfinite(gradient_norm):
                raise FloatingPointError(f"Non-finite gradient norm at step {step}: {gradient_norm}")
            optimizer.step()
            completed_step = step

            loss_value = loss.item()
            interval_tokens += TOKENS_PER_STEP
            elapsed_seconds = previous_elapsed + time.monotonic() - run_started_at
            interval_elapsed = max(time.monotonic() - interval_started_at, 1e-9)
            tokens_per_second = interval_tokens / interval_elapsed
            peak_memory_mib = torch.cuda.max_memory_allocated() / 2**20
            metrics_writer.writerow(
                {
                    "step": step,
                    "tokens_seen": step * TOKENS_PER_STEP,
                    "elapsed_seconds": f"{elapsed_seconds:.6f}",
                    "split": "train",
                    "loss": f"{loss_value:.8f}",
                    "learning_rate": f"{learning_rate:.12g}",
                    "gradient_norm": f"{gradient_norm:.8f}",
                    "tokens_per_second": f"{tokens_per_second:.2f}",
                    "peak_gpu_memory_mib": f"{peak_memory_mib:.2f}",
                }
            )
            tensorboard_writer.add_scalar("loss/train", loss_value, step)
            tensorboard_writer.add_scalar("optimization/learning_rate", learning_rate, step)
            tensorboard_writer.add_scalar("optimization/gradient_norm", gradient_norm, step)
            tensorboard_writer.add_scalar("performance/tokens_per_second", tokens_per_second, step)
            tensorboard_writer.add_scalar("performance/peak_gpu_memory_mib", peak_memory_mib, step)

            should_evaluate = step % EVAL_INTERVAL == 0 or step == NUM_STEPS
            if should_evaluate:
                valid_loss = _mean_loss(model, valid_data, device)
                elapsed_seconds = previous_elapsed + time.monotonic() - run_started_at
                metrics_writer.writerow(
                    {
                        "step": step,
                        "tokens_seen": step * TOKENS_PER_STEP,
                        "elapsed_seconds": f"{elapsed_seconds:.6f}",
                        "split": "valid",
                        "loss": f"{valid_loss:.8f}",
                        "learning_rate": f"{learning_rate:.12g}",
                        "gradient_norm": "",
                        "tokens_per_second": f"{tokens_per_second:.2f}",
                        "peak_gpu_memory_mib": f"{peak_memory_mib:.2f}",
                    }
                )
                metrics_file.flush()
                tensorboard_writer.add_scalar("loss/valid", valid_loss, step)
                tensorboard_writer.flush()
                _log(
                    log_file,
                    f"step={step:,}/{NUM_STEPS:,} tokens={step * TOKENS_PER_STEP:,} "
                    f"train_loss={loss_value:.4f} valid_loss={valid_loss:.4f} "
                    f"lr={learning_rate:.3e} grad_norm={gradient_norm:.3f} "
                    f"throughput={tokens_per_second:,.0f} tok/s peak_memory={peak_memory_mib:,.0f} MiB",
                )
                interval_started_at = time.monotonic()
                interval_tokens = 0
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    best_step = step
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "iteration": step,
                            "valid_loss": valid_loss,
                            "config": config,
                        },
                        best_model_path,
                    )

            if step % CHECKPOINT_INTERVAL == 0 and step < NUM_STEPS:
                elapsed_seconds = previous_elapsed + time.monotonic() - run_started_at
                _save_resume_checkpoint(latest_checkpoint_path, model, optimizer, step, elapsed_seconds)
                data_volume.commit()
                _log(log_file, f"saved resumable checkpoint at step {step:,}")

        total_elapsed = previous_elapsed + time.monotonic() - run_started_at
        _save_resume_checkpoint(final_checkpoint_path, model, optimizer, NUM_STEPS, total_elapsed)
        _save_resume_checkpoint(latest_checkpoint_path, model, optimizer, NUM_STEPS, total_elapsed)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "iteration": NUM_STEPS,
                "config": config,
            },
            final_model_path,
        )

        tokenizer = _load_reference_tokenizer()
        prompt = "Once upon a time, there was a little girl named Lily who"
        prompt_ids = torch.tensor(tokenizer.encode(prompt), dtype=torch.long, device=device).unsqueeze(0)
        eos_token_id = tokenizer.byte_to_id[SPECIAL_TOKEN.encode("utf-8")]
        torch.manual_seed(RANDOM_SEED + 20_000)
        generated_ids = model.generate(
            prompt_ids,
            max_new_tokens=160,
            eos_token_id=eos_token_id,
            temperature=0.8,
            top_p=0.9,
        )[0].tolist()
        sample = tokenizer.decode(generated_ids)
        (RUN_DIR / "sample.txt").write_text(sample + "\n", encoding="utf-8")
        _write_loss_svg(metrics_path, RUN_DIR / "loss_curves.svg")

        _, final_valid_points = _read_loss_history(metrics_path)
        final_valid_loss = final_valid_points[-1][1]
        summary = {
            "status": "complete",
            "run_name": RUN_NAME,
            "completed_steps": NUM_STEPS,
            "training_tokens": TARGET_TRAINING_TOKENS,
            "parameter_count": parameter_count,
            "best_valid_loss": best_valid_loss,
            "best_step": best_step,
            "final_valid_loss": final_valid_loss,
            "elapsed_seconds": total_elapsed,
            "average_training_tokens_per_second": TARGET_TRAINING_TOKENS / total_elapsed,
            "gpu": gpu_name,
            "artifacts": {
                "metrics_csv": str(RUN_DIR / "metrics.csv"),
                "tensorboard": str(RUN_DIR / "tensorboard"),
                "loss_curves_svg": str(RUN_DIR / "loss_curves.svg"),
                "final_model": str(final_model_path),
                "resumable_final_checkpoint": str(final_checkpoint_path),
                "training_log": str(train_log_path),
                "generated_sample": str(RUN_DIR / "sample.txt"),
            },
            "artifact_sizes_bytes": {
                "final_model": final_model_path.stat().st_size,
                "resumable_final_checkpoint": final_checkpoint_path.stat().st_size,
            },
            "sample": sample,
        }
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)
        _log(
            log_file,
            f"training complete: best_valid_loss={best_valid_loss:.4f} at step={best_step:,}; "
            f"final_valid_loss={final_valid_loss:.4f}; elapsed={total_elapsed / 60:.1f} minutes",
        )
        tensorboard_writer.flush()
        data_volume.commit()
        return summary
    except Exception:
        elapsed_seconds = previous_elapsed + time.monotonic() - run_started_at
        if completed_step > start_step:
            _save_resume_checkpoint(latest_checkpoint_path, model, optimizer, completed_step, elapsed_seconds)
            data_volume.commit()
        _log(log_file, "training interrupted; latest state was saved for automatic resume")
        raise
    finally:
        tensorboard_writer.close()
        metrics_file.close()
        log_file.close()


@app.local_entrypoint()
def main(mode: str = "all", force_prepare: bool = False):
    if mode not in {"prepare", "train", "all"}:
        raise ValueError("mode must be one of: prepare, train, all")
    if mode in {"prepare", "all"}:
        metadata = prepare_full_dataset.remote(force=force_prepare)
        print(json.dumps(metadata, ensure_ascii=False, indent=2))
    if mode in {"train", "all"}:
        summary = train_320m.remote()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
