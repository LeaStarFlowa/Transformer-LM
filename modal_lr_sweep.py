from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path

import modal


APP_NAME = "cs336-lr-sweep"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_ROOT = Path("/data")
REMOTE_DATASET_DIR = VOLUME_ROOT / "lr_sweep_dataset"
REMOTE_EXPERIMENT_DIR = VOLUME_ROOT / "experiments" / "lr_sweep_comparison_20260902"
LOCAL_EXPERIMENT_DIR = Path("experiments/lr_sweep_comparison_20260902")

TRAIN_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_DIR = VOLUME_ROOT / "tokenizers" / "tinystories_10k"
TRAIN_IDS_PATH = REMOTE_DATASET_DIR / "train_first_2m_chars.npy"
VALID_IDS_PATH = REMOTE_DATASET_DIR / "valid_first_500k_chars.npy"
DATASET_METADATA_PATH = REMOTE_DATASET_DIR / "metadata.json"

LEARNING_RATES = (3e-4, 1e-3, 1.25e-3, 2e-3, 4e-3, 8e-3)
MIN_LR_RATIO = 0.1
RANDOM_SEED = 42

MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 256,
    "d_model": 512,
    "num_layers": 4,
    "num_heads": 16,
    "d_ff": 1_344,
    "rope_theta": 10_000.0,
}
BATCH_SIZE = 16
MAX_GRAD_NORM = 1.0
WEIGHT_DECAY = 0.01
BETAS = (0.9, 0.95)
EVAL_ITERS = 5


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").uv_sync().add_local_python_source("cs336_basics")


def _load_tokenizer():
    from cs336_basics.tokenizer import BPETokenizer
    from cs336_basics.train_bpe import bytes_to_unicode

    vocab_path = TOKENIZER_DIR / "vocab.json"
    merges_path = TOKENIZER_DIR / "merges.txt"
    for path in (vocab_path, merges_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required tokenizer file not found: {path}")

    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    with vocab_path.open("r", encoding="utf-8") as file:
        serialized_vocab = json.load(file)
    vocab = {
        int(token_id): bytes(byte_decoder[character] for character in serialized_token)
        for token_id, serialized_token in serialized_vocab.items()
    }

    merges = []
    with merges_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.rstrip("\n")
            if not line:
                continue
            try:
                left, right = line.split(" ", maxsplit=1)
            except ValueError as error:
                raise ValueError(f"Invalid merge at {merges_path}:{line_number}") from error
            merges.append(
                (
                    bytes(byte_decoder[character] for character in left),
                    bytes(byte_decoder[character] for character in right),
                )
            )
    return BPETokenizer(vocab, merges, ["<|endoftext|>"])


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    cpu=4.0,
    memory=8_192,
    timeout=1_800,
)
def prepare_sweep_dataset(force: bool = False) -> dict:
    import numpy as np

    expected_metadata = {
        "train_source": str(TRAIN_TEXT_PATH),
        "valid_source": str(VALID_TEXT_PATH),
        "train_character_limit": 2_000_000,
        "valid_character_limit": 500_000,
        "token_dtype": "uint16",
        "tokenizer_vocab_size": MODEL_CONFIG["vocab_size"],
    }
    if not force and TRAIN_IDS_PATH.is_file() and VALID_IDS_PATH.is_file() and DATASET_METADATA_PATH.is_file():
        with DATASET_METADATA_PATH.open("r", encoding="utf-8") as file:
            existing_metadata = json.load(file)
        if all(existing_metadata.get(key) == value for key, value in expected_metadata.items()):
            print("Reusing cached LR-sweep token dataset")
            return existing_metadata

    for path in (TRAIN_TEXT_PATH, VALID_TEXT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required corpus not found: {path}")

    tokenizer = _load_tokenizer()
    with TRAIN_TEXT_PATH.open("r", encoding="utf-8") as file:
        train_text = file.read(expected_metadata["train_character_limit"])
    with VALID_TEXT_PATH.open("r", encoding="utf-8") as file:
        valid_text = file.read(expected_metadata["valid_character_limit"])

    train_ids = np.asarray(tokenizer.encode(train_text), dtype=np.uint16)
    valid_ids = np.asarray(tokenizer.encode(valid_text), dtype=np.uint16)
    minimum_length = MODEL_CONFIG["context_length"] + 1
    if len(train_ids) < minimum_length or len(valid_ids) < minimum_length:
        raise RuntimeError("Tokenized LR-sweep dataset is too short")

    REMOTE_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    np.save(TRAIN_IDS_PATH, train_ids)
    np.save(VALID_IDS_PATH, valid_ids)
    metadata = {
        **expected_metadata,
        "train_tokens": int(len(train_ids)),
        "valid_tokens": int(len(valid_ids)),
    }
    with DATASET_METADATA_PATH.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)
    data_volume.commit()
    print(f"Prepared train tokens={len(train_ids):,}, valid tokens={len(valid_ids):,}")
    return metadata


def _mean_loss(model, dataset, eval_iters: int, device: str) -> float:
    import torch

    from cs336_basics.data import get_batch
    from cs336_basics.losses import cross_entropy

    total = 0.0
    model.eval()
    with torch.no_grad():
        for _ in range(eval_iters):
            inputs, targets = get_batch(dataset, BATCH_SIZE, MODEL_CONFIG["context_length"], device)
            total += cross_entropy(model(inputs), targets).item()
    return total / eval_iters


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    gpu="T4",
    cpu=2.0,
    memory=12_288,
    timeout=3_600,
    max_containers=6,
)
def run_learning_rate(config: dict) -> dict:
    import time

    import numpy as np
    import torch

    from cs336_basics.data import get_batch
    from cs336_basics.losses import cross_entropy
    from cs336_basics.nn import TransformerLM
    from cs336_basics.optimizer import AdamW, clip_gradient_norm
    from cs336_basics.scheduler import get_lr_cosine_schedule

    data_volume.reload()
    train_data = np.load(TRAIN_IDS_PATH, mmap_mode="r")
    valid_data = np.load(VALID_IDS_PATH, mmap_mode="r")
    max_lr = float(config["max_lr"])
    min_lr = float(config["min_lr"])
    num_steps = int(config["num_steps"])
    warmup_steps = int(config["warmup_steps"])
    eval_interval = int(config["eval_interval"])
    stage = str(config["stage"])

    torch.manual_seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA GPU")
    device = "cuda"

    model = TransformerLM(
        **MODEL_CONFIG,
        device=device,
        use_rms_norm=True,
        norm_mode="pre",
        ffn_type="swiglu",
    )
    optimizer = AdamW(
        model.parameters(),
        lr=max_lr,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )

    history = []
    initial_train_loss = _mean_loss(model, train_data, EVAL_ITERS, device)
    initial_valid_loss = _mean_loss(model, valid_data, EVAL_ITERS, device)
    history.append(
        {
            "step": 0,
            "train_loss": initial_train_loss,
            "valid_loss": initial_valid_loss,
            "learning_rate": 0.0,
        }
    )
    print(f"[{stage} lr={max_lr:g}] step=0 train={initial_train_loss:.4f} valid={initial_valid_loss:.4f}")

    started_at = time.monotonic()
    diverged = False
    divergence_reason = ""
    final_step = 0
    last_learning_rate = 0.0
    for step in range(1, num_steps + 1):
        last_learning_rate = get_lr_cosine_schedule(
            step,
            max_learning_rate=max_lr,
            min_learning_rate=min_lr,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=num_steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = last_learning_rate

        model.train()
        inputs, targets = get_batch(train_data, BATCH_SIZE, MODEL_CONFIG["context_length"], device)
        optimizer.zero_grad()
        loss = cross_entropy(model(inputs), targets)
        if not torch.isfinite(loss):
            diverged = True
            divergence_reason = f"non-finite train loss at step {step}"
            break
        loss.backward()
        clip_gradient_norm(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        final_step = step

        if step % eval_interval == 0 or step == num_steps:
            train_loss = _mean_loss(model, train_data, EVAL_ITERS, device)
            valid_loss = _mean_loss(model, valid_data, EVAL_ITERS, device)
            history.append(
                {
                    "step": step,
                    "train_loss": train_loss,
                    "valid_loss": valid_loss,
                    "learning_rate": last_learning_rate,
                }
            )
            print(
                f"[{stage} lr={max_lr:g}] step={step} "
                f"train={train_loss:.4f} valid={valid_loss:.4f} lr={last_learning_rate:.3e}"
            )
            if not math.isfinite(valid_loss) or valid_loss > 50:
                diverged = True
                divergence_reason = f"invalid validation loss {valid_loss} at step {step}"
                break

    valid_points = [point for point in history if point["step"] > 0 and math.isfinite(point["valid_loss"])]
    if valid_points:
        best_point = min(valid_points, key=lambda point: point["valid_loss"])
        final_point = valid_points[-1]
    else:
        best_point = {"step": final_step, "valid_loss": float("inf")}
        final_point = {"step": final_step, "valid_loss": float("inf"), "train_loss": float("inf")}

    result = {
        "stage": stage,
        "label": f"{max_lr:g}",
        "max_lr": max_lr,
        "min_lr": min_lr,
        "num_steps": num_steps,
        "warmup_steps": warmup_steps,
        "eval_interval": eval_interval,
        "eval_iters": EVAL_ITERS,
        "seed": RANDOM_SEED,
        "device": torch.cuda.get_device_name(0),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "best_step": int(best_point["step"]),
        "best_valid_loss": float(best_point["valid_loss"]),
        "final_step": int(final_point["step"]),
        "final_train_loss": float(final_point.get("train_loss", float("inf"))),
        "final_valid_loss": float(final_point["valid_loss"]),
        "diverged": diverged,
        "divergence_reason": divergence_reason,
        "elapsed_seconds": time.monotonic() - started_at,
        "history": history,
    }
    print(
        f"[{stage} lr={max_lr:g}] finished: best_valid={result['best_valid_loss']:.4f} "
        f"final_valid={result['final_valid_loss']:.4f} elapsed={result['elapsed_seconds']:.1f}s"
    )
    return result


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    cpu=1.0,
    memory=1_024,
    timeout=300,
)
def persist_artifacts(artifacts: dict[str, str]) -> str:
    REMOTE_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for relative_path, contents in artifacts.items():
        destination = REMOTE_EXPERIMENT_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    data_volume.commit()
    return str(REMOTE_EXPERIMENT_DIR)


def _summary_csv(results: list[dict]) -> str:
    output = io.StringIO()
    fieldnames = [
        "stage",
        "label",
        "max_lr",
        "min_lr",
        "num_steps",
        "best_step",
        "best_valid_loss",
        "final_step",
        "final_train_loss",
        "final_valid_loss",
        "diverged",
        "elapsed_seconds",
        "device",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        writer.writerow({key: result[key] for key in fieldnames})
    return output.getvalue()


def _summary_svg(results: list[dict]) -> str:
    width, height = 1_150, 720
    left, right, top, bottom = 105, 45, 70, 130
    plot_width = width - left - right
    plot_height = height - top - bottom
    finite_results = [result for result in results if math.isfinite(result["best_valid_loss"])]
    full_results = [result for result in finite_results if result["stage"] == "full"]
    full_learning_rates = {result["max_lr"] for result in full_results}
    representative_results = [
        *full_results,
        *[
            result
            for result in finite_results
            if result["stage"] == "probe" and result["max_lr"] not in full_learning_rates
        ],
    ]
    representative_results.sort(key=lambda result: result["max_lr"])

    x_logs = [math.log10(result["max_lr"]) for result in representative_results]
    y_values = [result["best_valid_loss"] for result in representative_results]
    x_min = math.floor(min(x_logs))
    x_max = math.ceil(max(x_logs))
    y_padding = max(0.08, (max(y_values) - min(y_values)) * 0.14)
    y_min = min(y_values) - y_padding
    y_max = max(y_values) + y_padding

    def coordinates(result: dict) -> tuple[float, float]:
        x = left + (math.log10(result["max_lr"]) - x_min) / (x_max - x_min) * plot_width
        y = top + (y_max - result["best_valid_loss"]) / (y_max - y_min) * plot_height
        return x, y

    def superscript(exponent: int) -> str:
        return str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))

    blue = "#1f77b4"
    orange = "#df9200"
    red = "#b94b55"
    gray = "#a6a6a6"
    font = "DejaVu Sans, Arial, sans-serif"
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="38" text-anchor="middle" font-family="{font}" font-size="27">Learning-rate sweep on TinyStories Transformer LM</text>',
    ]

    high_lr_probes = [
        result
        for result in representative_results
        if result["stage"] == "probe" and result["max_lr"] > max(full_learning_rates)
    ]
    if high_lr_probes:
        high_lr_start = min(result["max_lr"] for result in high_lr_probes)
        shade_x = left + (math.log10(high_lr_start) - x_min) / (x_max - x_min) * plot_width
        elements.append(
            f'<rect x="{shade_x:.1f}" y="{top}" width="{left + plot_width - shade_x:.1f}" height="{plot_height}" fill="#f5dfe1" opacity="0.48"/>'
        )

    for decade in range(x_min, x_max + 1):
        for multiplier in range(1, 10):
            value_log = math.log10(multiplier * 10**decade)
            if not x_min <= value_log <= x_max:
                continue
            x = left + (value_log - x_min) / (x_max - x_min) * plot_width
            is_major = multiplier == 1
            grid_color = "#d8d8d8" if is_major else "#eeeeee"
            grid_width = 1.2 if is_major else 0.7
            elements.append(
                f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="{grid_color}" stroke-width="{grid_width}"/>'
            )
            if is_major:
                elements.append(
                    f'<text x="{x:.1f}" y="{top + plot_height + 34}" text-anchor="middle" font-family="{font}" font-size="17">10{superscript(decade)}</text>'
                )

    y_tick_count = 6
    for index in range(y_tick_count):
        y_value = y_min + (y_max - y_min) * index / (y_tick_count - 1)
        y = top + plot_height - plot_height * index / (y_tick_count - 1)
        elements.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#d8d8d8" stroke-width="1.2"/>'
        )
        elements.append(
            f'<text x="{left - 13}" y="{y + 6:.1f}" text-anchor="end" font-family="{font}" font-size="16">{y_value:.2f}</text>'
        )

    connected_points = " ".join(f"{x:.1f},{y:.1f}" for x, y in map(coordinates, representative_results))
    elements.extend(
        [
            f'<polyline points="{connected_points}" fill="none" stroke="{gray}" stroke-width="2.4"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#2f2f2f" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#2f2f2f" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2}" y="{top + plot_height + 74}" text-anchor="middle" font-family="{font}" font-size="22">Peak learning rate</text>',
            f'<text x="29" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 29 {top + plot_height / 2})" font-family="{font}" font-size="22">Best validation loss observed</text>',
        ]
    )

    for result in representative_results:
        x, y = coordinates(result)
        if result["diverged"]:
            elements.extend(
                [
                    f'<line x1="{x - 7:.1f}" y1="{y - 7:.1f}" x2="{x + 7:.1f}" y2="{y + 7:.1f}" stroke="{red}" stroke-width="4"/>',
                    f'<line x1="{x - 7:.1f}" y1="{y + 7:.1f}" x2="{x + 7:.1f}" y2="{y - 7:.1f}" stroke="{red}" stroke-width="4"/>',
                ]
            )
        elif result["stage"] == "full":
            elements.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="{blue}" stroke="white" stroke-width="1.5"/>'
            )
        else:
            elements.append(
                f'<polygon points="{x:.1f},{y - 10:.1f} {x - 9:.1f},{y + 8:.1f} {x + 9:.1f},{y + 8:.1f}" fill="{orange}" stroke="white" stroke-width="1.5"/>'
            )

    best_full = min(full_results, key=lambda result: result["best_valid_loss"])
    best_x, best_y = coordinates(best_full)
    elements.extend(
        [
            f'<line x1="{best_x:.1f}" y1="{top}" x2="{best_x:.1f}" y2="{top + plot_height}" stroke="{blue}" stroke-width="2" stroke-dasharray="7 5" opacity="0.82"/>',
            f'<line x1="{best_x + 55:.1f}" y1="{best_y - 63:.1f}" x2="{best_x + 6:.1f}" y2="{best_y - 5:.1f}" stroke="{blue}" stroke-width="2"/>',
            f'<text x="{best_x + 58:.1f}" y="{best_y - 72:.1f}" font-family="{font}" font-size="18" fill="{blue}">best full run</text>',
            f'<text x="{best_x + 58:.1f}" y="{best_y - 49:.1f}" font-family="{font}" font-size="18" fill="{blue}">peak LR = {best_full["max_lr"]:.3g}</text>',
            f'<text x="{best_x + 58:.1f}" y="{best_y - 26:.1f}" font-family="{font}" font-size="18" fill="{blue}">valid loss = {best_full["best_valid_loss"]:.3f}</text>',
        ]
    )

    if high_lr_probes:
        worst_high_lr = max(high_lr_probes, key=lambda result: result["max_lr"])
        high_x, high_y = coordinates(worst_high_lr)
        text_x = min(high_x - 35, left + plot_width - 230)
        text_y = top + 47
        elements.extend(
            [
                f'<line x1="{text_x + 195:.1f}" y1="{text_y + 7:.1f}" x2="{high_x - 5:.1f}" y2="{high_y - 7:.1f}" stroke="#9d8e90" stroke-width="2"/>',
                f'<text x="{text_x:.1f}" y="{text_y:.1f}" font-family="{font}" font-size="18" fill="#9b3340">higher-loss probe regime</text>',
            ]
        )

    # The lowest-LR probe is high on this experiment's curve, so place the
    # legend at lower-left to keep every marker visible.
    legend_x, legend_y = left + 12, top + plot_height - 116
    legend_height = 102 + int(any(result["diverged"] for result in representative_results)) * 31
    elements.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="255" height="{legend_height}" rx="5" fill="white" stroke="#cfcfcf" opacity="0.92"/>',
            f'<circle cx="{legend_x + 24}" cy="{legend_y + 29}" r="8" fill="{blue}"/><text x="{legend_x + 55}" y="{legend_y + 35}" font-family="{font}" font-size="17">1000-step training run</text>',
            f'<polygon points="{legend_x + 24},{legend_y + 51} {legend_x + 15},{legend_y + 69} {legend_x + 33},{legend_y + 69}" fill="{orange}"/><text x="{legend_x + 55}" y="{legend_y + 67}" font-family="{font}" font-size="17">short stability probe</text>',
            f'<text x="{width / 2}" y="{height - 28}" text-anchor="middle" font-family="{font}" font-size="15" fill="#4a4a4a">Setup: AdamW (weight decay=0.01), cosine decay to 0.1× peak LR, gradient clipping max norm=1.0, TinyStories subset.</text>',
            "</svg>",
        ]
    )
    return "\n".join(elements)


def _report_markdown(dataset: dict, results: list[dict], selected_lrs: list[float]) -> str:
    probe_results = [result for result in results if result["stage"] == "probe"]
    full_results = [result for result in results if result["stage"] == "full"]
    ranked_full = sorted(full_results, key=lambda result: result["best_valid_loss"])
    best = ranked_full[0]
    runner_up = ranked_full[1] if len(ranked_full) > 1 else None
    probe_best = min(probe_results, key=lambda result: result["best_valid_loss"])
    runner_up_sentence = ""
    if runner_up is not None:
        runner_up_gap = runner_up["best_valid_loss"] - best["best_valid_loss"]
        runner_up_sentence = (
            f"第二名 `{runner_up['max_lr']:.3g}` 的最佳验证损失为 "
            f"`{runner_up['best_valid_loss']:.4f}`，只比最佳值高 `{runner_up_gap:.4f}`，"
            "仍是值得在完整数据上复核的候选。"
        )
    overfit_results = [
        result
        for result in full_results
        if result["best_step"] < result["final_step"] and result["final_valid_loss"] > result["best_valid_loss"]
    ]
    overfit_sentence = ""
    if overfit_results:
        best_steps = ", ".join(str(result["best_step"]) for result in overfit_results)
        overfit_sentence = (
            f"{len(overfit_results)} 个完整运行的最佳验证步数依次为 {best_steps}；此后训练损失继续下降，"
            f"而验证损失回升到 {min(result['final_valid_loss'] for result in overfit_results):.4f} 或更高，"
            f"表明这个 {dataset['train_tokens']:,}-token 子集已开始过拟合。"
            "后续应增大训练数据覆盖并保存 best-validation checkpoint，而不是只保留最后一步。"
        )
    rows = []
    for result in sorted(results, key=lambda item: (item["stage"], item["max_lr"])):
        rows.append(
            f"| {result['stage']} | {result['max_lr']:.3g} | {result['num_steps']} | "
            f"{result['best_valid_loss']:.4f} @ {result['best_step']} | "
            f"{result['final_valid_loss']:.4f} | {result['elapsed_seconds']:.1f}s | "
            f"{'yes' if result['diverged'] else 'no'} |"
        )
    selected = ", ".join(f"{learning_rate:.3g}" for learning_rate in selected_lrs)
    return f"""# Learning Rate Sweep — 2026-09-02

## 目标

参考 `Hurricane0698/TransformerLM-from-scratch` 的两阶段 TinyStories 学习率扫描：先用较短 probe
淘汰较差学习率，再对候选学习率使用更长预算，并以验证集逐 token loss 选择最佳值。

## 固定配置

- 数据：TinyStories train 前 {dataset["train_character_limit"]:,} 字符、valid 前 {dataset["valid_character_limit"]:,} 字符
- Token 数：train {dataset["train_tokens"]:,}，valid {dataset["valid_tokens"]:,}
- 模型：4 layers, d_model=512, 16 heads, d_ff=1344, context=256, vocab=10,000
- Batch size：16
- 优化器：本项目 AdamW，betas=(0.9, 0.95)，weight decay=0.01
- 调度：线性 warmup + cosine decay，min_lr=max_lr×0.1
- 梯度裁剪：1.0
- 随机种子：42；所有 LR 使用相同初始化与数据采样顺序
- 粗扫：300 steps；完整阶段：1000 steps；每次验证平均 {EVAL_ITERS} 个 batch

## 两阶段选择

粗扫学习率：{", ".join(f"{learning_rate:.3g}" for learning_rate in LEARNING_RATES)}。

根据粗扫 best validation loss 自动选择：{selected}。

## 结果

| Stage | Max LR | Steps | Best valid loss | Final valid loss | Time | Diverged |
|---|---:|---:|---:|---:|---:|:---:|
{chr(10).join(rows)}

## 结论

本次预算和数据子集下，最佳学习率为 **{best["max_lr"]:.3g}**，最佳验证损失为
**{best["best_valid_loss"]:.4f}**（step {best["best_step"]}），最终验证损失为
**{best["final_valid_loss"]:.4f}**。

粗扫阶段最好的学习率是 `{probe_best["max_lr"]:.3g}`，但完整阶段由 `{best["max_lr"]:.3g}` 反超，
说明短 probe 适合淘汰明显较差的范围，不能替代完整预算比较。{runner_up_sentence}

{overfit_sentence}

该结果用于选择后续完整 TinyStories 训练的候选学习率，不应与参考仓库 10,000-step、完整数据结果作绝对数值比较。
建议用最佳值及其相邻值做更长训练后再最终定参。

## 复现

```bash
modal run modal_lr_sweep.py
```

原始逐点评估记录位于 `results.json` 和 `runs/*.json`，汇总见 `summary.csv`，参考风格图见
`lr_sweep_research_summary.svg`。相同文件也持久化在 Modal Volume 的
`/experiments/lr_sweep_comparison_20260902/`。
"""


def _build_artifacts(dataset: dict, results: list[dict], selected_lrs: list[float], config: dict) -> dict[str, str]:
    artifacts = {
        "README.md": _report_markdown(dataset, results, selected_lrs),
        "config.json": json.dumps(config, ensure_ascii=False, indent=2),
        "results.json": json.dumps(results, ensure_ascii=False, indent=2),
        "summary.csv": _summary_csv(results),
        "lr_sweep_summary.svg": _summary_svg(results),
        "lr_sweep_research_summary.svg": _summary_svg(results),
    }
    for result in results:
        lr_tag = f"{result['max_lr']:.2e}".replace("+", "").replace(".", "p")
        artifacts[f"runs/{result['stage']}_lr_{lr_tag}.json"] = json.dumps(result, ensure_ascii=False, indent=2)
    return artifacts


@app.local_entrypoint()
def main(
    probe_steps: int = 300,
    full_steps: int = 1_000,
    top_k: int = 3,
    force_prepare: bool = False,
) -> None:
    if probe_steps <= 0 or full_steps <= 0:
        raise ValueError("probe_steps and full_steps must be positive")
    if not 1 <= top_k <= len(LEARNING_RATES):
        raise ValueError(f"top_k must be between 1 and {len(LEARNING_RATES)}")

    dataset = prepare_sweep_dataset.remote(force=force_prepare)
    probe_configs = [
        {
            "stage": "probe",
            "max_lr": learning_rate,
            "min_lr": learning_rate * MIN_LR_RATIO,
            "num_steps": probe_steps,
            "warmup_steps": max(1, probe_steps // 10),
            "eval_interval": max(1, probe_steps // 6),
        }
        for learning_rate in LEARNING_RATES
    ]
    print(f"Starting {len(probe_configs)} probe runs")
    probe_results = list(run_learning_rate.map(probe_configs))
    stable_probes = [
        result for result in probe_results if not result["diverged"] and math.isfinite(result["best_valid_loss"])
    ]
    if len(stable_probes) < top_k:
        raise RuntimeError(f"Only {len(stable_probes)} stable probe runs; cannot select top {top_k}")
    selected_lrs = [
        result["max_lr"] for result in sorted(stable_probes, key=lambda result: result["best_valid_loss"])[:top_k]
    ]
    print(f"Selected learning rates for full stage: {selected_lrs}")

    full_configs = [
        {
            "stage": "full",
            "max_lr": learning_rate,
            "min_lr": learning_rate * MIN_LR_RATIO,
            "num_steps": full_steps,
            "warmup_steps": max(1, full_steps // 10),
            "eval_interval": max(1, full_steps // 10),
        }
        for learning_rate in selected_lrs
    ]
    full_results = list(run_learning_rate.map(full_configs))
    results = [*probe_results, *full_results]
    experiment_config = {
        "learning_rates": LEARNING_RATES,
        "selected_learning_rates": selected_lrs,
        "probe_steps": probe_steps,
        "full_steps": full_steps,
        "top_k": top_k,
        "min_lr_ratio": MIN_LR_RATIO,
        "random_seed": RANDOM_SEED,
        "model": MODEL_CONFIG,
        "batch_size": BATCH_SIZE,
        "max_grad_norm": MAX_GRAD_NORM,
        "weight_decay": WEIGHT_DECAY,
        "betas": BETAS,
        "eval_iters": EVAL_ITERS,
        "dataset": dataset,
    }
    artifacts = _build_artifacts(dataset, results, selected_lrs, experiment_config)

    LOCAL_EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)
    for relative_path, contents in artifacts.items():
        destination = LOCAL_EXPERIMENT_DIR / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    remote_path = persist_artifacts.remote(artifacts)

    best_full = min(full_results, key=lambda result: result["best_valid_loss"])
    print("\nLearning-rate sweep completed")
    print(f"Best max_lr: {best_full['max_lr']:.3g}")
    print(f"Best validation loss: {best_full['best_valid_loss']:.4f} at step {best_full['best_step']}")
    print(f"Local results: {LOCAL_EXPERIMENT_DIR}")
    print(f"Modal Volume results: {remote_path}")
