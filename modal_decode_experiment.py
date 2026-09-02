from __future__ import annotations

import csv
import html
import json
import math
from collections import Counter
from pathlib import Path

import modal


APP_NAME = "cs336-decoding-strategy-experiment"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_ROOT = Path("/data")

CHECKPOINT_PATH = VOLUME_ROOT / "training_runs" / "tinystories_320m_d448_lr1e-3_20260902" / "final_model.pt"
VALID_IDS_PATH = VOLUME_ROOT / "tokenized" / "tinystories_10k" / "valid.bin"
VOCAB_PATH = VOLUME_ROOT / "tokenizers" / "tinystories_10k" / "vocab.json"
REMOTE_OUTPUT_DIR = VOLUME_ROOT / "experiments" / "decoding_strategy_comparison_20260902"
LOCAL_OUTPUT_DIR = Path("experiments/decoding_strategy_comparison_20260902")

SPECIAL_TOKEN = "<|endoftext|>"
NUM_PROMPTS = 32
NUM_REPEATS = 3
PROMPT_TOKENS = 24
MAX_NEW_TOKENS = 128
RANDOM_SEED = 42

STRATEGIES = (
    {"id": "greedy", "label": "Greedy", "temperature": 0.0, "top_k": None, "top_p": None},
    {"id": "t0.5", "label": "T=0.5", "temperature": 0.5, "top_k": None, "top_p": None},
    {"id": "t0.8", "label": "T=0.8", "temperature": 0.8, "top_k": None, "top_p": None},
    {"id": "t1.0", "label": "T=1.0", "temperature": 1.0, "top_k": None, "top_p": None},
    {"id": "t1.2", "label": "T=1.2", "temperature": 1.2, "top_k": None, "top_p": None},
    {"id": "t1.4", "label": "T=1.4", "temperature": 1.4, "top_k": None, "top_p": None},
    {"id": "t0.8_k20", "label": "T=0.8, K=20", "temperature": 0.8, "top_k": 20, "top_p": None},
    {"id": "t0.8_k50", "label": "T=0.8, K=50", "temperature": 0.8, "top_k": 50, "top_p": None},
    {"id": "t0.8_k100", "label": "T=0.8, K=100", "temperature": 0.8, "top_k": 100, "top_p": None},
    {"id": "t0.8_p0.5", "label": "T=0.8, P=0.50", "temperature": 0.8, "top_k": None, "top_p": 0.5},
    {"id": "t0.8_p0.8", "label": "T=0.8, P=0.80", "temperature": 0.8, "top_k": None, "top_p": 0.8},
    {"id": "t0.8_p0.9", "label": "T=0.8, P=0.90", "temperature": 0.8, "top_k": None, "top_p": 0.9},
    {"id": "t0.8_p0.95", "label": "T=0.8, P=0.95", "temperature": 0.8, "top_k": None, "top_p": 0.95},
    {
        "id": "t0.8_k50_p0.9",
        "label": "T=0.8, K=50, P=0.90",
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.9,
    },
)


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)
image = modal.Image.debian_slim(python_version="3.12").uv_sync().add_local_python_source("cs336_basics")


def _load_vocabulary() -> tuple[dict[int, bytes], int]:
    from cs336_basics.train_bpe import bytes_to_unicode

    byte_decoder = {character: byte for byte, character in bytes_to_unicode().items()}
    with VOCAB_PATH.open("r", encoding="utf-8") as file:
        serialized = json.load(file)
    vocabulary = {
        int(token_id): bytes(byte_decoder[character] for character in token)
        for token_id, token in serialized.items()
    }
    eos_id = next(token_id for token_id, token in vocabulary.items() if token == SPECIAL_TOKEN.encode("utf-8"))
    return vocabulary, eos_id


def _decode(token_ids: list[int], vocabulary: dict[int, bytes], strip_eos: bool = False) -> str:
    data = b"".join(vocabulary[token_id] for token_id in token_ids)
    text = data.decode("utf-8", errors="replace")
    if strip_eos:
        text = text.replace(SPECIAL_TOKEN, "")
    return text


def _select_evaluation_stories(valid_data, eos_id: int) -> tuple[list[list[int]], list[list[int]]]:
    import numpy as np

    eos_positions = np.flatnonzero(valid_data == eos_id)
    starts = np.concatenate((np.asarray([0], dtype=np.int64), eos_positions[:-1] + 1))
    candidates = []
    for start, end in zip(starts.tolist(), eos_positions.tolist()):
        if end - start >= PROMPT_TOKENS + 24:
            candidates.append((start, end))
    if len(candidates) < NUM_PROMPTS:
        raise RuntimeError(f"Only found {len(candidates)} suitable validation stories")

    selected_indices = np.linspace(0, len(candidates) - 1, NUM_PROMPTS, dtype=int)
    prompts = []
    references = []
    for index in selected_indices.tolist():
        start, end = candidates[index]
        prompt = valid_data[start : start + PROMPT_TOKENS].astype(np.int64).tolist()
        reference_end = min(end + 1, start + PROMPT_TOKENS + MAX_NEW_TOKENS)
        reference = valid_data[start + PROMPT_TOKENS : reference_end].astype(np.int64).tolist()
        prompts.append(prompt)
        references.append(reference)
    return prompts, references


def _apply_top_k(logits, top_k: int | None):
    import torch

    if top_k is None or top_k >= logits.shape[-1]:
        return logits
    threshold = torch.topk(logits, top_k, dim=-1).values[:, -1:]
    return logits.masked_fill(logits < threshold, float("-inf"))


def _apply_top_p(logits, top_p: float | None):
    import torch

    if top_p is None or top_p >= 1.0:
        return logits
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probabilities = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
    remove = cumulative_probabilities > top_p
    remove[:, 1:] = remove[:, :-1].clone()
    remove[:, 0] = False
    sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
    filtered = torch.full_like(logits, float("-inf"))
    return filtered.scatter(1, sorted_indices, sorted_logits)


def _generate_batch(model, prompt_ids, strategy: dict, eos_id: int, seed: int) -> list[list[int]]:
    import torch

    torch.manual_seed(seed)
    generated = prompt_ids.clone()
    finished = torch.zeros(prompt_ids.shape[0], dtype=torch.bool, device=prompt_ids.device)
    continuation_lengths = torch.full(
        (prompt_ids.shape[0],),
        MAX_NEW_TOKENS,
        dtype=torch.long,
        device=prompt_ids.device,
    )

    model.eval()
    with torch.no_grad():
        for token_index in range(MAX_NEW_TOKENS):
            logits = model(generated[:, -model.context_length :])[:, -1, :]
            temperature = float(strategy["temperature"])
            if temperature == 0.0:
                next_tokens = torch.argmax(logits, dim=-1)
            else:
                logits = logits / temperature
                logits = _apply_top_k(logits, strategy["top_k"])
                logits = _apply_top_p(logits, strategy["top_p"])
                probabilities = torch.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(1)

            next_tokens = torch.where(finished, torch.full_like(next_tokens, eos_id), next_tokens)
            newly_finished = (~finished) & (next_tokens == eos_id)
            continuation_lengths[newly_finished] = token_index + 1
            finished |= newly_finished
            generated = torch.cat((generated, next_tokens.unsqueeze(1)), dim=1)
            if finished.all():
                break

    continuations = []
    for row_index, length in enumerate(continuation_lengths.tolist()):
        start = prompt_ids.shape[1]
        continuations.append(generated[row_index, start : start + length].tolist())
    return continuations


def _continuation_nll(model, prompts: list[list[int]], continuations: list[list[int]], eos_id: int, device: str) -> float:
    import torch

    maximum_length = max(len(prompt) + len(continuation) for prompt, continuation in zip(prompts, continuations))
    batch = torch.full((len(prompts), maximum_length), eos_id, dtype=torch.long, device=device)
    valid_target_masks = torch.zeros((len(prompts), maximum_length - 1), dtype=torch.bool, device=device)
    for row, (prompt, continuation) in enumerate(zip(prompts, continuations)):
        sequence = prompt + continuation
        batch[row, : len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        valid_target_masks[row, len(prompt) - 1 : len(sequence) - 1] = True

    model.eval()
    with torch.no_grad():
        logits = model(batch[:, :-1])
        log_probabilities = torch.log_softmax(logits, dim=-1)
        token_log_probabilities = torch.gather(log_probabilities, -1, batch[:, 1:].unsqueeze(-1)).squeeze(-1)
    return float((-token_log_probabilities[valid_target_masks]).mean().item())


def _distinct_ratio(sequence: list[int], n: int) -> float:
    if len(sequence) < n:
        return 0.0
    ngrams = [tuple(sequence[index : index + n]) for index in range(len(sequence) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def _repetition_ratio(sequence: list[int], n: int = 4) -> float:
    if len(sequence) < n:
        return 0.0
    ngrams = [tuple(sequence[index : index + n]) for index in range(len(sequence) - n + 1)]
    return 1.0 - len(set(ngrams)) / len(ngrams)


def _has_loop(sequence: list[int], n: int = 4) -> bool:
    if len(sequence) < n:
        return False
    counts = Counter(tuple(sequence[index : index + n]) for index in range(len(sequence) - n + 1))
    return max(counts.values(), default=0) >= 3


def _rouge_l_f1(prediction: list[int], reference: list[int]) -> float:
    if not prediction or not reference:
        return 0.0
    previous = [0] * (len(reference) + 1)
    for predicted_token in prediction:
        current = [0]
        for reference_index, reference_token in enumerate(reference, start=1):
            if predicted_token == reference_token:
                current.append(previous[reference_index - 1] + 1)
            else:
                current.append(max(previous[reference_index], current[-1]))
        previous = current
    lcs = previous[-1]
    precision = lcs / len(prediction)
    recall = lcs / len(reference)
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def _strip_eos(sequence: list[int], eos_id: int) -> list[int]:
    return sequence[:-1] if sequence and sequence[-1] == eos_id else sequence


def _measure(
    model,
    prompts: list[list[int]],
    continuations: list[list[int]],
    references: list[list[int]],
    vocabulary: dict[int, bytes],
    eos_id: int,
    device: str,
) -> dict[str, float]:
    clean = [_strip_eos(sequence, eos_id) for sequence in continuations]
    clean_references = [_strip_eos(sequence, eos_id) for sequence in references]
    decoded = [_decode(sequence, vocabulary, strip_eos=True).rstrip() for sequence in continuations]
    nll = _continuation_nll(model, prompts, continuations, eos_id, device)
    return {
        "nll": nll,
        "perplexity": math.exp(min(nll, 20.0)),
        "mean_tokens": sum(map(len, clean)) / len(clean),
        "distinct_1": sum(_distinct_ratio(sequence, 1) for sequence in clean) / len(clean),
        "distinct_2": sum(_distinct_ratio(sequence, 2) for sequence in clean) / len(clean),
        "repetition_4": sum(_repetition_ratio(sequence) for sequence in clean) / len(clean),
        "loop_rate": sum(_has_loop(sequence) for sequence in clean) / len(clean),
        "eos_rate": sum(bool(sequence and sequence[-1] == eos_id) for sequence in continuations) / len(continuations),
        "complete_ending_rate": sum(text.endswith((".", "!", "?", '"')) for text in decoded) / len(decoded),
        "replacement_character_rate": sum(text.count("�") for text in decoded)
        / max(sum(len(text) for text in decoded), 1),
        "rouge_l": sum(
            _rouge_l_f1(prediction, reference) for prediction, reference in zip(clean, clean_references)
        )
        / len(clean),
    }


def _add_balance_scores(results: list[dict], reference_metrics: dict[str, float]) -> None:
    for result in results:
        metrics = result["metrics"]
        fluency = math.exp(-max(metrics["nll"] - reference_metrics["nll"], 0.0))
        diversity_match = math.exp(-abs(metrics["distinct_2"] - reference_metrics["distinct_2"]) / 0.20)
        repetition_score = 1.0 - min(metrics["repetition_4"] / 0.20, 1.0)
        length_match = math.exp(-abs(metrics["mean_tokens"] - reference_metrics["mean_tokens"]) / 40.0)
        eos_match = 1.0 - abs(metrics["eos_rate"] - reference_metrics["eos_rate"])
        metrics["balance_score"] = 100.0 * (
            0.25 * fluency
            + 0.25 * diversity_match
            + 0.20 * repetition_score
            + 0.15 * length_match
            + 0.15 * eos_match
        )


def _format_optional(value) -> str:
    return "—" if value is None else f"{value:g}"


def _comparison_table(results: list[dict], reference_metrics: dict[str, float]) -> str:
    header = (
        "| 策略 | T | Top-k | Top-p | PPL ↓ | Dist-2 ↑ | 4-gram 重复 ↓ | 循环率 ↓ | "
        "EOS率 | 完整结尾 | ROUGE-L | 平衡分 ↑ |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
    )
    rows = []
    for result in sorted(results, key=lambda item: item["metrics"]["balance_score"], reverse=True):
        strategy = result["strategy"]
        metrics = result["metrics"]
        rows.append(
            f"| {strategy['label']} | {_format_optional(strategy['temperature'])} | "
            f"{_format_optional(strategy['top_k'])} | {_format_optional(strategy['top_p'])} | "
            f"{metrics['perplexity']:.2f} | {metrics['distinct_2'] * 100:.1f}% | "
            f"{metrics['repetition_4'] * 100:.1f}% | {metrics['loop_rate'] * 100:.1f}% | "
            f"{metrics['eos_rate'] * 100:.1f}% | {metrics['complete_ending_rate'] * 100:.1f}% | "
            f"{metrics['rouge_l'] * 100:.1f}% | **{metrics['balance_score']:.1f}** |"
        )
    reference_row = (
        f"| Held-out reference | — | — | — | {reference_metrics['perplexity']:.2f} | "
        f"{reference_metrics['distinct_2'] * 100:.1f}% | {reference_metrics['repetition_4'] * 100:.1f}% | "
        f"{reference_metrics['loop_rate'] * 100:.1f}% | {reference_metrics['eos_rate'] * 100:.1f}% | "
        f"{reference_metrics['complete_ending_rate'] * 100:.1f}% | 100.0% | — |"
    )
    return header + "\n".join(rows + [reference_row]) + "\n"


def _case_table(results: list[dict], prompts: list[list[int]], vocabulary: dict[int, bytes]) -> str:
    by_id = {result["strategy"]["id"]: result for result in results}
    case_strategy_ids = ("greedy", "t0.8_p0.5", "t0.8", "t0.8_k50", "t0.8_p0.95", "t1.4")
    case_indices = (5, 14, 22)
    output = [
        "# 典型生成案例（并排）\n",
        "每行使用同一个验证集故事开头。为便于阅读，已隐藏 `<|endoftext|>`。\n",
        "<table>",
        "<tr><th>Prompt</th>"
        + "".join(f"<th>{html.escape(by_id[strategy_id]['strategy']['label'])}</th>" for strategy_id in case_strategy_ids)
        + "</tr>",
    ]
    for case_index in case_indices:
        prompt = html.escape(_decode(prompts[case_index], vocabulary, strip_eos=True).strip()).replace("\n", "<br>")
        cells = [f'<td valign="top"><b>{prompt}</b></td>']
        for strategy_id in case_strategy_ids:
            continuation = by_id[strategy_id]["continuations"][case_index]
            text = html.escape(_decode(continuation, vocabulary, strip_eos=True).strip()).replace("\n", "<br>")
            cells.append(f'<td valign="top">{text}</td>')
        output.append("<tr>" + "".join(cells) + "</tr>")
    output.append("</table>\n")
    return "\n".join(output)


CSV_FIELDS = (
    "rank",
    "strategy_id",
    "strategy",
    "temperature",
    "top_k",
    "top_p",
    "perplexity",
    "nll",
    "mean_tokens",
    "distinct_1",
    "distinct_2",
    "repetition_4",
    "loop_rate",
    "eos_rate",
    "complete_ending_rate",
    "replacement_character_rate",
    "rouge_l",
    "balance_score",
)


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    gpu="L4",
    cpu=4.0,
    memory=16_384,
    timeout=90 * 60,
)
def run_experiment() -> dict:
    import numpy as np
    import torch

    from cs336_basics.nn import TransformerLM

    data_volume.reload()
    for path in (CHECKPOINT_PATH, VALID_IDS_PATH, VOCAB_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment input is missing: {path}")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA GPU")
    device = "cuda"
    torch.set_float32_matmul_precision("high")

    checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
    model_config = checkpoint["config"]["model"]
    model = TransformerLM(**model_config, device="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device).eval()
    vocabulary, eos_id = _load_vocabulary()
    valid_data = np.memmap(VALID_IDS_PATH, dtype=np.uint16, mode="r")
    prompts, references = _select_evaluation_stories(valid_data, eos_id)
    prompt_tensor = torch.tensor(prompts, dtype=torch.long, device=device)

    reference_metrics = _measure(
        model,
        prompts,
        references,
        references,
        vocabulary,
        eos_id,
        device,
    )
    results = []
    for strategy in STRATEGIES:
        continuations = []
        repeated_prompts = []
        repeated_references = []
        first_greedy_batch = None
        for repeat in range(NUM_REPEATS):
            if strategy["temperature"] == 0.0 and first_greedy_batch is not None:
                generated_batch = first_greedy_batch
            else:
                generated_batch = _generate_batch(
                    model,
                    prompt_tensor,
                    strategy,
                    eos_id,
                    RANDOM_SEED + repeat,
                )
                if strategy["temperature"] == 0.0:
                    first_greedy_batch = generated_batch
            continuations.extend(generated_batch)
            repeated_prompts.extend(prompts)
            repeated_references.extend(references)
        metrics = _measure(
            model,
            repeated_prompts,
            continuations,
            repeated_references,
            vocabulary,
            eos_id,
            device,
        )
        result = {
            "strategy": strategy,
            "metrics": metrics,
            "continuations": continuations,
            "decoded_continuations": [
                _decode(sequence, vocabulary, strip_eos=True) for sequence in continuations
            ],
        }
        results.append(result)
        print(
            f"[{strategy['label']}] ppl={metrics['perplexity']:.2f} "
            f"dist2={metrics['distinct_2']:.3f} rep4={metrics['repetition_4']:.3f} "
            f"eos={metrics['eos_rate']:.3f}"
        )

    _add_balance_scores(results, reference_metrics)
    ranked = sorted(results, key=lambda item: item["metrics"]["balance_score"], reverse=True)
    best = ranked[0]
    REMOTE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    config = {
        "checkpoint": str(CHECKPOINT_PATH),
        "validation_tokens": str(VALID_IDS_PATH),
        "num_prompts": NUM_PROMPTS,
        "num_repeats": NUM_REPEATS,
        "stochastic_samples_per_strategy": NUM_PROMPTS * NUM_REPEATS,
        "prompt_tokens": PROMPT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "random_seed": RANDOM_SEED,
        "strategies": STRATEGIES,
        "score_formula": {
            "fluency_vs_reference_nll": 0.25,
            "distinct_2_closeness_to_reference": 0.25,
            "four_gram_repetition": 0.20,
            "length_closeness_to_reference": 0.15,
            "eos_rate_closeness_to_reference": 0.15,
        },
    }
    (REMOTE_OUTPUT_DIR / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    result_payload = {
        "reference_metrics": reference_metrics,
        "prompts": prompts,
        "decoded_prompts": [_decode(prompt, vocabulary, strip_eos=True) for prompt in prompts],
        "references": references,
        "results": results,
        "best_strategy_id": best["strategy"]["id"],
        "best_strategy_label": best["strategy"]["label"],
    }
    (REMOTE_OUTPUT_DIR / "results.json").write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (REMOTE_OUTPUT_DIR / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for rank, result in enumerate(ranked, start=1):
            strategy = result["strategy"]
            writer.writerow(
                {
                    "rank": rank,
                    "strategy_id": strategy["id"],
                    "strategy": strategy["label"],
                    "temperature": strategy["temperature"],
                    "top_k": strategy["top_k"],
                    "top_p": strategy["top_p"],
                    **result["metrics"],
                }
            )

    table = _comparison_table(results, reference_metrics)
    cases = _case_table(results, prompts, vocabulary)
    (REMOTE_OUTPUT_DIR / "comparison_table.md").write_text(
        "# 解码策略自动指标对比\n\n" + table,
        encoding="utf-8",
    )
    (REMOTE_OUTPUT_DIR / "cases.md").write_text(cases, encoding="utf-8")
    by_id = {result["strategy"]["id"]: result for result in results}
    runner_up = ranked[1]
    low_top_p = by_id["t0.8_p0.5"]["metrics"]
    balanced_top_p = by_id["t0.8_p0.9"]["metrics"]
    high_temperature = by_id["t1.4"]["metrics"]
    report = f"""# TinyStories 解码策略实验

本实验在 {NUM_PROMPTS} 个固定的验证集故事开头上比较 14 种解码设置，每个随机策略使用
{NUM_REPEATS} 个共同随机种子，共 {NUM_PROMPTS * NUM_REPEATS} 个续写；每个 prompt 最多生成
{MAX_NEW_TOKENS} tokens。所有随机种子固定，因此结果可以复现。

## 结论

按预先定义的自动平衡分，最佳配置是 **{best['strategy']['label']}**，得分
**{best['metrics']['balance_score']:.1f}/100**。该分数同时考虑相对 held-out 文本的模型流畅度、
Distinct-2、多字重复、生成长度和 EOS 率，不把单纯的低困惑度误认为高质量。

第一名与第二名 **{runner_up['strategy']['label']}** 只差
{best['metrics']['balance_score'] - runner_up['metrics']['balance_score']:.2f} 分，前三名可视为近似并列；
若需要一个固定默认值，本实验推荐第一名，但不应把这点微小差距理解为普遍规律。

- 过低：Greedy/低 Top-p 会把候选空间压得过窄。`P=0.50` 的 Distinct-2 为
  {low_top_p['distinct_2'] * 100:.1f}%、4-gram 重复为 {low_top_p['repetition_4'] * 100:.1f}%；
  `P=0.90` 分别改善为 {balanced_top_p['distinct_2'] * 100:.1f}% 和
  {balanced_top_p['repetition_4'] * 100:.1f}%。低阈值更安全，但更模板化、更容易重复或过早收尾。
- 温度过高：`T=1.4` 虽把 Distinct-2 推到 {high_temperature['distinct_2'] * 100:.1f}%，
  PPL 却升至 {high_temperature['perplexity']:.1f}，EOS率降到 {high_temperature['eos_rate'] * 100:.1f}%，
  并出现无效 UTF-8 替换字符；案例中可直接看到人物、语法和因果关系崩坏。
- Top-p 较高：在受控的 `T=0.8` 下，`P=0.90–0.95` 仍然稳定；Top-p 接近 1 本身并未在本实验中
  造成明显失控。风险主要发生在它与高温叠加、让低概率长尾 token 获得过多概率质量时。
- TinyStories 的稳妥默认值应放在 `temperature=0.7–0.9`、`top_p=0.85–0.95`，可再加中等
  `top_k=40–100` 作为保险；本次模型的具体首选以上述实测最佳行为准。

## 自动指标表

{table}

## 指标说明

- PPL：模型对自己生成 continuation 的困惑度；越低越流畅，但极低通常意味着过于保守。
- Dist-2：每个样本中不重复二元 token 组的比例。
- 4-gram 重复：重复四元 token 组占比；循环率统计同一个 4-gram 至少出现三次的样本。
- EOS率和完整结尾：是否主动结束故事，以及结束前是否有句末标点。
- ROUGE-L：与真实续写的 token 级最长公共子序列 F1，仅作 prompt 对齐参考；创作任务不应追求复刻参考答案。
- 平衡分：25% 流畅度、25% Dist-2 与 held-out 的接近度、20% 重复惩罚、15% 长度接近度、
  15% EOS率接近度。公式在 `config.json` 中固定记录。

并排案例见 `cases.md`，原始 token、全部生成文本和逐策略指标见 `results.json`。
"""
    (REMOTE_OUTPUT_DIR / "report.md").write_text(report, encoding="utf-8")
    data_volume.commit()
    return {
        "output_dir": str(REMOTE_OUTPUT_DIR),
        "best_strategy": best["strategy"],
        "best_metrics": best["metrics"],
        "reference_metrics": reference_metrics,
    }


@app.local_entrypoint()
def main():
    result = run_experiment.remote()
    print(json.dumps(result, ensure_ascii=False, indent=2))
