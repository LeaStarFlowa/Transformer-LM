from __future__ import annotations

import json
import time
from pathlib import Path

import modal


APP_NAME = "cs336-training-smoke-test"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_ROOT = Path("/data")

TRAIN_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-train.txt"
VALID_TEXT_PATH = VOLUME_ROOT / "TinyStoriesV2-GPT4-valid.txt"
TOKENIZER_DIR = VOLUME_ROOT / "tokenizers" / "tinystories_10k"
OUTPUT_DIR = VOLUME_ROOT / "smoke_test"

VOCAB_SIZE = 10_000
CONTEXT_LENGTH = 64
D_MODEL = 128
NUM_LAYERS = 1
NUM_HEADS = 4
D_FF = 384
BATCH_SIZE = 4
DEFAULT_STEPS = 100
SPECIAL_TOKENS = ["<|endoftext|>"]


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

    return BPETokenizer(vocab, merges, SPECIAL_TOKENS)


def _assert_optimizer_states_equal(first: dict, second: dict) -> None:
    import torch

    if first["param_groups"] != second["param_groups"]:
        raise AssertionError("Loaded optimizer parameter groups do not match the saved optimizer")
    if first["state"].keys() != second["state"].keys():
        raise AssertionError("Loaded optimizer state keys do not match the saved optimizer")

    for parameter_id, first_state in first["state"].items():
        second_state = second["state"][parameter_id]
        if first_state.keys() != second_state.keys():
            raise AssertionError(f"Optimizer state fields differ for parameter {parameter_id}")
        for key, first_value in first_state.items():
            second_value = second_state[key]
            if torch.is_tensor(first_value):
                if not torch.equal(first_value.cpu(), second_value.cpu()):
                    raise AssertionError(f"Optimizer tensor {key!r} differs for parameter {parameter_id}")
            elif first_value != second_value:
                raise AssertionError(f"Optimizer value {key!r} differs for parameter {parameter_id}")


@app.function(
    image=image,
    volumes={str(VOLUME_ROOT): data_volume},
    gpu="T4",
    cpu=2.0,
    memory=8_192,
    timeout=1_800,
)
def run_smoke_test(steps: int = DEFAULT_STEPS) -> dict:
    import numpy as np
    import torch

    from cs336_basics.checkpointing import load_checkpoint, save_checkpoint
    from cs336_basics.data import get_batch
    from cs336_basics.losses import cross_entropy
    from cs336_basics.nn import TransformerLM
    from cs336_basics.optimizer import AdamW, clip_gradient_norm
    from cs336_basics.scheduler import get_lr_cosine_schedule

    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}")
    for path in (TRAIN_TEXT_PATH, VALID_TEXT_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required corpus not found: {path}")

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_float32_matmul_precision("high")
    if not torch.cuda.is_available():
        raise RuntimeError("Modal allocated no CUDA GPU")
    device = torch.device("cuda")
    print(f"Device: {torch.cuda.get_device_name(0)}")

    tokenizer = _load_tokenizer()
    if len(tokenizer.vocab) != VOCAB_SIZE:
        raise AssertionError(f"Expected a {VOCAB_SIZE}-token vocabulary, got {len(tokenizer.vocab)}")

    # This is deliberately an overfit smoke test. Reusing one short passage
    # makes a broken training loop obvious within only 100 optimizer steps.
    with TRAIN_TEXT_PATH.open("r", encoding="utf-8") as file:
        train_text = file.read(250_000)
    story_start = train_text.find("Once upon a time")
    if story_start == -1:
        story_start = 0
    train_token_ids = tokenizer.encode(train_text[story_start : story_start + 8_000])
    if len(train_token_ids) < CONTEXT_LENGTH + 1:
        raise RuntimeError("The tokenized smoke-test passage is too short")

    fixed_sequence = torch.tensor(
        train_token_ids[: CONTEXT_LENGTH + 1],
        dtype=torch.long,
        device=device,
    )
    train_x = fixed_sequence[:-1].unsqueeze(0).repeat(BATCH_SIZE, 1)
    train_y = fixed_sequence[1:].unsqueeze(0).repeat(BATCH_SIZE, 1)

    with VALID_TEXT_PATH.open("r", encoding="utf-8") as file:
        valid_text = file.read(100_000)
    valid_token_ids = np.asarray(tokenizer.encode(valid_text), dtype=np.uint16)
    if len(valid_token_ids) < CONTEXT_LENGTH + 1:
        raise RuntimeError("The tokenized validation passage is too short")
    valid_x, valid_y = get_batch(valid_token_ids, BATCH_SIZE, CONTEXT_LENGTH, str(device))

    model_kwargs = {
        "vocab_size": VOCAB_SIZE,
        "context_length": CONTEXT_LENGTH,
        "d_model": D_MODEL,
        "num_layers": NUM_LAYERS,
        "num_heads": NUM_HEADS,
        "d_ff": D_FF,
        "rope_theta": 10_000.0,
        "device": device,
        "use_rms_norm": True,
        "norm_mode": "pre",
        "ffn_type": "swiglu",
    }
    model = TransformerLM(**model_kwargs)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        "Model: "
        f"layers={NUM_LAYERS}, d_model={D_MODEL}, heads={NUM_HEADS}, "
        f"d_ff={D_FF}, context={CONTEXT_LENGTH}, parameters={parameter_count:,}"
    )

    model.eval()
    with torch.no_grad():
        initial_train_loss = cross_entropy(model(train_x), train_y).item()
        initial_valid_loss = cross_entropy(model(valid_x), valid_y).item()
    print(f"Step 0/{steps}: train_loss={initial_train_loss:.4f}, val_loss={initial_valid_loss:.4f}")

    losses = []
    started_at = time.monotonic()
    warmup_steps = min(5, max(1, steps // 10))
    for step in range(1, steps + 1):
        learning_rate = get_lr_cosine_schedule(
            step,
            max_learning_rate=1e-3,
            min_learning_rate=1e-4,
            warmup_iters=warmup_steps,
            cosine_cycle_iters=steps,
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate

        model.train()
        logits = model(train_x)
        loss = cross_entropy(logits, train_y)
        optimizer.zero_grad()
        loss.backward()
        clip_gradient_norm(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(loss.item())

        if step == 1 or step % 10 == 0 or step == steps:
            print(f"Step {step}/{steps}: train_loss={loss.item():.4f}, lr={learning_rate:.2e}")

    model.eval()
    with torch.no_grad():
        final_train_logits = model(train_x)
        final_train_loss = cross_entropy(final_train_logits, train_y).item()
        final_valid_loss = cross_entropy(model(valid_x), valid_y).item()
    if not final_train_loss < initial_train_loss:
        raise AssertionError(f"Loss did not decrease: initial={initial_train_loss:.4f}, final={final_train_loss:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = OUTPUT_DIR / f"checkpoint_step_{steps}.pt"
    save_checkpoint(model, optimizer, steps, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")

    reloaded_model = TransformerLM(**model_kwargs)
    reloaded_optimizer = AdamW(reloaded_model.parameters(), lr=1e-3, weight_decay=0.1)
    loaded_iteration = load_checkpoint(checkpoint_path, reloaded_model, reloaded_optimizer)
    if loaded_iteration != steps:
        raise AssertionError(f"Expected checkpoint iteration {steps}, got {loaded_iteration}")

    for name, parameter in model.state_dict().items():
        if not torch.equal(parameter, reloaded_model.state_dict()[name]):
            raise AssertionError(f"Loaded model parameter does not match: {name}")
    _assert_optimizer_states_equal(optimizer.state_dict(), reloaded_optimizer.state_dict())

    reloaded_model.eval()
    with torch.no_grad():
        reloaded_logits = reloaded_model(train_x)
    if not torch.equal(final_train_logits, reloaded_logits):
        max_difference = (final_train_logits - reloaded_logits).abs().max().item()
        raise AssertionError(f"Reloaded model output differs; max absolute difference={max_difference}")
    print(f"Checkpoint loaded and verified at step {loaded_iteration}")

    prompt_token_count = min(12, CONTEXT_LENGTH // 2)
    prompt_ids = fixed_sequence[:prompt_token_count].unsqueeze(0)
    eos_token_id = tokenizer.byte_to_id.get(b"<|endoftext|>")
    generated_ids = reloaded_model.generate(
        prompt_ids,
        max_new_tokens=48,
        eos_token_id=eos_token_id,
        temperature=0.2,
        top_p=0.9,
    )[0].tolist()
    prompt = tokenizer.decode(prompt_ids[0].tolist())
    sample = tokenizer.decode(generated_ids)
    print("\n--- PROMPT ---")
    print(prompt)
    print("--- GENERATED SAMPLE ---")
    print(sample)
    print("--- END SAMPLE ---\n")

    elapsed_seconds = time.monotonic() - started_at
    result = {
        "passed": True,
        "device": torch.cuda.get_device_name(0),
        "steps": steps,
        "parameter_count": parameter_count,
        "initial_train_loss": initial_train_loss,
        "final_train_loss": final_train_loss,
        "loss_change": final_train_loss - initial_train_loss,
        "initial_valid_loss": initial_valid_loss,
        "final_valid_loss": final_valid_loss,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_iteration": loaded_iteration,
        "checkpoint_reload_verified": True,
        "prompt": prompt,
        "sample": sample,
        "elapsed_seconds": elapsed_seconds,
    }
    summary_path = OUTPUT_DIR / f"summary_step_{steps}.json"
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    data_volume.commit()
    print(f"Smoke-test summary saved: {summary_path}")
    return result


@app.local_entrypoint()
def main(steps: int = DEFAULT_STEPS) -> None:
    result = run_smoke_test.remote(steps=steps)
    print("\nModal smoke test PASSED")
    print(f"Loss: {result['initial_train_loss']:.4f} -> {result['final_train_loss']:.4f}")
    print(f"Checkpoint: {result['checkpoint_path']} (reload verified)")
    print(f"Sample: {result['sample']}")
