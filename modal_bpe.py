from __future__ import annotations

import time
from pathlib import Path

import modal


APP_NAME = "cs336-bpe"
VOLUME_NAME = "cs336-assignment1-data"
VOLUME_MOUNT_PATH = Path("/data")

DEFAULT_INPUT_FILENAME = "TinyStoriesV2-GPT4-train.txt"
DEFAULT_OUTPUT_DIR = "tokenizers/tinystories_10k"
DEFAULT_VOCAB_SIZE = 10_000
SPECIAL_TOKENS = ["<|endoftext|>"]


app = modal.App(APP_NAME)
data_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False)

# uv_sync() installs the dependencies pinned by pyproject.toml and uv.lock.
# It intentionally does not install this project itself, so upload the local
# cs336_basics package separately.
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_sync()
    .add_local_python_source("cs336_basics")
)


def _path_in_volume(relative_path: str) -> Path:
    """Turn a user-provided relative path into a safe path under /data."""
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Path must be relative to {VOLUME_MOUNT_PATH}: {relative_path!r}")
    return VOLUME_MOUNT_PATH / path


@app.function(
    image=image,
    volumes={str(VOLUME_MOUNT_PATH): data_volume},
    cpu=4.0,
    memory=65_536,
    timeout=86_400,
)
def train_bpe_remote(
    input_filename: str = DEFAULT_INPUT_FILENAME,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> dict[str, str | int | float]:
    from cs336_basics.train_bpe import save_tokenizer_files, train_bpe

    input_path = _path_in_volume(input_filename)
    output_path = _path_in_volume(output_dir)

    minimum_vocab_size = 256 + len(SPECIAL_TOKENS)
    if vocab_size < minimum_vocab_size:
        raise ValueError(f"vocab_size must be at least {minimum_vocab_size}, got {vocab_size}")
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Training corpus not found at {input_path}. "
            f"Check it with: modal volume ls {VOLUME_NAME} /"
        )

    input_size_gib = input_path.stat().st_size / (1024**3)
    print(f"Input: {input_path} ({input_size_gib:.2f} GiB)")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Special tokens: {SPECIAL_TOKENS}")
    print(f"Output: {output_path}")

    started_at = time.monotonic()
    vocab, merges = train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
    )
    save_tokenizer_files(vocab, merges, output_path)

    # Make the generated files immediately visible to later Modal Functions and
    # to `modal volume ls/get` on the local machine.
    data_volume.commit()

    elapsed_seconds = time.monotonic() - started_at
    vocab_path = output_path / "vocab.json"
    merges_path = output_path / "merges.txt"
    print(f"Finished in {elapsed_seconds / 60:.2f} minutes")
    print(f"Saved {vocab_path} ({vocab_path.stat().st_size:,} bytes)")
    print(f"Saved {merges_path} ({merges_path.stat().st_size:,} bytes)")

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "vocab_size": len(vocab),
        "num_merges": len(merges),
        "elapsed_seconds": elapsed_seconds,
    }


@app.local_entrypoint()
def main(
    input_filename: str = DEFAULT_INPUT_FILENAME,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    output_dir: str = DEFAULT_OUTPUT_DIR,
) -> None:
    result = train_bpe_remote.remote(
        input_filename=input_filename,
        vocab_size=vocab_size,
        output_dir=output_dir,
    )
    print("Remote BPE training result:")
    for key, value in result.items():
        print(f"  {key}: {value}")
