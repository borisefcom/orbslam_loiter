from __future__ import annotations

import tarfile
from pathlib import Path


def ensure_orb_vocab(*, vocab_txt: Path, vocab_tar_gz: Path) -> Path:
    """
    Ensure ORBvoc.txt exists (extract from the provided .tar.gz if needed).
    """
    vocab_txt = Path(vocab_txt)
    vocab_tar_gz = Path(vocab_tar_gz)
    if vocab_txt.exists():
        return vocab_txt
    if not vocab_tar_gz.exists():
        raise FileNotFoundError(f"Missing vocabulary archive: {vocab_tar_gz}")

    vocab_txt.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(vocab_tar_gz, mode="r:gz") as tf:
        member = None
        for m in tf.getmembers():
            if str(m.name).replace("\\", "/").endswith("ORBvoc.txt"):
                member = m
                break
        if member is None:
            raise RuntimeError(f"ORBvoc.txt not found inside: {vocab_tar_gz}")
        tf.extract(member, path=vocab_txt.parent)

        extracted = vocab_txt.parent / Path(member.name).name
        if extracted != vocab_txt:
            extracted.replace(vocab_txt)
    return vocab_txt

