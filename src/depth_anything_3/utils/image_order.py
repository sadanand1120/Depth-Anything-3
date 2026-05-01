# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates

from pathlib import Path
from typing import Iterable, TypeVar


T = TypeVar("T")


def sort_image_sequence(paths: Iterable[T]) -> list[T]:
    """Sort pure numeric frame names numerically; otherwise use plain lexicographic order."""
    paths = list(paths)
    if paths and all(Path(path).stem.isdigit() for path in paths):
        return sorted(paths, key=lambda path: (int(Path(path).stem), str(path)))
    return sorted(paths, key=str)
