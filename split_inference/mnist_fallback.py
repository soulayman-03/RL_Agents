from __future__ import annotations

import gzip
import os
import struct
import urllib.request
from typing import Callable, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


_MNIST_MIRRORS = (
    # Official LeCun host is intentionally not used (often blocked/unavailable).
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
)
_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(url: str, dst_path: str) -> None:
    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    if os.path.exists(dst_path):
        return
    tmp = dst_path + ".tmp"
    urllib.request.urlretrieve(url, tmp)
    os.replace(tmp, dst_path)

def _download_from_mirrors(fname: str, dst_path: str) -> None:
    last_err: Exception | None = None
    for base in _MNIST_MIRRORS:
        try:
            _download(base + fname, dst_path)
            return
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Failed to download {fname} from mirrors: {_MNIST_MIRRORS}") from last_err


def _read_idx_images_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"Bad MNIST image magic {magic} for {path}")
        data = f.read(n * rows * cols)
    arr = np.frombuffer(data, dtype=np.uint8).reshape(n, rows, cols)
    return arr


def _read_idx_labels_gz(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"Bad MNIST label magic {magic} for {path}")
        data = f.read(n)
    arr = np.frombuffer(data, dtype=np.uint8)
    return arr


def _ensure_mnist_files(root: str) -> Tuple[str, str, str, str]:
    raw_dir = os.path.join(root, "MNIST_fallback", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    paths = {}
    for k, fname in _FILES.items():
        p = os.path.join(raw_dir, fname)
        _download_from_mirrors(fname, p)
        paths[k] = p

    return paths["train_images"], paths["train_labels"], paths["test_images"], paths["test_labels"]


def mnist_tensor_transform(x: torch.Tensor) -> torch.Tensor:
    # x: (1,28,28) float in [0,1]
    mean = 0.1307
    std = 0.3081
    return (x - mean) / std


class MNISTDataset(Dataset):
    def __init__(
        self,
        root: str = "./data",
        train: bool = True,
        download: bool = True,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.root = str(root)
        self.train = bool(train)
        self.transform = transform

        processed_dir = os.path.join(self.root, "MNIST_fallback", "processed")
        os.makedirs(processed_dir, exist_ok=True)
        split = "train" if self.train else "test"
        cache_path = os.path.join(processed_dir, f"{split}.npz")

        if os.path.exists(cache_path):
            data = np.load(cache_path)
            self.images = data["images"]
            self.labels = data["labels"]
            return

        if not download:
            raise FileNotFoundError(
                f"MNIST not found at {cache_path} and download=False. "
                "Install torchvision or run with download=True."
            )

        train_images_p, train_labels_p, test_images_p, test_labels_p = _ensure_mnist_files(self.root)
        if self.train:
            images = _read_idx_images_gz(train_images_p)
            labels = _read_idx_labels_gz(train_labels_p)
        else:
            images = _read_idx_images_gz(test_images_p)
            labels = _read_idx_labels_gz(test_labels_p)

        np.savez_compressed(cache_path, images=images, labels=labels)
        self.images = images
        self.labels = labels

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        img = self.images[int(idx)]  # (28,28) uint8
        x = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # (1,28,28)
        if self.transform is not None:
            x = self.transform(x)
        y = int(self.labels[int(idx)])
        return x, y
