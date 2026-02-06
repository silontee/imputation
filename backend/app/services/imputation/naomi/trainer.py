"""NAOMI training loop (notebook-style, teacher forcing).

GAN components intentionally omitted.
"""

from __future__ import annotations

from typing import Optional, Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from app.services.imputation.naomi.model import NAOMIModel


def build_training_batches(
    data: np.ndarray,
    row_complete_mask: np.ndarray,
    window_size: int = 50,
) -> np.ndarray:
    """Build sliding window batches from fully observed rows."""
    window_size = max(2, int(window_size))
    data_batch = []
    window = []

    for idx, row in enumerate(data):
        if not row_complete_mask[idx]:
            window = []
            continue
        window.append(row)
        if len(window) == window_size:
            data_batch.append(window.copy())
            window.pop(0)

    if not data_batch:
        return np.empty((0, window_size, data.shape[1]), dtype=data.dtype)

    return np.array(data_batch)


def build_fallback_batches(
    data: np.ndarray,
    window_size: int = 50,
) -> np.ndarray:
    """Fallback: build sliding windows from all rows (no completeness check)."""
    window_size = max(2, int(window_size))
    data_batch = []
    window = []
    for row in data:
        window.append(row)
        if len(window) == window_size:
            data_batch.append(window.copy())
            window.pop(0)
    if not data_batch:
        return np.empty((0, window_size, data.shape[1]), dtype=data.dtype)
    return np.array(data_batch)


def run_epoch(
    train: bool,
    model: NAOMIModel,
    exp_data: torch.Tensor,
    clip: float,
    optimizer: Optional[optim.Optimizer] = None,
    batch_size: int = 64,
    num_missing: Optional[int] = None,
    teacher_forcing: bool = True,
) -> float:
    losses = []
    n = exp_data.shape[0]
    if n == 0:
        return 0.0

    batch_size = min(batch_size, n)
    inds = np.random.permutation(n)
    i = 0

    while i + batch_size <= n:
        ind = torch.from_numpy(inds[i : i + batch_size]).long()
        i += batch_size
        data = exp_data[ind]

        # (batch, time, x) -> (time, batch, x)
        data = data.transpose(0, 1)
        ground_truth = data.clone()
        seq_len = data.shape[0]

        if num_missing is None:
            num_missing = np.random.randint(seq_len * 4 // 5, seq_len)
        missing_list = torch.from_numpy(
            np.random.choice(np.arange(1, seq_len), num_missing, replace=False)
        ).long()

        data[missing_list] = 0.0
        has_value = torch.ones(seq_len, data.shape[1], 1, device=data.device)
        has_value[missing_list] = 0.0
        data = torch.cat([has_value, data], 2)

        if teacher_forcing:
            batch_loss = model(data, ground_truth)
        else:
            data_list = [data[j : j + 1] for j in range(seq_len)]
            samples = model.sample(data_list)
            batch_loss = torch.mean((ground_truth - samples).pow(2))

        if train and optimizer is not None:
            optimizer.zero_grad()
            batch_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

        losses.append(float(batch_loss.detach().cpu().numpy()))

    return float(np.mean(losses)) if losses else 0.0


def train_naomi(
    data: np.ndarray,
    row_complete_mask: np.ndarray,
    hidden_dim: int = 64,
    num_resolutions: Optional[int] = None,
    highest: Optional[int] = None,
    epochs: int = 50,
    lr: float = 1e-3,
    window_size: int = 50,
    batch_size: int = 64,
    n_layers: int = 2,
    clip: float = 10.0,
    teacher_forcing: bool = True,
    device: str = "cpu",
    progress_callback: Optional[Callable[[int, int, float, NAOMIModel], None]] = None,
) -> NAOMIModel:
    """Train NAOMI using notebook-style loop."""
    if highest is not None:
        highest = max(1, int(highest))
    elif num_resolutions is not None:
        highest = 2 ** max(0, int(num_resolutions) - 1)
    else:
        # heuristic: largest power of two <= window_size/2
        highest = 1
        limit = max(1, window_size // 2)
        while highest * 2 <= limit:
            highest *= 2

    model = NAOMIModel(
        input_dim=data.shape[1],
        rnn_dim=hidden_dim,
        n_layers=n_layers,
        highest=highest,
        stochastic=False,
    ).to(device)
    model.train()

    exp_data_np = build_training_batches(data, row_complete_mask, window_size=window_size)
    if exp_data_np.shape[0] == 0:
        # Fallback when no fully observed windows are available
        exp_data_np = build_fallback_batches(data, window_size=window_size)
    exp_data = torch.tensor(exp_data_np, dtype=torch.float32, device=device)

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    for epoch in range(1, epochs + 1):
        loss_val = run_epoch(
            True,
            model,
            exp_data,
            clip,
            optimizer=optimizer,
            batch_size=batch_size,
            teacher_forcing=teacher_forcing,
        )
        if progress_callback is not None:
            progress_callback(epoch, epochs, loss_val, model)

    model.eval()
    return model
