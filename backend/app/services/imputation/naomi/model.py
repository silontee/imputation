"""NAOMI: Non-Autoregressive Multiresolution Sequence Imputation (NeurIPS 2019).

Notebook-style implementation (teacher-forcing training + multiresolution decoders).
GAN components intentionally omitted.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Callable, Tuple

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers (stochastic path kept for completeness, but unused by default)
# ---------------------------------------------------------------------------

def nll_gauss(mean: torch.Tensor, std: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    pi = torch.tensor(np.pi, device=mean.device, dtype=mean.dtype)
    nll_element = (x - mean).pow(2) / std.pow(2) + 2 * torch.log(std) + torch.log(2 * pi)
    return 0.5 * torch.sum(nll_element)


def reparam_sample_gauss(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mean)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class NAOMIModel(nn.Module):
    """Notebook-style NAOMI implementation.

    Args:
        input_dim: D
        rnn_dim: hidden size
        n_layers: LSTM layers
        highest: maximum step size (power of two)
        stochastic: enable stochastic decoding (unused by default)
    """

    def __init__(
        self,
        input_dim: int,
        rnn_dim: int,
        n_layers: int,
        highest: int,
        stochastic: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.rnn_dim = rnn_dim
        self.n_layers = n_layers
        self.highest = max(1, int(highest))
        self.stochastic = stochastic

        # NOTE: The original NAOMI paper uses GRUs. This project uses LSTMs.
        self.lstm = nn.LSTM(self.input_dim, self.rnn_dim, self.n_layers)
        self.back_lstm = nn.LSTM(self.input_dim + 1, self.rnn_dim, self.n_layers)

        # Multiresolution decoders (powers of two)
        self.networks: Dict[str, nn.ModuleDict] = {}
        step = 1
        while step <= self.highest:
            curr_level: Dict[str, nn.Module] = {}
            curr_level["dec"] = nn.Sequential(
                nn.Linear(2 * self.rnn_dim, self.rnn_dim),
                nn.ReLU(),
            )
            curr_level["mean"] = nn.Linear(self.rnn_dim, self.input_dim)
            if self.stochastic:
                curr_level["std"] = nn.Sequential(
                    nn.Linear(self.rnn_dim, self.input_dim),
                    nn.Softplus(),
                )
            self.networks[str(step)] = nn.ModuleDict(curr_level)
            step *= 2

        self.networks = nn.ModuleDict(self.networks)

    # ---- training forward (teacher forcing) ------------------------------

    def forward(self, data: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        """Compute multiresolution loss.

        Args:
            data: (T, B, D+1) = [mask, x]
            ground_truth: (T, B, D)
        """
        seq_len = data.shape[0]
        batch = data.shape[1]
        device = data.device

        h = torch.zeros(self.n_layers, batch, self.rnn_dim, device=device)
        c = torch.zeros(self.n_layers, batch, self.rnn_dim, device=device)
        h_back = torch.zeros(self.n_layers, batch, self.rnn_dim, device=device)
        c_back = torch.zeros(self.n_layers, batch, self.rnn_dim, device=device)

        loss = 0.0
        h_back_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        count = 0

        # backward encoding over masked data
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = (h_back, c_back)
            state_t = data[t]
            _, (h_back, c_back) = self.back_lstm(state_t.unsqueeze(0), (h_back, c_back))

        # forward encoding over ground truth (teacher forcing)
        for t in range(seq_len):
            state_t = ground_truth[t]
            _, (h, c) = self.lstm(state_t.unsqueeze(0), (h, c))
            count += 1
            for l in self.networks.keys():
                step_size = int(l)
                if t + 2 * step_size <= seq_len:
                    next_t = ground_truth[t + step_size]
                    h_back, _ = h_back_dict[t + 2 * step_size]

                    curr_level = self.networks[str(step_size)]
                    dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1))
                    dec_mean = curr_level["mean"](dec_t)

                    if self.stochastic:
                        dec_std = curr_level["std"](dec_t)
                        loss = loss + nll_gauss(dec_mean, dec_std, next_t)
                    else:
                        loss = loss + torch.sum((dec_mean - next_t).pow(2))

        if count == 0:
            return torch.tensor(0.0, device=device)
        return loss / count / batch

    # ---- inference -------------------------------------------------------

    @torch.no_grad()
    def impute(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        progress_callback: Optional[Callable[[int, int, torch.Tensor, Optional[np.ndarray]], None]] = None,
    ) -> torch.Tensor:
        """Impute missing values.

        Args:
            x: (T, D)
            mask: (T, 1)
        """
        data_list = self._build_data_list(x, mask)
        x_imputed = self.sample(data_list, progress_callback=progress_callback)
        return x_imputed.squeeze(1)

    # ---- helpers ---------------------------------------------------------

    def _build_data_list(self, x: torch.Tensor, mask: torch.Tensor) -> List[torch.Tensor]:
        seq_len = x.shape[0]
        data_list: List[torch.Tensor] = []
        for t in range(seq_len):
            has_value = mask[t].view(1, 1, 1).to(x.device)
            value = (x[t] * mask[t]).view(1, 1, -1)
            data_list.append(torch.cat([has_value, value], dim=2))
        return data_list

    # ---- sampling --------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        data_list: List[torch.Tensor],
        progress_callback: Optional[Callable[[int, int, torch.Tensor, Optional[np.ndarray]], None]] = None,
    ) -> torch.Tensor:
        seq_len = len(data_list)
        device = data_list[0].device

        filled = np.array([bool(x[0, 0, 0].item()) for x in data_list], dtype=bool)
        total_missing = int((~filled).sum())
        step = 0

        def _emit(step_data_list: List[torch.Tensor]) -> None:
            nonlocal step
            step += 1
            if progress_callback is None:
                return
            x_imputed = torch.cat(step_data_list, dim=0)[:, :, 1:].squeeze(1)
            progress_callback(step, max(1, total_missing), x_imputed, filled.copy())

        h = torch.zeros(self.n_layers, 1, self.rnn_dim, device=device)
        c = torch.zeros(self.n_layers, 1, self.rnn_dim, device=device)
        h_back = torch.zeros(self.n_layers, 1, self.rnn_dim, device=device)
        c_back = torch.zeros(self.n_layers, 1, self.rnn_dim, device=device)

        h_back_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for t in range(seq_len - 1, 0, -1):
            h_back_dict[t + 1] = (h_back, c_back)
            state_t = data_list[t]
            _, (h_back, c_back) = self.back_lstm(state_t, (h_back, c_back))

        curr_p = 0
        _, (h, c) = self.lstm(data_list[curr_p][:, :, 1:], (h, c))

        while curr_p < seq_len - 1:
            if data_list[curr_p + 1][0, 0, 0] == 1:
                curr_p += 1
                _, (h, c) = self.lstm(data_list[curr_p][:, :, 1:], (h, c))
                continue

            next_p = curr_p + 1
            while next_p < seq_len and data_list[next_p][0, 0, 0] == 0:
                next_p += 1

            step_size = 1
            while curr_p + 2 * step_size <= next_p and step_size <= self.highest:
                step_size *= 2
            step_size = step_size // 2

            self._interpolate(data_list, curr_p, h, h_back_dict, step_size)
            filled[curr_p + step_size] = True
            _emit(data_list)

        x_imputed = torch.cat(data_list, dim=0)[:, :, 1:]
        return x_imputed

    def _interpolate(
        self,
        data_list: List[torch.Tensor],
        curr_p: int,
        h: torch.Tensor,
        h_back_dict: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        step_size: int,
    ) -> None:
        h_back, _ = h_back_dict[curr_p + 2 * step_size]
        curr_level = self.networks[str(step_size)]

        dec_t = curr_level["dec"](torch.cat([h[-1], h_back[-1]], 1))
        dec_mean = curr_level["mean"](dec_t)
        if self.stochastic:
            dec_std = curr_level["std"](dec_t)
            state_t = reparam_sample_gauss(dec_mean, dec_std)
        else:
            state_t = dec_mean

        added_state = state_t.unsqueeze(0)
        has_value = torch.ones(added_state.shape[0], added_state.shape[1], 1, device=added_state.device)
        added_state = torch.cat([has_value, added_state], 2)

        if step_size > 1:
            right = curr_p + step_size
            left = curr_p + step_size // 2
            h_back, c_back = h_back_dict[right + 1]
            _, (h_back, c_back) = self.back_lstm(added_state, (h_back, c_back))
            h_back_dict[right] = (h_back, c_back)

            zeros = torch.zeros(
                added_state.shape[0], added_state.shape[1], self.input_dim + 1, device=added_state.device
            )
            for i in range(right - 1, left - 1, -1):
                _, (h_back, c_back) = self.back_lstm(zeros, (h_back, c_back))
                h_back_dict[i] = (h_back, c_back)

        data_list[curr_p + step_size] = added_state
