"""TOTEM VQVAE inference functions for encoding and decoding time series."""

from typing import Tuple

import numpy as np
import torch


def revintime2codes(
    revin_data: torch.Tensor,
    compression_factor: int,
    vqvae_encoder,
    vqvae_quantizer,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode time series data to discrete codes using VQVAE.

    Args:
        revin_data: Input tensor [bs x nvars x time_len]
        compression_factor: Temporal compression factor (typically 4)
        vqvae_encoder: VQVAE encoder module
        vqvae_quantizer: VQVAE quantizer module

    Returns:
        Tuple of:
            - codes: Quantized codes [bs, nvars, code_dim, compressed_time]
            - code_ids: Code indices [bs, nvars, compressed_time]
            - codebook: Embedding weights [num_codes, code_dim]
    """
    bs = revin_data.shape[0]
    nvar = revin_data.shape[1]
    T = revin_data.shape[2]
    compressed_time = int(T / compression_factor)

    with torch.no_grad():
        # Flatten: [bs * nvars, T]
        flat_revin = revin_data.reshape(-1, T)

        # Encode: [bs * nvars, code_dim, compressed_time]
        latent = vqvae_encoder(flat_revin.float(), compression_factor)

        # Quantize
        vq_loss, quantized, perplexity, embedding_weight, encoding_indices, encodings = vqvae_quantizer(latent)

        code_dim = quantized.shape[-2]

        # Reshape outputs
        codes = quantized.reshape(bs, nvar, code_dim, compressed_time)
        code_ids = encoding_indices.view(bs, nvar, compressed_time)

    return codes, code_ids, embedding_weight


def codes2timerevin(
    code_ids: torch.Tensor,
    codebook: torch.Tensor,
    compression_factor: int,
    vqvae_decoder,
) -> torch.Tensor:
    """Decode discrete codes back to time series.

    Args:
        code_ids: Code indices [bs x nvars x compressed_time]
        codebook: Embedding weights [num_codes, code_dim]
        compression_factor: Temporal compression factor
        vqvae_decoder: VQVAE decoder module

    Returns:
        Reconstructed time series [bs x time_len x nvars]
    """
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # One-hot encode and lookup from codebook
        one_hot_encodings = torch.zeros(
            int(bs * nvars * compressed_len), num_code_words, device=device
        )
        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device), 1)

        # Lookup quantized values
        quantized = torch.matmul(one_hot_encodings, codebook).view(input_shape)

        # Swap to [bs * nvars, code_dim, compressed_len]
        quantized_swapped = torch.swapaxes(quantized, 1, 2)

        # Decode: [bs * nvars, time_len]
        prediction_recon = vqvae_decoder(quantized_swapped.to(device), compression_factor)

        # Reshape: [bs, nvars, time_len]
        prediction_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])

        # Swap to [bs, time_len, nvars]
        predictions_revin_space = torch.swapaxes(prediction_reshaped, 1, 2)

    return predictions_revin_space


def apply_zscore_normalization(
    data: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Apply z-score normalization using only observed values.

    Args:
        data: Input data array
        mask: Boolean mask (True = observed, False = missing)

    Returns:
        Tuple of (normalized_data, mean, std)
    """
    observed = data[mask]
    if len(observed) == 0:
        return data.copy(), 0.0, 1.0

    mean = float(np.nanmean(observed))
    std = float(np.nanstd(observed))
    if std == 0 or np.isnan(std):
        std = 1.0

    normalized = (data - mean) / std
    return normalized, mean, std


def inverse_zscore(data: np.ndarray, mean: float, std: float) -> np.ndarray:
    """Inverse z-score normalization."""
    return data * std + mean


def apply_minmax_normalization(
    data: np.ndarray, mask: np.ndarray
) -> Tuple[np.ndarray, float, float]:
    """Apply min-max normalization using only observed values.

    Args:
        data: Input data array
        mask: Boolean mask (True = observed, False = missing)

    Returns:
        Tuple of (normalized_data, min_val, range_val)
    """
    observed = data[mask]
    if len(observed) == 0:
        return data.copy(), 0.0, 1.0

    min_val = float(np.nanmin(observed))
    max_val = float(np.nanmax(observed))
    range_val = max_val - min_val
    if range_val == 0 or np.isnan(range_val):
        range_val = 1.0

    normalized = (data - min_val) / range_val
    return normalized, min_val, range_val


def inverse_minmax(data: np.ndarray, min_val: float, range_val: float) -> np.ndarray:
    """Inverse min-max normalization."""
    return data * range_val + min_val
