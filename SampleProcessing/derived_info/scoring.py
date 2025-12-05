# derived_info/scoring.py
from __future__ import annotations
import numpy as np
from typing import Tuple
from keras.models import load_model  # or from tensorflow.keras.models import load_model

def load_autoencoders(paths: Tuple[str, str]):
    """
    Return a tuple of loaded keras models for (ae01, ae04).
    """
    ae01 = load_model(paths[0])
    ae04 = load_model(paths[1])
    return ae01, ae04

def calculate_batch_loss(autoencoder, images: np.ndarray) -> np.ndarray:
    """
    Per-event MSE between input and recon (averaged over axes 1,2).
    images: (n_events, n_jets, n_features)
    returns: (n_events,)
    """
    recon = autoencoder.predict(images, verbose=0)
    if recon.shape != images.shape:
        raise ValueError(f"Shape mismatch: recon {recon.shape} vs images {images.shape}")
    return np.mean((images - recon) ** 2, axis=(1, 2))

def calculate_H_met(events: np.ndarray, ht_values: np.ndarray) -> np.ndarray:
    """
    Compute H_MET = ||sum_j pT_j [cos phi, sin phi]|| with pT denormalized by HT per event.
    events: (n_events, n_jets, 4) columns: Eta,Phi,Pt(normed),NPV
    ht_values: (n_events,)
    """
    pt_norm = events[:, :, 2]
    phi     = events[:, :, 1]
    pt = pt_norm * ht_values[:, None]  # de-normalize pT per event
    px = pt * np.cos(phi)
    py = pt * np.sin(phi)
    return np.sqrt(np.sum(px, axis=1) ** 2 + np.sum(py, axis=1) ** 2)

def count_njets(events: np.ndarray) -> np.ndarray:
    """Number of jets with pt>0 per event."""
    return np.sum(events[:, :, 2] > 0, axis=1)


from typing import Union
import numpy as np

ArrayLike = Union[np.ndarray]

def batch_mse_scores(model, X: ArrayLike, batch_size: int = 2048) -> np.ndarray:
    """
    Vectorized version of `calculate_batch_score` from testAE.py.

    Parameters
    ----------
    model : keras.Model
        Trained autoencoder taking X -> X_hat.
    X : np.ndarray
        Input of shape (N, ..., ...) – e.g. (N, 8, 4).
    batch_size : int
        Batch size for model.predict.

    Returns
    -------
    scores : np.ndarray, shape (N,)
        Per-event mean squared error.
    """
    X = np.asarray(X)
    n = X.shape[0]
    scores = []

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        x_batch = X[start:end]

        # Same as: reconstructed_images_ = autoencoder_.predict(images_, verbose=0)
        x_hat = model.predict(x_batch, verbose=0)

        # Sanity check (same as your assert)
        if x_hat.shape != x_batch.shape:
            raise ValueError(
                f"Shape mismatch in batch_mse_scores: "
                f"pred {x_hat.shape} vs input {x_batch.shape}"
            )

        # Original code: np.mean(np.square(images_ - reconstructed_images_), axis=(1,2))
        # Generalized to any number of non-batch dims:
        diff = x_batch - x_hat
        # collapse all but batch dim, then mean over features
        diff = diff.reshape(diff.shape[0], -1)
        mse_batch = np.mean(diff ** 2, axis=1)

        scores.append(mse_batch)

    return np.concatenate(scores, axis=0)


def percentile_threshold(values: ArrayLike, q: float) -> float:
    """
    q-th percentile of the array, e.g. q=99.75.

    This mirrors:
        percen_9975 = np.percentile(bkg_test_scores, 99.75)
    """
    values = np.asarray(values)
    return float(np.percentile(values, q))


def pass_rate_above(values: ArrayLike, threshold: float) -> float:
    """
    Percentage (0–100) of entries strictly above `threshold`.

    Mirrors:
        AA_passed = 100 * np.sum(AA_test_scores > percen_9975) / len(AA_test_scores)
    """
    values = np.asarray(values)
    if values.size == 0:
        return 0.0
    return 100.0 * float((values > threshold).sum()) / float(values.size)