"""
plot_snake.py
=============
Port of plot_snake.m (AC Kwan 170720).

Plots a pseudo-color snake (trial-by-trial heatmap) of neural signal,
with each trial normalized to [0,1] or displayed in z-score units.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_snake(
    signal: np.ndarray,
    t: np.ndarray,
    label: str = "",
    sort_time: tuple = None,
    zscore_mode: bool = False,
    ax: plt.Axes = None,
) -> plt.Axes:
    """
    Parameters
    ----------
    signal      : (n_trials, T) float array — dF/F or z-score values
    t           : (T,) time vector in seconds, relative to cue onset
    label       : subplot title
    sort_time   : (t_start, t_end) window for sorting trials by mean activity.
                  None → original order (MATLAB default: not sorted).
    zscore_mode : If True, skip 0-1 normalization (signal is already z-scored).
    ax          : matplotlib Axes; created if None.

    Returns
    -------
    ax : Axes with the snake plot drawn.
    """
    if ax is None:
        _, ax = plt.subplots()

    signal = np.asarray(signal, float)
    if signal.ndim == 1:
        signal = signal[np.newaxis, :]
    n_trials, T = signal.shape

    # ---- Normalize each trial to [0,1], or keep as z-score ----
    norm = np.full_like(signal, np.nan)
    for j in range(n_trials):
        row = signal[j, :]
        if zscore_mode:
            norm[j, :] = row
        else:
            lo = np.nanmin(row)
            hi = np.nanmax(row)
            norm[j, :] = (row - lo) / (hi - lo) if hi > lo else np.zeros_like(row)

    # ---- Optional: sort by mean activity in sort_time window ----
    if sort_time is not None:
        t0, t1 = sort_time
        win = (t >= t0) & (t <= t1)
        order = np.argsort(np.nanmean(norm[:, win], axis=1))
        norm = norm[order, :]

    # ---- Colour limits (mirrors MATLAB prctile-based caxis) ----
    if zscore_mode:
        ampl = max(
            abs(np.nanpercentile(norm, 95)),
            abs(np.nanpercentile(norm, 5)),
        )
        vmin, vmax = -ampl, ampl
    else:
        vmin = np.nanpercentile(norm, 5)
        vmax = np.nanpercentile(norm, 95)

    # ---- Plot ----
    ax.imshow(
        norm,
        aspect="auto",
        extent=[float(t[0]), float(t[-1]), n_trials, 0],
        cmap="OrRd",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.axvline(0, color="white", linewidth=1.5)
    ax.set_xlabel("Time from stimulus (s)")
    ax.set_ylabel("Trials")
    ax.set_title(label)

    return ax
