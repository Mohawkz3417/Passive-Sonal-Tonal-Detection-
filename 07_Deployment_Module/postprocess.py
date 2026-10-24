"""
postprocess.py
================
Deployment-side post-processing chain (Modules 6-7 of the proposal):

    SegFormer prob map -> CFAR verification -> Morphological Processing
    -> Connected Component Analysis -> per-tonal blob descriptors

CFAR (Constant False Alarm Rate) is included as an independent, classical
confirmation stage on top of the network's probability map: a detection is
only kept if it is *both* judged tonal by the network *and* stands out
above the locally-estimated noise floor by a CFAR-derived margin. This
combination (network + classical detector, "ensemble detection" in the
proposal) substantially reduces false alarms compared to using either
method alone.
"""
from __future__ import annotations
import numpy as np
from scipy import ndimage as ndi
from skimage import measure


# ---------------------------------------------------------------------------
# CFAR (Cell-Averaging, 1-D along frequency axis, applied per time frame)
# ---------------------------------------------------------------------------
def ca_cfar_1d(mag_spec: np.ndarray, num_train: int = 16, num_guard: int = 4,
                pfa: float = 1e-2, valid_bin_range: tuple[int, int] | None = None) -> np.ndarray:
    """
    Cell-averaging CFAR applied independently down each time-frame (column)
    of the spectrogram, across the frequency axis. Operates on the *power*
    (magnitude^2) spectrogram since the standard CA-CFAR threshold factor
    alpha = N*(Pfa^(-1/N) - 1) assumes exponentially-distributed cells,
    which holds for power (not magnitude) under a complex-Gaussian noise
    model.

    mag_spec : (n_freq, n_frames) non-negative *magnitude* spectrogram
    valid_bin_range : optional (lo, hi) bin indices (see
        signal_processing.passband_bin_range) -- bins outside this range
        are forced to 0 rather than CFAR-tested. Strongly recommended:
        outside the band-pass filter's passband the "noise" is a
        deterministic near-zero attenuated tail, not the random process
        CFAR's statistics assume, and left untested it produces spurious
        always-on false detections right at the passband edge.
    returns: binary mask (n_freq, n_frames), 1 = exceeds adaptive threshold
    """
    spec_full = mag_spec.astype(np.float64) ** 2
    n_freq, n_frames = spec_full.shape
    lo_bin, hi_bin = valid_bin_range if valid_bin_range is not None else (0, n_freq)
    alpha = num_train * (pfa ** (-1.0 / num_train) - 1.0)  # CA-CFAR threshold factor
    mask = np.zeros_like(spec_full, dtype=np.float32)

    # crop to the passband *before* padding, so training/guard windows never
    # pull in stopband bins (see valid_bin_range docstring above)
    spec = spec_full[lo_bin:hi_bin]
    n_valid = spec.shape[0]
    if n_valid == 0:
        return mask

    win = num_train + num_guard
    padded = np.pad(spec, ((win, win), (0, 0)), mode="edge")
    csum = np.cumsum(padded, axis=0)

    for k in range(n_valid):
        i = k + win  # index in padded array
        lead_sum = csum[i - num_guard] - csum[i - win]           # leading training cells
        lag_sum = csum[i + win + 1] - csum[i + num_guard + 1] if i + win + 1 < csum.shape[0] \
            else csum[-1] - csum[i + num_guard + 1]
        noise_est = (lead_sum + lag_sum) / (2 * num_train)
        threshold = alpha * noise_est
        mask[lo_bin + k, :] = (spec[k, :] > threshold).astype(np.float32)
    return mask


# ---------------------------------------------------------------------------
# Morphological processing
# ---------------------------------------------------------------------------
def morphological_clean(mask: np.ndarray, open_size=(2, 2), close_size=(3, 3),
                         min_area: int = 6, min_elongation: float = 1.5) -> np.ndarray:
    """
    (a) opening  - removes isolated single-pixel noise
    (b) closing  - fills small gaps inside a genuine tonal line
    (c) area threshold - discards components smaller than `min_area` px
    (d) elongation filter - keeps only elongated (line-like) blobs,
        rejecting round/irregular noise blobs (proposal 4(d): "Noise Blob Removal")
    """
    m = mask.astype(bool)
    m = ndi.binary_opening(m, structure=np.ones(open_size))
    m = ndi.binary_closing(m, structure=np.ones(close_size))

    labeled = measure.label(m, connectivity=2)  # 8-connectivity
    cleaned = np.zeros_like(m)
    for region in measure.regionprops(labeled):
        if region.area < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        h, w = max(1, maxr - minr), max(1, maxc - minc)
        elongation = max(h, w) / min(h, w)
        # a real tonal is elongated mostly along the *time* axis (columns)
        if elongation < min_elongation and region.area < min_area * 3:
            continue
        cleaned[labeled == region.label] = True
    return cleaned.astype(np.float32)


# ---------------------------------------------------------------------------
# Connected Component Analysis -> tonal blob descriptors
# ---------------------------------------------------------------------------
def connected_components(mask: np.ndarray, weight_img: np.ndarray | None = None) -> list[dict]:
    """
    Extract one descriptor dict per connected tonal blob.

    weight_img (optional): intensity image (e.g. PCEN or magnitude spectrogram)
    used for intensity-weighted centroid / energy.
    """
    labeled = measure.label(mask.astype(bool), connectivity=2)
    props = measure.regionprops(labeled, intensity_image=weight_img)
    out = []
    for r in props:
        minr, minc, maxr, maxc = r.bbox
        if weight_img is not None:
            sub = weight_img[minr:maxr, minc:maxc]
            sub_mask = (labeled[minr:maxr, minc:maxc] == r.label)
            ys, xs = np.nonzero(sub_mask)
            weights = sub[ys, xs]
            freq_bin_centroid = minr + (ys * weights).sum() / (weights.sum() + 1e-12)
            energy = float(weights.sum())
        else:
            freq_bin_centroid = r.centroid[0]
            energy = float(r.area)
        out.append({
            "label": r.label,
            "bbox": (minr, minc, maxr, maxc),   # (freq_lo, t_lo, freq_hi, t_hi) in bin/frame index
            "area": int(r.area),
            "freq_bin_centroid": float(freq_bin_centroid),
            "time_frame_centroid": float(r.centroid[1]),
            "width_frames": int(maxc - minc),
            "height_bins": int(maxr - minr),
            "energy": energy,
        })
    return out
