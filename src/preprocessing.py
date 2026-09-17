import numpy as np
import re
import os
import antropy as ant
import matplotlib.pyplot as plt
from scipy.signal import welch, hilbert
from scipy.stats import skew, kurtosis
import antropy as ant


def parse_summary_file(summary_path, subject):
    """
    Parses a summary text file to extract seizure time intervals for a specific subject.
    Returns a dictionary mapping filenames to lists of (start, end) tuples of a seizures.
    Example output:
    {'chb24_01.edf': [(480, 505), (2451, 2476)],'chb24_02.edf': [], 'chb24_03.edf': [(231, 260), (2883, 2908)], ...}
    """
    seizure_info = {}
    # Open the file and read its content
    with open(summary_path, "r") as f:
        content = f.read()

    # Split the content into blocks, each starting with "File Name: "
    file_blocks = content.split("File Name: ")
    for block in file_blocks:
        # Skip the block if it doesn't belong to the target subject (e.g., 'chb01')
        if subject not in block:
            continue

        # Extract the filename from the first line of the block
        filename = block.split("\n")[0].strip()

        # Use regex to find the total number of seizures reported in this file
        num_seizures = int(
            re.search(r"Number of Seizures in File: (\d+)", block).group(1)
        )

        intervals = []
        # If there are seizures, extract their start and end times
        if num_seizures > 0:
            # Seizure        -> literal string
            # (?:\s+\d+)?    -> optional non-capturing group for space and digits (e.g., " 1")
            # \s+Start Time: -> literal string with space
            # \s*(\d+)       -> captured digits (the actual timestamp)
            starts = re.findall(r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)", block)
            ends = re.findall(r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)", block)

            for s, e in zip(starts, ends):
                intervals.append((int(s), int(e)))

        seizure_info[filename] = intervals

    return seizure_info



def _safe_div(a, b, eps=1e-8):
    return a / (b + eps)


def _sample_entropy_safe(x):
    try:
        return ant.sample_entropy(x)
    except:
        return 0.0


def _perm_entropy_safe(x):
    try:
        return ant.perm_entropy(x, normalize=True)
    except:
        return 0.0


def _spectral_entropy_safe(psd_row):
    psd_norm = psd_row / (np.sum(psd_row) + 1e-8)
    psd_norm = psd_norm + 1e-12  
    return -np.sum(psd_norm * np.log(psd_norm))


def extract_node_features(window, fs):
    """
    window: (n_channels, n_samples)
    return: (n_channels, n_features)
    """

    n_channels, n_samples = window.shape

    # =========================
    # 1. Amplitude / energy features
    # =========================
    stds = np.std(window, axis=1)
    rmss = np.sqrt(np.mean(window**2, axis=1))

    d1 = np.diff(window, axis=1)
    line_length = np.sum(np.abs(d1), axis=1)

    # =========================
    # 2. Hjorth features
    # =========================
    d2 = np.diff(d1, axis=1)

    m0 = np.var(window, axis=1)
    m2 = np.var(d1, axis=1)
    m4 = np.var(d2, axis=1)

    mobility = np.sqrt(_safe_div(m2, m0))
    complexity = np.sqrt(_safe_div(m4, m2)) / (mobility + 1e-8)

    # =========================
    # 3. Statistics
    # =========================
    skews = skew(window, axis=1)
    kurts = kurtosis(window, axis=1)
    ptp = np.ptp(window, axis=1)

    # =========================
    # 4. Zero crossings
    # =========================
    zero_crossings = np.sum(np.diff(np.sign(window), axis=1) != 0, axis=1)

    # =========================
    # 5. Higuchi FD
    # =========================
    hfds = np.array([ant.higuchi_fd(ch) for ch in window])

    # =========================
    # 6. Teager Energy
    # =========================
    teo = window[:, 1:-1]**2 - window[:, :-2] * window[:, 2:]
    teo_mean = np.mean(teo, axis=1)

    # =========================
    # 7. PSD (Welch)
    # =========================
    freqs, psd = welch(window, fs, nperseg=fs, axis=1)

    total_power = np.trapz(psd, freqs, axis=1)

    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 12),
        'beta': (12, 30),
    }

    band_powers = []
    for low, high in bands.values():
        idx = (freqs >= low) & (freqs <= high)
        power = np.trapz(psd[:, idx], freqs[idx], axis=1)
        band_powers.append(power)

    band_powers = np.array(band_powers)  # (5, n_channels)

    # =========================
    # 8. Relative band powers
    # =========================
    rel_band_powers = _safe_div(band_powers, total_power)

    rel_delta, rel_theta, rel_alpha, rel_beta = rel_band_powers

    # =========================
    # 9. Ratio features
    # =========================
    theta_alpha_ratio = _safe_div(rel_theta, rel_alpha)
    delta_beta_ratio = _safe_div(rel_delta, rel_beta)

    # =========================
    # 10. Entropy
    # =========================
    # sampen = np.array([_sample_entropy_safe(ch) for ch in window])
    perm_entropy = np.array([_perm_entropy_safe(ch) for ch in window])
    spectral_entropy = np.array([
        _spectral_entropy_safe(psd[i]) for i in range(n_channels)
    ])

    # =========================
    # FINAL STACK
    # =========================
    features = np.column_stack([
        stds, rmss, line_length,
        mobility, complexity,
        skews, kurts, ptp,
        zero_crossings,
        hfds, teo_mean,
        rel_delta, rel_theta, rel_alpha, rel_beta,
        theta_alpha_ratio, delta_beta_ratio,
        perm_entropy, spectral_entropy
    ])

    return np.nan_to_num(features)


def compute_corr(window):
    corr = np.corrcoef(window)
    corr = np.nan_to_num(corr)
    corr_abs = np.abs(corr)
    np.fill_diagonal(corr_abs, 0)
    return corr_abs

def compute_coherence(window):
    """
    window: [N_channels, T]
    """
    fft = np.fft.rfft(window, axis=1)
    
    Sxy = fft[:, None, :] * np.conj(fft[None, :, :])
    Sxx = np.abs(fft[:, None, :])**2
    Syy = np.abs(fft[None, :, :])**2

    coh = np.abs(Sxy)**2 / (Sxx * Syy + 1e-8)

    # average coherence across frequencies
    coh = coh.mean(axis=2)

    np.fill_diagonal(coh, 0)
    return coh

def compute_plv_pli(window):
    """
    window: [N_channels, T]
    returns:
        plv: [N, N]
        pli: [N, N]
    """
    analytic_signal = hilbert(window, axis=1)
    phase_vectors = analytic_signal / np.abs(analytic_signal)
    n_samples = window.shape[1]

    # --- PLV ---
    plv = np.abs(np.dot(phase_vectors, phase_vectors.conj().T)) / n_samples

    # --- PLI ---
    # imag part = sin(delta_phase)
    imag_part = np.imag(
        phase_vectors[:, None, :] * np.conj(phase_vectors[None, :, :])
    )

    pli = np.abs(np.mean(np.sign(imag_part), axis=2))

    # --- cleanup ---
    np.fill_diagonal(plv, 0)
    np.fill_diagonal(pli, 0)

    return plv, pli


#=========================
# Artifacts visualization
#=========================
def save_artifact_plot(window, fs, channel_names, start_sec, reason, subject, filename, label, save_dir="artifact_plots"):
    """
    Creates a clear cascade plot of all EEG channels for a window with an artifact.
    """
    os.makedirs(save_dir, exist_ok=True)
    n_channels, n_samples = window.shape
    times = np.arange(n_samples) / fs
    
    # Offset between channels to avoid overlaping
    offset = np.max(np.abs(window)) * 0.8 if np.max(np.abs(window)) < 5000 else 1000
    
    plt.figure(figsize=(15, 10))
    for i in range(n_channels):
        plt.plot(times, window[i, :] - (i * offset), color='black', linewidth=0.7)
    
    plt.yticks([-i * offset for i in range(n_channels)], channel_names)
    plt.xlabel("Time in window [s]")
    plt.title(f"Artifact in {subject} | File: {filename} | Start: {start_sec:.1f}s\nReason: {reason} | Label: {label}")
    plt.grid(axis='x', linestyle='--', alpha=0.5)
    
    save_path = os.path.join(save_dir, f"{filename}_{start_sec:.1f}s_{reason.replace(' ', '_')}.png")
    plt.savefig(save_path, dpi=100)
    plt.close()
