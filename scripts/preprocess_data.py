import os
import numpy as np
import mne
import torch
import gc
import pandas as pd 
import random
import warnings
from torch_geometric.data import Data
from tqdm import tqdm
from pathlib import Path
from typing import Union
from collections import Counter
from scipy.signal import savgol_filter

from src.config import PROCESSED_DIR, RAW_DIR, COMMON_CHANNELS
from src.preprocessing import parse_summary_file, extract_node_features, compute_plv_pli, compute_corr, compute_coherence


def set_seed(seed: int = 42):
    """
    Set the random seed for reproducibility.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seed set to: {seed}")

set_seed(42)


WINDOW_SIZE_SEC = 6
SAMPLING_RATE = 256             # Hz
SAMPLES_PER_WINDOW = WINDOW_SIZE_SEC * SAMPLING_RATE
OVERLAP_STEP_SEC = 0.5          # WINDOW_SIZE_SEC - OVERLAP_STEP_SEC = overlap of consecutive windows
LINE_FREQ = 60                  # Eventually 50 Hz depending on the region. CHB-MIT dataset is from USA, thus 60 Hz
PREICTAL_DURATION_SEC = 10 * 60 # For preictal labeling, we consider 10 minutes before the seizure onset as preictal period, however those labels was not directly used in this project (converted back again to 'ictal' label), but can be used for future work
BUFFER_SEC = 15                 # Safety buffer before and after ictal activity (based on article: https://doi.org/10.1109/ISBI56570.2024.10635821)
LABEL_INTERICTAL = 0
LABEL_ICTAL = 1
LABEL_PREICTAL = 2
MIN_VAR = 0.1                   # for dead channels detection and removal
TRIM_SEC = 10                   # initial and final seconds to trim from each recording, as they usually contain artifacts due to electrode placement and removal
DOWNSAMPLE_PROB = 0.7           # probability of skipping interictal windows to balance the dataset


# for cosmetic reasons, we can ignore plotting those warnings as they are addressed in our processing
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Channel names are not unique.*")      # duplicates later removed
warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*Scaling factor is not defined.*")     # appears for redundant channels, which are removed during processing


def process_data(subject: str, data_folder: Union[str, Path], summary_file: Union[str, Path]):
    """Processes raw .edf EEG recordings for a patient into a list of PyTorch Geometric graphs.

    Preprocessing pipeline:
      1. Channel standardization: Standardizes names and retains COMMON_CHANNELS.
      2. Filtering & smoothing: Applies Notch filter, 1-30 Hz bandpass, and Savitzky-Golay filter.
      3. Windowing & labeling: Extracts windows with overlap for ictal states, downsamples
         interictal periods, applies safety transition buffers, and filters low-variance/flat channels.
      4. Graph construction: Extracts node features, computes connectivity metrics (PLV, PLI)
         as edge attributes, and packages the data into torch_geometric.data.Data objects.
      5. Serialization: Saves the aggregated dataset to a .pt file.

    Args:
        subject: Patient identifier (e.g., 'chb01').
        data_folder: Directory containing the patient's raw .edf files.
        summary_file: Path to the dataset summary/annotation file with seizure intervals.

    Returns:
        None. Saves the processed graphs directly to PROCESSED_DIR / f"patient_{subject}.pt".
    """
    seizure_map = parse_summary_file(summary_file, subject)
    overlap_samples = int(OVERLAP_STEP_SEC * SAMPLING_RATE)
    data_list = []

    # --- DIAGNOSTICS ---
    stats = {
        'total_windows': 0,
        'skipped_artifact': 0,
        'skipped_buffer': 0,
        'labels': Counter()
    }


    for filename in tqdm(seizure_map.keys(), desc=f"EDF files of {subject}"):
        path = os.path.join(data_folder, filename)
        if not os.path.exists(path):
            continue

        raw = mne.io.read_raw_edf(path, preload=True, verbose=False)

        # bad channels (none of them are in COMMON_CHANNELS, so they will be ignored) 
        bads = raw.info['bads']
        if len(bads) > 0:
            print(f"Bad channels: {bads}")


        sfreq = raw.info['sfreq']
        if sfreq != SAMPLING_RATE:
            raise ValueError(f"Unexpected sampling rate {sfreq} Hz in file {filename}. Expected {SAMPLING_RATE} Hz.")


        # 1. CHANNELS STANDARDIZATION
        raw.rename_channels(lambda x: x.strip().upper())


        # Mapping for singular T8-P8, as is tends to be copied into T8-P8-0 and T8-P8-1 
        mapping = {'T8-P8-0': 'T8-P8'}
        if 'T8-P8-0' in raw.ch_names:
            raw.rename_channels(mapping)


        try:
            raw.pick(COMMON_CHANNELS)
        except ValueError as e:     
            print(f"Skipping file {filename}: missing required channels. Error: {e}")
            raw.close(); del raw; gc.collect()
            continue
        

        # 2. FILTERING AND SMOOTHING
        raw.notch_filter(freqs=LINE_FREQ, n_jobs=-1, verbose=False)
        raw.filter(1.0, 30, method='iir', n_jobs=-1, verbose=False)
        data = raw.get_data() * 1e6     # for better numerical stability, from Volts to microVolts (uV)

        # Savitzky-Golay filter parameters
        window_len = 11 
        poly_order = 3

        # smoothing
        data = savgol_filter(data, window_length=window_len, polyorder=poly_order, axis=-1)


        # 3. WINDOWING AND LABELING
        n_samples = data.shape[1]
        intervals = seizure_map[filename]

        trim_samples = TRIM_SEC * SAMPLING_RATE
        idx = trim_samples          # Ommiting the first TRIM_SEC seconds of the recording to avoid initial artifacts
        n_samples -= trim_samples   # Ommiting the last TRIM_SEC seconds of the recording to avoid final artifacts


        # windowing with overlap, labeling based on seizure intervals, and skipping windows based on buffer and artifact checks
        while idx + SAMPLES_PER_WINDOW <= n_samples:
            start_sec = idx / SAMPLING_RATE
            mid_sec = start_sec + WINDOW_SIZE_SEC / 2
            label = None

            is_seizure = any(s <= mid_sec <= e for s, e in intervals) # at least 50% of the window should be within the seizure interval to be labeled as ictal

            # labeling
            if is_seizure:  
                label = LABEL_ICTAL
            else:
                # Downsampling of interictal windows to balance the dataset
                a = np.random.rand() 
                if a < DOWNSAMPLE_PROB:
                    idx += SAMPLES_PER_WINDOW 
                    continue # Skip this window


                # Safety buffer
                is_in_buffer = any((s - BUFFER_SEC <= mid_sec <= s) or (e <= mid_sec <= e + BUFFER_SEC) 
                                   for s, e in intervals)
                
                if is_in_buffer:
                    label = None 
                    stats['skipped_buffer'] += 1
                    print(f"Skipping window in {filename} from {start_sec:.1f}s due to safety buffer.")
                else:
                    is_preictal = any(s - PREICTAL_DURATION_SEC <= mid_sec <= s - BUFFER_SEC 
                                      for s, e in intervals)
                    if is_preictal:
                        label = LABEL_PREICTAL
                    else:
                        label = LABEL_INTERICTAL


            # Dead channels detection and removal based on variance threshold
            window = data[:, idx:idx + SAMPLES_PER_WINDOW]

            is_flat = np.any(np.var(window, axis=1) < MIN_VAR)

            if label is None or is_flat:
                if label is not None:
                    stats['skipped_artifact'] += 1

                    print(f"Skipping window in {filename} from {start_sec:.1f}s due to flat-line. Label {label}")
                idx += overlap_samples if is_seizure else SAMPLES_PER_WINDOW
                continue


            # 4. FEATURE EXTRACTION AND GRAPH CREATION
            node_x = extract_node_features(window, SAMPLING_RATE)       # Calculating node features for each channel in the window

            # Edge features. For this project PLV (Phase Locking Value) was mainly used, as the rest were experimental.
            plv, pli = compute_plv_pli(window)
            # corr = compute_corr(window)
            # coherence = compute_coherence(window)

            N = plv.shape[0]

            row, col = np.where(~np.eye(N, dtype=bool))  # removing self-loops, which can be added later if needed

            edge_index = torch.tensor(
                np.vstack([row, col]),
                dtype=torch.long
            )

            edge_attr = np.stack([
                plv[row, col],
                pli[row, col],
                # corr[row, col],
                # coherence[row, col]
            ], axis=1)

            edge_attr = torch.tensor(edge_attr, dtype=torch.float)

            pyg_graph = Data(
                x=torch.from_numpy(node_x).float(),
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.tensor([label], dtype=torch.long)
            )
            data_list.append(pyg_graph)
            stats['labels'][label] += 1

            # window advancement logic based on label type: ictal -> overlap, preictal and interictal -> non-overlapping
            if label == LABEL_ICTAL:
                idx += overlap_samples
            elif label == LABEL_PREICTAL:
                idx += SAMPLES_PER_WINDOW
            else:
                idx += SAMPLES_PER_WINDOW

        raw.close(); del raw; del data; gc.collect()

    # --- 5. SANITY CHECK AND SAVING ---
    raw_features = torch.cat([g.x for g in data_list], dim=0)
    print(f"\n--- SANITY CHECK: {subject} ---")
    print(f"Rozkład etykiet: {dict(stats['labels'])}")
    print(f"Labels distribution: {dict(stats['labels'])}")
    print(f"Skipped windows (buffer): {stats['skipped_buffer']}")
    print(f"Skipped windows (artifacts): {stats['skipped_artifact']}")
    
    df_stats = pd.DataFrame({
        'Mean': raw_features.mean(dim=0).numpy(),
        'Std': raw_features.std(dim=0).numpy(),
        'Min': raw_features.min(dim=0).values.numpy(),
        'Max': raw_features.max(dim=0).values.numpy()
    })
    print("\nStatistics:")
    print(df_stats.to_string(index=False))


    torch.save(data_list, PROCESSED_DIR / f"patient_{subject}.pt")
    print(f"Saved {len(data_list)} graphs for patient {subject}.")



if __name__ == "__main__":
    subjects_to_process = [f"chb{str(i).zfill(2)}" for i in range(1, 25)]   

    for SUBJECT in tqdm(subjects_to_process):
        DATA_FOLDER = RAW_DIR / SUBJECT
        SUMMARY_FILE = DATA_FOLDER / f"{SUBJECT}-summary.txt"

        process_data(SUBJECT, DATA_FOLDER, SUMMARY_FILE)
        gc.collect()
