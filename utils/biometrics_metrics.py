"""VoiceBankDEMAND"""

import os
import sys
import io
from collections import defaultdict
from unittest.mock import MagicMock

# 1. BYPASS SPEECHBRAIN 1.0 INTEGRATION BUGS
for module in ["k2", "flair"]:
    sys.modules[module] = MagicMock()

import torch
import torchaudio
import matplotlib.pyplot as plt
# Import 'Value' and 'Features' to strip Audio metadata
from datasets import load_dataset, Audio, Value, Features
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER
from eer_statistics import (
    build_speaker_scores,
    compare_eer,
    plot_eer_comparison,
    speechbrain_eer,
)

# Silence warnings
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define folders
enhanced_dir = "../evaluation/outputs/4sec_samples_mha/enhanced"

# Load Model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
verification.eval()
for param in verification.parameters():
    param.requires_grad = False

# Initialize the Plot
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.suptitle('VoiceBank-DEMAND dataset', fontsize=16, fontweight='bold')
results_summary = {}
eer_score_sets = []

# --- A. PROCESS NOISY (TRAINING DATASET - FIRST 2 SPEAKERS) ---
print("\nProcessing Noisy Training samples (First 2 Speakers)...")

# 1. Load the dataset
ds = load_dataset("JacobLinCool/VoiceBank-DEMAND-16k", split="train")

# 2. THE ULTIMATE FIX: Strip the 'Audio' type metadata completely
# This redefines the columns as simple nested dictionaries, so torchcodec is never called.
new_features = ds.features.copy()
new_features["noisy"] = {"bytes": Value("binary"), "path": Value("string")}
new_features["clean"] = {"bytes": Value("binary"), "path": Value("string")}
ds = ds.cast(new_features)

speaker_map_noisy = defaultdict(list)
embeddings_noisy = {}
found_speakers = []

for i in range(len(ds)):
    item = ds[i]
    speaker_id = item['id'].split('_')[0]

    if speaker_id not in found_speakers:
        if len(found_speakers) < 2:
            found_speakers.append(speaker_id)
            print(f"Found speaker: {speaker_id}")
        else:
            continue

    if speaker_id in found_speakers:
        # Load waveform from raw binary bytes
        raw_bytes = item['noisy']['bytes']
        signal, fs = torchaudio.load(io.BytesIO(raw_bytes))

        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            embeddings_noisy[item['id']] = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            speaker_map_noisy[speaker_id].append(item['id'])

# --- B. PROCESS ENHANCED (LOCAL FOLDER) ---
print(f"\nProcessing Enhanced samples from {enhanced_dir}...")
speaker_map_enh = defaultdict(list)
embeddings_enh = {}

if os.path.exists(enhanced_dir):
    all_enh_files = [f for f in os.listdir(enhanced_dir) if f.endswith('.wav')]
    for filename in all_enh_files:
        file_path = os.path.join(enhanced_dir, filename)
        signal, fs = torchaudio.load(file_path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            embeddings_enh[filename] = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()

            # Grouping logic: enhanced_{batch}_{idx}.wav -> use index as speaker ID
            parts = filename.split('_')
            spk_idx = parts[1]
            speaker_map_enh[spk_idx].append(filename)

# --- 5. GENERATE SCORES AND PLOT ---
eval_tasks = [
    ("Noisy", speaker_map_noisy, embeddings_noisy, 0),
    ("Enhanced", speaker_map_enh, embeddings_enh, 1)
]

for label, spk_map, emb_dict, plt_idx in eval_tasks:
    if len(spk_map) < 2:
        print(f"Skipping {label}: Need 2 speakers.")
        continue

    speaker_scores = build_speaker_scores(
        spk_map,
        emb_dict,
        lambda first, second: torch.dot(first, second).item(),
        seed=0,
    )
    pos_scores, neg_scores = speaker_scores.scores_for(speaker_scores.speaker_ids)
    eer_score_sets.append((label, speaker_scores, plt_idx))

    eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
    results_summary[label] = eer * 100

    ax = axes[plt_idx]
    ax.hist(neg_scores, bins=50, alpha=0.5, label='Impostors', color='red', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.5, label='Targets', color='blue', density=True)
    ax.set_title(f"{label}\nEER: {eer * 100:.2f}%")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')

print("\n" + "=" * 30)
for k, v in results_summary.items():
    print(f"{k} EER: {v:.2f}%")
print("=" * 30)

if len(eer_score_sets) == 2:
    comparison = compare_eer(
        eer_score_sets[0][1],
        eer_score_sets[1][1],
        speechbrain_eer,
        (eer_score_sets[0][0], eer_score_sets[1][0]),
        seed=0,
    )
    for index, (label, interval) in enumerate(
        zip(comparison.labels, comparison.ci_percent)
    ):
        print(
            f"{label} EER: {comparison.eer_percent[index]:.2f}% "
            f"(95% speaker-bootstrap CI {interval[0]:.2f} to {interval[1]:.2f}%)"
        )
        axes[eer_score_sets[index][2]].set_title(
            f"{label}\nEER: {comparison.eer_percent[index]:.2f}% "
            f"(95% CI {interval[0]:.2f}-{interval[1]:.2f}%)"
        )

    delta_low, delta_high = comparison.difference_ci_percent
    significance = "yes" if comparison.statistically_significant else "no"
    print(
        f"EER difference ({comparison.labels[1]} - {comparison.labels[0]}): "
        f"{comparison.difference_percent:.2f} percentage points "
        f"(95% speaker-bootstrap CI {delta_low:.2f} to {delta_high:.2f}); "
        f"significant at alpha=0.05: {significance}"
    )
    eer_fig, eer_axis = plt.subplots(figsize=(7, 3))
    plot_eer_comparison(eer_axis, comparison)
    eer_axis.set_title(
        f"EER comparison; difference 95% CI "
        f"[{delta_low:.2f}, {delta_high:.2f}] percentage points"
    )
    eer_fig.tight_layout()

plt.tight_layout()
plt.show()

"""LibriSpeech"""

import os
import torch
import torchaudio
import matplotlib.pyplot as plt
from collections import defaultdict
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER

eer_score_sets = []

# Paths
base_dir = "../evaluation/outputs/librispeech_mha"
# base_dir = "../evaluation/outputs/4sec_samples_mha"
noisy_dir = os.path.join(base_dir, "noisy")
enhanced_dir = os.path.join(base_dir, "enhanced")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
verification.eval()
for param in verification.parameters():
    param.requires_grad = False

def voicebank_get_embeddings(folder_path):
    speaker_map = defaultdict(list)
    emb_dict = {}
    files = [f for f in os.listdir(folder_path) if f.endswith('.wav')]

    print(f"Processing {len(files)} files from {folder_path}...")
    for filename in files:
        path = os.path.join(folder_path, filename)
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            # Normalize for cosine similarity
            norm_emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            emb_dict[filename] = norm_emb

            # Extract speaker ID (e.g., noisy_p232_001.wav -> p232)
            parts = filename.split('_')
            spk_id = parts[1]
            speaker_map[spk_id].append(filename)

    return speaker_map, emb_dict


def librespeech_get_embeddings(folder_path):
    speaker_map = defaultdict(list)
    emb_dict = {}
    files = [f for f in os.listdir(folder_path) if f.endswith(('.wav', '.flac'))]

    print(f"Processing {len(files)} files from {folder_path}...")
    for filename in files:
        path = os.path.join(folder_path, filename)
        signal, fs = torchaudio.load(path)
        if fs != 16000:
            signal = torchaudio.transforms.Resample(fs, 16000)(signal)

        with torch.no_grad():
            emb = verification.encode_batch(signal.to(device))
            norm_emb = torch.nn.functional.normalize(emb.squeeze(), dim=-1).cpu()
            emb_dict[filename] = norm_emb

            # LIBRISPEECH FIX: Split by '-' instead of '_'
            # Example: '1272-128104-0000.wav' -> parts[0] is '1272'
            parts = filename.split('-')
            if len(parts) > 1:
                spk_id = parts[0]  # For LibriSpeech, the first part is the Speaker ID
                speaker_map[spk_id].append(filename)
            else:
                # Fallback for other naming conventions like 'enhanced_p232_001.wav'
                parts_alt = filename.split('_')
                spk_id = parts_alt[1] if len(parts_alt) > 1 else "unknown"
                speaker_map[spk_id].append(filename)

    return speaker_map, emb_dict


# 1. Generate Embeddings
noisy_map, noisy_embs = librespeech_get_embeddings(noisy_dir)
enh_map, enh_embs = librespeech_get_embeddings(enhanced_dir)

# 2. Evaluation Loop
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
plt.suptitle('LibriSpeech dataset', fontsize=16, fontweight='bold')
eval_tasks = [
    ("Noisy Samples", noisy_map, noisy_embs, 0),
    ("Enhanced Samples", enh_map, enh_embs, 1)
]

for label, spk_map, emb_dict, plt_idx in eval_tasks:
    if len(spk_map) < 2:
        print(f"Skipping {label}: Need 2 speakers.")
        continue
    speaker_scores = build_speaker_scores(
        spk_map,
        emb_dict,
        lambda first, second: torch.dot(first, second).item(),
        seed=0,
    )
    pos_scores, neg_scores = speaker_scores.scores_for(speaker_scores.speaker_ids)
    eer_score_sets.append((label, speaker_scores, plt_idx))
    eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))

    ax = axes[plt_idx]
    ax.hist(neg_scores, bins=50, alpha=0.5, label='Impostors', color='red', density=True)
    ax.hist(pos_scores, bins=50, alpha=0.5, label='Targets', color='blue', density=True)
    ax.set_title(f"{label}\nEER: {eer * 100:.2f}%")
    ax.set_xlabel("Cosine Similarity Score")
    ax.set_ylabel("Probability Density")
    ax.legend(loc='upper right')

if len(eer_score_sets) == 2:
    comparison = compare_eer(
        eer_score_sets[0][1],
        eer_score_sets[1][1],
        speechbrain_eer,
        (eer_score_sets[0][0], eer_score_sets[1][0]),
        seed=0,
    )
    for index, (label, interval) in enumerate(
        zip(comparison.labels, comparison.ci_percent)
    ):
        print(
            f"{label} EER: {comparison.eer_percent[index]:.2f}% "
            f"(95% speaker-bootstrap CI {interval[0]:.2f} to {interval[1]:.2f}%)"
        )
        axes[eer_score_sets[index][2]].set_title(
            f"{label}\nEER: {comparison.eer_percent[index]:.2f}% "
            f"(95% CI {interval[0]:.2f}-{interval[1]:.2f}%)"
        )

    delta_low, delta_high = comparison.difference_ci_percent
    significance = "yes" if comparison.statistically_significant else "no"
    print(
        f"EER difference ({comparison.labels[1]} - {comparison.labels[0]}): "
        f"{comparison.difference_percent:.2f} percentage points "
        f"(95% speaker-bootstrap CI {delta_low:.2f} to {delta_high:.2f}); "
        f"significant at alpha=0.05: {significance}"
    )
    eer_fig, eer_axis = plt.subplots(figsize=(7, 3))
    plot_eer_comparison(eer_axis, comparison)
    eer_axis.set_title(
        f"EER comparison; difference 95% CI "
        f"[{delta_low:.2f}, {delta_high:.2f}] percentage points"
    )
    eer_fig.tight_layout()

plt.tight_layout()
plt.show()
