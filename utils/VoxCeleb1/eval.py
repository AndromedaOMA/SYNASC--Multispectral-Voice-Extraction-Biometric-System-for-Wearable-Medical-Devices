import io
import torch
import torchaudio
import torchaudio.transforms as T
import matplotlib.pyplot as plt
import warnings
from collections import defaultdict
from datasets import load_dataset, config, Features, Value
from speechbrain.inference.speaker import SpeakerRecognition
from speechbrain.utils.metric_stats import EER
from FSPEN.utils.VoxCeleb1.efspen_pred import EFSPENEnhancer
from FSPEN.utils.eer_statistics import (
    build_speaker_scores,
    compare_eer,
    plot_eer_comparison,
    speechbrain_eer,
)

# 1. Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# --- 1. Global Configurations ---
config.HF_HUB_READ_TIMEOUT = 60
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Load Enhancer Predictor
predictor = EFSPENEnhancer("../../models/best_model_0.0613.pth", device=device)

# Load Baseline Verification Model (for unadapted enhanced audio)
baseline_verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
baseline_verification.eval()
for param in baseline_verification.parameters():
    param.requires_grad = False

# Load Adapted Verification Model (for EFSPEN enhanced audio)
adapted_verification = SpeakerRecognition.from_hparams(
    source="speechbrain/spkrec-ecapa-voxceleb",
    run_opts={"device": device}
)
# Load the newly fine-tuned weights
state_dict = torch.load("best_ecapa_adapted_efspen.pth", map_location=device, weights_only=True)
adapted_verification.mods.embedding_model.load_state_dict(state_dict)
adapted_verification.eval()
for param in adapted_verification.parameters():
    param.requires_grad = False

print("Baseline and Adapted models loaded successfully.")

# --- 2. Dataset Loading ---
custom_features = Features({
    "audio": {"bytes": Value("binary"), "path": Value("string")},
    "id": Value("string"),
    "speaker_id": Value("string")
})

ds = load_dataset(
    "asahi417/voxceleb1-test-split",
    split="test",
    streaming=True,
    features=custom_features
)

NUM_SPEAKERS = 25
SAMPLES_PER_SPK = 35
speaker_data = defaultdict(list)
found_speakers = []
resampler = None

print(f"Processing {NUM_SPEAKERS} speakers for 2-Way Evaluation...")

# --- 3. Processing Loop ---
try:
    with torch.inference_mode():
        for item in ds:
            spk_id = item.get('speaker_id')
            audio_feature = item.get('audio')

            if spk_id is None or audio_feature is None:
                continue

            if spk_id not in found_speakers:
                if len(found_speakers) < NUM_SPEAKERS:
                    found_speakers.append(spk_id)
                else:
                    continue

            if spk_id in found_speakers and len(speaker_data[spk_id]) < SAMPLES_PER_SPK:
                audio_bytes = audio_feature.get('bytes')
                if audio_bytes is None:
                    continue

                wav, original_sr = torchaudio.load(io.BytesIO(audio_bytes))

                if original_sr != 16000:
                    if resampler is None or resampler.orig_freq != original_sr:
                        resampler = T.Resample(orig_freq=original_sr, new_freq=16000)
                    wav = resampler(wav)

                wav = wav.to(device)

                target_len = 64000
                if wav.shape[1] > target_len:
                    wav = wav[:, :target_len]
                elif wav.shape[1] < target_len:
                    wav = torch.nn.functional.pad(wav, (0, target_len - wav.shape[1]))

                # EFSPEN Enhancement
                enhanced_wav = predictor.enhance_audio(wav)

                # --- 1. Unadapted Enhanced Pathway ---
                emb_enh_baseline = baseline_verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()

                # --- 2. Adapted Enhanced Pathway ---
                emb_enh_adapted = adapted_verification.encode_batch(enhanced_wav.to(device)).squeeze().cpu()

                # Store Normalized Embeddings
                speaker_data[spk_id].append({
                    "Unadapted Enhanced": torch.nn.functional.normalize(emb_enh_baseline, dim=-1),
                    "Adapted Enhanced": torch.nn.functional.normalize(emb_enh_adapted, dim=-1)
                })
                print(f"Captured: Spk {spk_id} | Sample {len(speaker_data[spk_id])}/{SAMPLES_PER_SPK}")
                if (
                    len(speaker_data) == NUM_SPEAKERS
                    and all(
                        len(samples) >= SAMPLES_PER_SPK
                        for samples in speaker_data.values()
                    )
                ):
                    break

except Exception as e:
    raise RuntimeError("VoxCeleb audio processing failed.") from e

# --- 4. EER Calculation & Final Results ---
if len(speaker_data) < 2:
    print(f"Insufficient data. Collected {len(speaker_data)} speakers.")
else:
    print("\nCalculating 2-Way EER Results...")
    # Adjusted figsize to 14x6 for a clean 2-chart layout
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle('VoxCeleb1 Performance Comparison', fontsize=16, fontweight='bold')

    results_summary = {}
    eer_score_sets = []
    modes = ["Unadapted Enhanced", "Adapted Enhanced"]

    for idx, mode in enumerate(modes):
        mode_speaker_map = {
            speaker_id: [(speaker_id, index) for index in range(len(samples))]
            for speaker_id, samples in speaker_data.items()
        }
        mode_embeddings = {
            (speaker_id, index): sample[mode]
            for speaker_id, samples in speaker_data.items()
            for index, sample in enumerate(samples)
        }
        speaker_scores = build_speaker_scores(
            mode_speaker_map,
            mode_embeddings,
            lambda first, second: torch.dot(first, second).item(),
            seed=0,
            impostor_sample=lambda files, _rng: files[0],
        )
        pos_scores, neg_scores = speaker_scores.scores_for(
            speaker_scores.speaker_ids
        )
        eer_score_sets.append((mode, speaker_scores))

        if pos_scores and neg_scores:
            eer, _ = EER(torch.tensor(pos_scores), torch.tensor(neg_scores))
            results_summary[mode] = eer * 100

            ax = axes[idx]
            ax.hist(neg_scores, bins=30, alpha=0.5, label='Impostors', color='red', density=True)
            ax.hist(pos_scores, bins=30, alpha=0.5, label='Targets', color='blue', density=True)
            ax.set_title(f"{mode}\nEER: {eer * 100:.2f}%")
            ax.set_xlabel("Cosine Similarity Score")
            ax.legend()

    print("\n" + "=" * 40)
    for k, v in results_summary.items():
        print(f"{k} EER: {v:.2f}%")
    print("=" * 40)

    if len(eer_score_sets) == 2:
        comparison = compare_eer(
            eer_score_sets[0][1],
            eer_score_sets[1][1],
            speechbrain_eer,
            (eer_score_sets[0][0], eer_score_sets[1][0]),
            seed=0,
            impostor_sample=lambda files, _rng: files[0],
        )
        for idx, (label, interval) in enumerate(
            zip(comparison.labels, comparison.ci_percent)
        ):
            print(
                f"{label} EER: {comparison.eer_percent[idx]:.2f}% "
                f"(95% speaker-bootstrap CI "
                f"{interval[0]:.2f} to {interval[1]:.2f}%)"
            )
            axes[idx].set_title(
                f"{label}\nEER: {comparison.eer_percent[idx]:.2f}% "
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