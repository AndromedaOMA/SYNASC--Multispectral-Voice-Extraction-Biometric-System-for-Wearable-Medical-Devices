"""Speaker-cluster bootstrap statistics for equal error rate (EER)."""

from dataclasses import dataclass
import random
from typing import Callable, Mapping, Sequence


Similarity = Callable[[object, object], float]
EERMetric = Callable[[Sequence[float], Sequence[float]], float]
ImpostorSample = Callable[[Sequence[object], random.Random], object]
ImpostorSample = Callable[[Sequence[object], random.Random], object]


@dataclass(frozen=True)
class SpeakerScores:
    speaker_ids: tuple[str, ...]
    target_scores: Mapping[str, tuple[float, ...]]
    impostor_scores: Mapping[tuple[str, str], tuple[float, ...]]

    def scores_for(self, speaker_ids: Sequence[str]) -> tuple[list[float], list[float]]:
        targets = [
            score
            for speaker_id in speaker_ids
            for score in self.target_scores.get(speaker_id, ())
        ]
        impostors = []
        for index, first in enumerate(speaker_ids):
            for second in speaker_ids[index + 1:]:
                if first == second:
                    continue
                pair = tuple(sorted((first, second)))
                impostors.extend(self.impostor_scores.get(pair, ()))
        if not targets or not impostors:
            raise ValueError("At least one target and one impostor score are required.")
        return targets, impostors


@dataclass(frozen=True)
class EERComparison:
    labels: tuple[str, str]
    eer_percent: tuple[float, float]
    ci_percent: tuple[tuple[float, float], tuple[float, float]]
    difference_percent: float
    difference_ci_percent: tuple[float, float]
    statistically_significant: bool


def build_speaker_scores(
    speaker_map: Mapping[str, Sequence[object]],
    embeddings: Mapping[object, object],
    similarity: Similarity,
    seed: int = 0,
    impostor_sample: ImpostorSample | None = None,
) -> SpeakerScores:
    speaker_ids = tuple(sorted(speaker_map))
    if len(speaker_ids) < 2:
        raise ValueError("EER requires at least two speakers.")

    targets: dict[str, tuple[float, ...]] = {}
    for speaker_id in speaker_ids:
        files = speaker_map[speaker_id]
        targets[speaker_id] = tuple(
            similarity(embeddings[files[i]], embeddings[files[j]])
            for i in range(len(files))
            for j in range(i + 1, len(files))
        )

    rng = random.Random(seed)
    selector = impostor_sample or (
        lambda files, generator: generator.choice(files)
    )
    impostors: dict[tuple[str, str], tuple[float, ...]] = {}
    for index, first in enumerate(speaker_ids):
        for second in speaker_ids[index + 1:]:
            if speaker_map[first] and speaker_map[second]:
                pair = tuple(sorted((first, second)))
                first_file = selector(speaker_map[first], rng)
                second_file = selector(speaker_map[second], rng)
                impostors[pair] = (
                    similarity(embeddings[first_file], embeddings[second_file]),
                )

    scores = SpeakerScores(speaker_ids, targets, impostors)
    scores.scores_for(speaker_ids)
    return scores


def compare_eer(
    first: SpeakerScores,
    second: SpeakerScores,
    eer_metric: EERMetric,
    labels: tuple[str, str],
    bootstrap_replicates: int = 2000,
    seed: int = 0,
) -> EERComparison:
    """Compare EERs with a paired speaker bootstrap where speaker IDs overlap."""
    if bootstrap_replicates < 100:
        raise ValueError("At least 100 bootstrap replicates are required.")

    first_targets, first_impostors = first.scores_for(first.speaker_ids)
    second_targets, second_impostors = second.scores_for(second.speaker_ids)
    first_eer = eer_metric(first_targets, first_impostors) * 100
    second_eer = eer_metric(second_targets, second_impostors) * 100

    common_ids = sorted(set(first.speaker_ids) & set(second.speaker_ids))
    first_only = sorted(set(first.speaker_ids) - set(common_ids))
    second_only = sorted(set(second.speaker_ids) - set(common_ids))
    rng = random.Random(seed)
    first_bootstrap: list[float] = []
    second_bootstrap: list[float] = []
    difference_bootstrap: list[float] = []
    max_attempts = bootstrap_replicates * 50

    for _ in range(max_attempts):
        shared_draw = rng.choices(common_ids, k=len(common_ids)) if common_ids else []
        first_sample = shared_draw.copy()
        second_sample = shared_draw.copy()
        if first_only:
            first_sample.extend(rng.choices(first_only, k=len(first_only)))
        if second_only:
            second_sample.extend(rng.choices(second_only, k=len(second_only)))
        try:
            first_pos, first_neg = first.scores_for(first_sample)
            second_pos, second_neg = second.scores_for(second_sample)
        except ValueError:
            continue

        first_value = eer_metric(first_pos, first_neg) * 100
        second_value = eer_metric(second_pos, second_neg) * 100
        first_bootstrap.append(first_value)
        second_bootstrap.append(second_value)
        difference_bootstrap.append(second_value - first_value)
        if len(difference_bootstrap) == bootstrap_replicates:
            break

    if len(difference_bootstrap) < bootstrap_replicates:
        raise RuntimeError(
            "Could not produce enough valid speaker-cluster bootstrap replicates."
        )

    first_ci = _percentile_interval(first_bootstrap)
    second_ci = _percentile_interval(second_bootstrap)
    difference_ci = _percentile_interval(difference_bootstrap)
    difference = second_eer - first_eer
    return EERComparison(
        labels=labels,
        eer_percent=(first_eer, second_eer),
        ci_percent=(first_ci, second_ci),
        difference_percent=difference,
        difference_ci_percent=difference_ci,
        statistically_significant=difference_ci[0] > 0 or difference_ci[1] < 0,
    )


def speechbrain_eer(
    positive_scores: Sequence[float], negative_scores: Sequence[float]
) -> float:
    """Vectorized equivalent of SpeechBrain's threshold-based EER procedure."""
    import numpy as np

    positive = np.asarray(positive_scores, dtype=np.float32)
    negative = np.asarray(negative_scores, dtype=np.float32)
    unique_scores = np.unique(np.concatenate((positive, negative)))
    thresholds = np.sort(
        np.concatenate((
            unique_scores,
            (unique_scores[:-1] + unique_scores[1:]) / np.float32(2),
        ))
    )
    positive_sorted = np.sort(positive)
    negative_sorted = np.sort(negative)
    false_reject_rates = (
        np.searchsorted(positive_sorted, thresholds, side="right").astype(np.float32)
        / np.float32(positive.size)
    )
    false_accept_rates = (
        (negative.size - np.searchsorted(negative_sorted, thresholds, side="right"))
        .astype(np.float32)
        / np.float32(negative.size)
    )
    closest_threshold = np.argmin(
        np.abs(false_accept_rates - false_reject_rates)
    )
    return float(
        (false_accept_rates[closest_threshold] + false_reject_rates[closest_threshold])
        / 2
    )


def plot_eer_comparison(axis: object, comparison: EERComparison) -> None:
    interval_centers = [
        (interval[0] + interval[1]) / 2 for interval in comparison.ci_percent
    ]
    lower_errors = [
        center - interval[0]
        for center, interval in zip(interval_centers, comparison.ci_percent)
    ]
    upper_errors = [
        interval[1] - center
        for center, interval in zip(interval_centers, comparison.ci_percent)
    ]
    positions = range(len(comparison.labels))
    axis.errorbar(
        interval_centers,
        positions,
        xerr=[lower_errors, upper_errors],
        fmt="none",
        capsize=5,
        label="95% CI",
    )
    axis.scatter(comparison.eer_percent, list(positions), label="EER estimate")
    axis.set_yticks(list(positions))
    axis.set_yticklabels(comparison.labels)
    axis.set_xlabel("EER (%)")
    axis.set_title("EER with 95% speaker-bootstrap CI")
    axis.grid(axis="x", alpha=0.3)
    axis.legend()


def _percentile_interval(values: Sequence[float]) -> tuple[float, float]:
    ordered = sorted(values)
    return _percentile(ordered, 0.025), _percentile(ordered, 0.975)


def _percentile(ordered: Sequence[float], probability: float) -> float:
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction
