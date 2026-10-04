import unittest

from FSPEN.utils.eer_statistics import (
    SpeakerScores,
    build_speaker_scores,
    compare_eer,
)


def mean_based_eer(targets, impostors):
    return sum(impostors) / len(impostors) / (
        sum(targets) / len(targets) + sum(impostors) / len(impostors)
    )


class EERStatisticsTests(unittest.TestCase):
    def setUp(self):
        self.speaker_map = {
            "speaker-a": ["a1", "a2"],
            "speaker-b": ["b1", "b2"],
            "speaker-c": ["c1", "c2"],
        }
        self.embeddings = {
            "a1": 0.9,
            "a2": 0.8,
            "b1": 0.7,
            "b2": 0.6,
            "c1": 0.5,
            "c2": 0.4,
        }

    def test_bootstrap_comparison_is_reproducible_and_clustered(self):
        scores = build_speaker_scores(
            self.speaker_map,
            self.embeddings,
            lambda first, second: first * second,
            seed=7,
        )
        comparison = compare_eer(
            scores,
            scores,
            mean_based_eer,
            ("Original", "Enhanced"),
            bootstrap_replicates=100,
            seed=12,
        )
        repeated = compare_eer(
            scores,
            scores,
            mean_based_eer,
            ("Original", "Enhanced"),
            bootstrap_replicates=100,
            seed=12,
        )

        self.assertEqual(comparison, repeated)
        self.assertEqual(comparison.difference_percent, 0)
        self.assertFalse(comparison.statistically_significant)
        self.assertLessEqual(comparison.ci_percent[0][0], comparison.eer_percent[0])
        self.assertGreaterEqual(comparison.ci_percent[0][1], comparison.eer_percent[0])

    def test_bootstrap_rejects_too_few_replicates(self):
        scores = build_speaker_scores(
            self.speaker_map,
            self.embeddings,
            lambda first, second: first * second,
        )
        with self.assertRaises(ValueError):
            compare_eer(
                scores,
                scores,
                mean_based_eer,
                ("Original", "Enhanced"),
                bootstrap_replicates=99,
            )

    def test_difference_ci_controls_significance_decision(self):
        speaker_ids = ("speaker-a", "speaker-b", "speaker-c")
        impostor_scores = {
            ("speaker-a", "speaker-b"): (0.5,),
            ("speaker-a", "speaker-c"): (0.5,),
            ("speaker-b", "speaker-c"): (0.5,),
        }
        first = SpeakerScores(
            speaker_ids,
            {speaker_id: (0.2,) for speaker_id in speaker_ids},
            impostor_scores,
        )
        second = SpeakerScores(
            speaker_ids,
            {speaker_id: (0.4,) for speaker_id in speaker_ids},
            impostor_scores,
        )
        comparison = compare_eer(
            first,
            second,
            lambda targets, _impostors: targets[0],
            ("Original", "Enhanced"),
            bootstrap_replicates=100,
        )

        self.assertEqual(comparison.difference_ci_percent, (20, 20))
        self.assertTrue(comparison.statistically_significant)


if __name__ == "__main__":
    unittest.main()
