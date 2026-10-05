"""Canonical imports fail before writing data and remain usable after round trips."""
import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from football_profiler import storage as S
from football_profiler.importers import import_canonical


class CanonicalImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.data = Path(self.temp.name) / "data"
        data_patch = patch.object(S, "DATA", self.data)
        data_patch.start()
        self.addCleanup(data_patch.stop)
        self.tracks = pd.DataFrame({
            "frame": [0, 1, 2], "time_s": [0, 0.1, 0.2],
            "player_id": ["p1"] * 3, "x": [80, 81, 82], "y": [34, 35, 36],
            "detected": [1, 0, -1],
        })
        self.manifest = {
            "source": "Reviewed example tracking", "sampling_hz": 10,
            "players": [{"player_id": "p1", "eligible_frames": 5, "direction_known": True}],
        }

    def assert_invalid(self, tracks=None, manifest=None, message=None):
        with self.assertRaisesRegex(ValueError, message or ".+"):
            import_canonical("invalid", self.tracks if tracks is None else tracks,
                             self.manifest if manifest is None else manifest)
        self.assertFalse(self.data.exists(), "Invalid input created dataset files")

    def test_minimal_import_builds_profiles_and_complete_metadata(self):
        original = copy.deepcopy(self.manifest)
        result = import_canonical("minimal", self.tracks, self.manifest)
        self.assertEqual(self.manifest, original, "Import must not modify its input metadata")
        self.assertEqual(result["title"], "minimal")
        self.assertEqual(result["match_id"], "minimal")
        self.assertEqual(result["total_sampled_frames"], 5)
        self.assertEqual(result["duration_seconds"], 0.5)
        self.assertEqual(result["players"][0]["name"], "p1")
        self.assertEqual(result["players"][0]["team"], "Unconfirmed")
        self.assertTrue(result["players"][0]["direction_known"])
        profile = S.read_json(S.dataset_dir("minimal") / "profiles.json")[0]
        self.assertEqual(profile["valid_position_frames"], 1)
        self.assertAlmostEqual(profile["position_coverage"], 0.2)
        loaded = S.load_tracks("minimal")
        self.assertEqual(loaded.period.tolist(), [1, 1, 1])
        self.assertEqual(loaded.calibration_valid.tolist(), [1, 1, 1])

    def test_blank_display_metadata_has_safe_defaults(self):
        self.manifest.update(title=None, match_id=" ")
        self.manifest["players"][0].update(name=" ", team=None)
        result = import_canonical("display", self.tracks, self.manifest)
        self.assertEqual(result["title"], "display")
        self.assertEqual(result["match_id"], "display")
        self.assertEqual(result["players"][0]["name"], "p1")
        self.assertEqual(result["players"][0]["team"], "Unconfirmed")

    def test_missing_coordinates_and_out_of_pitch_positions_remain_auditable(self):
        self.tracks["detected"] = 1
        self.tracks["x"] = [np.nan, 200, 82]
        import_canonical("missing", self.tracks, self.manifest)
        profile = S.read_json(S.dataset_dir("missing") / "profiles.json")[0]
        self.assertEqual(profile["detected_frames"], 3)
        self.assertEqual(profile["coordinate_frames"], 2)
        self.assertEqual(profile["valid_position_frames"], 1)
        self.assertTrue(pd.isna(S.load_tracks("missing").iloc[0].x))

    def test_invalid_frames_and_times_do_not_write(self):
        for column, values in {
            "frame": [np.nan, np.inf, -1, 0.5, "bad", 2**53],
            "time_s": [np.nan, np.inf, -0.1, "bad"],
        }.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    tracks = self.tracks.copy()
                    tracks[column] = tracks[column].astype(object)
                    tracks.loc[0, column] = value
                    self.assert_invalid(tracks=tracks, message=column)

    def test_duplicates_checked_after_numeric_frame_conversion(self):
        self.tracks["frame"] = ["1", "1.0", "2"]
        self.assert_invalid(message="Duplicate player/frame")

    def test_nullable_numeric_missing_values_do_not_write(self):
        for column in ["frame", "time_s", "detected", "period", "calibration_valid"]:
            with self.subTest(column=column):
                tracks = self.tracks.copy()
                tracks[column] = pd.Series([1, pd.NA, 2], dtype="Int64")
                self.assert_invalid(tracks=tracks, message=column)

    def test_missing_and_duplicate_player_metadata_do_not_write(self):
        for players in [None, {}, [], ["p1"], [{}],
                        [{"player_id": "wrong", "eligible_frames": 5}],
                        [self.manifest["players"][0]] * 2]:
            with self.subTest(players=players):
                manifest = {**self.manifest, "players": players}
                self.assert_invalid(manifest=manifest)
        for value in [None, np.nan, "", "  "]:
            with self.subTest(player_id=value):
                tracks = self.tracks.copy()
                tracks.loc[0, "player_id"] = value
                self.assert_invalid(tracks=tracks, message="player_id")

    def test_numeric_and_text_player_ids_match_without_losing_leading_zeroes(self):
        self.tracks["player_id"] = [7, "7", "007"]
        self.manifest["players"] = [
            {"player_id": 7, "eligible_frames": 3},
            {"player_id": "007", "eligible_frames": 3},
        ]
        result = import_canonical("identifiers", self.tracks, self.manifest)
        self.assertEqual([p["player_id"] for p in result["players"]], ["7", "007"])
        self.assertEqual(S.load_tracks("identifiers").player_id.tolist(), ["7", "7", "007"])

    def test_invalid_manifest_and_sampling_rate_do_not_write(self):
        for manifest in [[], "invalid", 5]:
            with self.subTest(manifest=manifest):
                self.assert_invalid(manifest=manifest, message="JSON object")
        for field, values in {
            "sampling_hz": [None, 0, -1, np.inf, np.nan, "bad", [], True, 121],
            "source": [None, " ", 12, []],
            "duration_seconds": [None, 0, np.inf, np.nan, "bad", 0.1],
            "total_sampled_frames": [None, 0, True, 1.5, "5", 2, 10**1000],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    self.assert_invalid(manifest={**self.manifest, field: value})

    def test_invalid_player_denominators_and_flags_do_not_write(self):
        for field, values in {
            "eligible_frames": [None, 0, -1, True, 2, 3.5, "5", 10**1000],
            "playing_seconds": [None, 0, -1, np.nan, np.inf, "bad"],
            "direction_known": [None, "false", "true", 0, 1],
            "identity_verified": [None, "false", 0],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    manifest = copy.deepcopy(self.manifest)
                    manifest["players"][0][field] = value
                    self.assert_invalid(manifest=manifest, message=field)

    def test_invalid_coordinates_flags_and_periods_do_not_write(self):
        for column, values in {
            "x": [np.inf, -np.inf, "bad"], "y": [np.inf, "bad"],
            "detected": [None, np.inf, 2, "bad"],
            "calibration_valid": [None, -1, 2, "bad"],
            "period": [None, 0, -1, 1.5, np.inf, "bad"],
        }.items():
            for value in values:
                with self.subTest(column=column, value=value):
                    tracks = self.tracks.copy()
                    tracks[column] = pd.Series([value, value, value], dtype=object)
                    self.assert_invalid(tracks=tracks, message=column)

    def test_empty_or_incomplete_csv_does_not_write(self):
        self.assert_invalid(tracks=self.tracks.iloc[:0], message="at least one")
        self.assert_invalid(tracks=self.tracks.drop(columns="x"), message="CSV requires")

    def test_optional_orientation_is_validated_without_breaking_existing_imports(self):
        for value in [None, "left", "unknown", [], True]:
            with self.subTest(orientation=value):
                self.assert_invalid(manifest={**self.manifest, "coordinate_orientation": value},
                                    message="coordinate_orientation")
        self.assert_invalid(manifest={**self.manifest, "coordinate_orientation": "stadium"},
                            message="direction_known")
        self.manifest["coordinate_orientation"] = "attack_right"
        import_canonical("oriented", self.tracks, self.manifest)
        self.assertTrue(S.read_json(S.dataset_dir("oriented") / "profiles.json")[0]["direction_known"])

    def test_exported_manifest_round_trip_preserves_provenance(self):
        self.manifest.update(title="Example fixture", match_id="fixture-1", note="Human reviewed")
        first = import_canonical("first", self.tracks, self.manifest)
        second = import_canonical("second", S.load_tracks("first"), first)
        for field in ["title", "match_id", "note", "source", "players", "duration_seconds"]:
            self.assertEqual(first[field], second[field])
        self.assertEqual(second["id"], "second")


if __name__ == "__main__":
    unittest.main()
