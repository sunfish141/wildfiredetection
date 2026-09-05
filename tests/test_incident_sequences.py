import unittest
from dataclasses import replace

import pandas as pd
from pyproj import Transformer

from wildfire_data.incident_sequences import (IncidentPolicy, associate_incidents,
    build_incident_sequences, feds_incident_key, stable_fraction, region_is_held)


def record(timestamp, fire_id, x, y=0):
    inverse = Transformer.from_crs("ESRI:102008", "EPSG:4326", always_xy=True)
    ring = [list(inverse.transform(px, py)) for px, py in
            ((x+250,y+250), (x+750,y+250), (x+750,y+750), (x+250,y+750), (x+250,y+250))]
    return {"region": "CONUS", "fire_id": fire_id, "source_snapshot_time": timestamp.isoformat(),
            "source_record_id": f"CONUS|{fire_id}|{timestamp.isoformat()}", "raw_artifact_id": "raw-id",
            "geometry": {"encoding": "esri-rings-wgs84/v1", "rings": [ring]}}


def examples():
    rows = []
    for step in (0, 1, 3):
        t = pd.Timestamp("2026-07-01T00:00:00Z") + pd.Timedelta(hours=step*12)
        for x in ([0, 10, 100, 1000] if step < 3 else [0]):
            rows.append({"example_id": f"e-{step}-{x}", "cell_id": f"naea-1km:x={x}:y=0",
                         "source_snapshot_time": t.isoformat(), "target_snapshot_time": None,
                         "anchor_at": t.isoformat(), "target_end_at": (t+pd.Timedelta(hours=12)).isoformat(),
                         "dataset_split": "train", "target_newly_burned_12h": int(x == 0)})
    return pd.DataFrame(rows)


class IncidentSequenceTests(unittest.TestCase):
    def policy(self):
        return IncidentPolicy(region_holdout_fraction=0., incident_holdout_fraction=0.,
                              calibration_fraction=0., later_test_at="2026-07-02T00:00:00Z")

    def loader(self, timestamp):
        return [record(timestamp, 1, 0), record(timestamp, 2, 10000), record(timestamp, 3, 100000)]

    def test_provider_identity_normalization_and_year_scope(self):
        a = record(pd.Timestamp("2026-07-01T00:00:00Z"), 42, 0)
        b = {**a, "fire_id": "42.0"}
        self.assertEqual(feds_incident_key(a), feds_incident_key(b))
        self.assertNotEqual(feds_incident_key(a), feds_incident_key({**a, "source_snapshot_time": "2027-07-01T00:00:00Z"}))

    def test_nearby_fires_stay_together_and_later_incidents_are_fully_withheld(self):
        frame = examples()
        annotations, manifest = associate_incidents(frame, self.loader, policy=self.policy())
        joined = frame.merge(annotations, on="example_id")
        close = joined.loc[joined.cell_id.isin(["naea-1km:x=0:y=0", "naea-1km:x=10:y=0"])]
        self.assertEqual(close.incident_group_id.nunique(), 1)
        self.assertEqual(set(close.incident_split), {"later_time"})
        self.assertEqual(set(joined.loc[joined.cell_id.eq("naea-1km:x=100:y=0"), "incident_split"]), {"train"})
        self.assertEqual(set(joined.loc[joined.cell_id.eq("naea-1km:x=1000:y=0"), "incident_split"]), {"unassigned"})
        self.assertEqual(len(manifest["groups"]), 2)

    def test_sequences_break_on_gaps_and_never_include_pretest_context_as_test_rows(self):
        frame = examples()
        annotation, _ = associate_incidents(frame, self.loader, policy=self.policy())
        joined = frame.merge(annotation, on="example_id")
        sequences = build_incident_sequences(joined, later_test_at=self.policy().later_test_at)
        later = [s for s in sequences if s.split == "later_time"]
        self.assertEqual(len(later), 1)
        self.assertEqual(len(later[0].snapshots), 1)
        # With the cutoff beyond the data, the same incident's missing step
        # separates its sequence into fragments of two and one snapshots.
        annotation, _ = associate_incidents(frame, self.loader,
            policy=replace(self.policy(), later_test_at="2026-08-01T00:00:00Z"))
        joined = frame.merge(annotation, on="example_id")
        sequence = build_incident_sequences(joined, later_test_at="2026-08-01T00:00:00Z")
        self.assertEqual(sorted(len(s.snapshots) for s in sequence), [1, 2, 2])

    def test_ambiguous_revisions_and_cross_incident_split_changes_fail(self):
        frame = examples()
        with self.assertRaisesRegex(ValueError, "revision"):
            associate_incidents(frame, lambda t: self.loader(t) * 2, policy=self.policy())
        annotation, _ = associate_incidents(frame, self.loader, policy=self.policy())
        joined = frame.merge(annotation, on="example_id")
        joined.loc[0, "incident_split"] = "train"
        with self.assertRaisesRegex(ValueError, "one split"):
            build_incident_sequences(joined, later_test_at=self.policy().later_test_at)

    def test_group_identity_and_split_are_independent_of_row_order_and_targets(self):
        frame = examples()
        a, _ = associate_incidents(frame, self.loader, policy=self.policy())
        shuffled = frame.iloc[::-1].copy()
        shuffled.target_newly_burned_12h = 1 - shuffled.target_newly_burned_12h
        b, _ = associate_incidents(shuffled, self.loader, policy=self.policy())
        pd.testing.assert_frame_equal(a.sort_values("example_id").reset_index(drop=True),
                                      b.sort_values("example_id").reset_index(drop=True))

    def test_region_holdout_includes_the_feature_halo(self):
        seed = next(str(i) for i in range(100)
                    if stable_fraction(f"{i}:region:albers-1000km:x=1:y=0") < .5
                    and stable_fraction(f"{i}:region:albers-1000km:x=0:y=0") >= .5)
        policy = replace(self.policy(), seed=seed, region_holdout_fraction=.5)
        self.assertFalse(region_is_held("naea-1km:x=998:y=500", policy))
        self.assertTrue(region_is_held("naea-1km:x=999:y=500", policy))
        frame = examples().iloc[:1].copy()
        frame.cell_id = "naea-1km:x=999:y=500"
        annotation, _ = associate_incidents(frame,
            lambda t: [record(t, 1, 999000, 500000)], policy=policy)
        self.assertEqual(annotation.incident_split.tolist(), ["held_region"])


if __name__ == "__main__":
    unittest.main()
