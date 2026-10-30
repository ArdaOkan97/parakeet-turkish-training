import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from retention_guardian import Retention, atomic_json, encoded, pin


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="retention_test_",
            dir="E:/gpu_training_scratchpad/scratch")
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        self.guardian = Retention(self.run, self.root / "protected")

    def tearDown(self):
        if not self.root.resolve().is_relative_to(Path("E:/gpu_training_scratchpad/scratch").resolve()):
            raise RuntimeError("Refusing cleanup outside the designated fixture directory")
        self.tmp.cleanup()

    def checkpoint(self, update, payload=b"original weights plus Adam and RNG" * 100):
        folder = self.guardian.checkpoint_folder(update)
        folder.mkdir(parents=True)
        payload_path = folder / "checkpoint.tensors.bin"
        payload_path.write_bytes(payload)
        payload_pin = pin(payload_path, payload)
        state = {"weights": 207, "adam": 215, "rng": 4, "update": update,
                 "identity": {"original_path_inside_state": str(folder / "pre_save.json")}}
        checkpoint = dict(kind="test", payload=payload_pin, tree=state)
        atomic_json(folder / "checkpoint.json", checkpoint)
        checkpoint_pin = pin(folder / "checkpoint.json", encoded(checkpoint))
        pointer = dict(committed_updates=update, diagnostic_only=False,
            saved_semantic_state_verified=True, writer_returned_and_payload_verified=True,
            checkpoint=checkpoint_pin, payload=payload_pin, identity=state["identity"])
        atomic_json(folder / "pointer.json", pointer)
        atomic_json(folder / "persistence_verified.json", dict(passed=True,
            optimizer_updates=update, weights_verified=207, Adam_states_verified=215,
            named_tensors_verified=854, checkpoint=checkpoint_pin, payload=payload_pin))
        atomic_json(folder / "pre_save.json", state)
        atomic_json(folder / "next_draw.json", {"draw": update + 1})
        (folder / "prepared_metadata.json.gz").write_bytes(b"gzip companion fixture")
        return folder, checkpoint, pointer

    @staticmethod
    def score(errors):
        return dict(word_errors={"tr": errors}, reference_words={"tr": 825},
                    strict_all_six_passed=False)

    def test_best_periodic_survives_latest_five_pruning_without_retention_pass(self):
        folder, checkpoint, pointer = self.checkpoint(384000)
        self.guardian.preserve(384000)
        self.guardian.record_score(384000, self.score(178))
        for update in range(384200, 385201, 200):
            self.checkpoint(update)
        (folder / "checkpoint.tensors.bin").unlink()
        best = self.guardian.load("best_retained.json", None)
        self.assertEqual(best["update"], 384000)
        self.assertFalse(best["all_six_passed"])
        archive = self.root / "protected/update_000384000"
        relocated = json.loads((archive / "checkpoint.json").read_bytes())
        portable = json.loads((archive / "pointer.json").read_bytes())
        data = Path(relocated["payload"]["path"]).read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), checkpoint["payload"]["sha256"])
        self.assertEqual(relocated["tree"], checkpoint["tree"])
        self.assertEqual(portable["identity"], pointer["identity"])
        self.assertEqual(pin(archive / "checkpoint.json", (archive / "checkpoint.json").read_bytes()),
                         portable["checkpoint"])

    def test_worse_score_does_not_replace_best_and_lower_score_does(self):
        for update, errors in ((384000, 178), (388000, 180), (392000, 170)):
            self.checkpoint(update)
            self.guardian.preserve(update)
            self.guardian.record_score(update, self.score(errors))
            expected = 392000 if update == 392000 else 384000
            self.assertEqual(self.guardian.load("best_retained.json", None)["update"], expected)

    def test_corrupt_copy_never_publishes_protected_checkpoint(self):
        folder, _, _ = self.checkpoint(384000)
        (folder / "checkpoint.tensors.bin").write_bytes(b"bad")
        with self.assertRaises(RuntimeError):
            self.guardian.preserve(384000)
        self.assertFalse((self.root / "protected/update_000384000").exists())
        self.assertNotIn("384000", self.guardian.preserved)

    def test_missing_historical_best_is_reported_not_fabricated(self):
        self.guardian.record_score(384000, self.score(178))
        observed = self.guardian.load("best_observed.json", None)
        self.assertFalse(observed["protected_copy_exists"])
        self.assertIsNone(self.guardian.load("best_retained.json", None))

    def test_restart_retains_best_and_archive_is_outside_pruning_tree(self):
        self.checkpoint(388000)
        self.guardian.preserve(388000)
        self.guardian.record_score(388000, self.score(180))
        restarted = Retention(self.run, self.root / "protected")
        self.assertEqual(restarted.load("best_retained.json", None)["update"], 388000)
        with self.assertRaises(ValueError):
            Retention(self.run, self.run / "checkpoints/protected")


if __name__ == "__main__":
    unittest.main(verbosity=2)
