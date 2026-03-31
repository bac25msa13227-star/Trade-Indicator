from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from xauusd_ai.infra.storage import StorageClient


class StorageClientTests(unittest.TestCase):
    def test_local_only_mode_is_noop_and_returns_object_names(self) -> None:
        with patch("xauusd_ai.infra.storage._MINIO_AVAILABLE", False):
            client = StorageClient()
            with tempfile.TemporaryDirectory() as td:
                p = Path(td) / "artifact.bin"
                p.write_bytes(b"abc")
                model_obj = client.upload_model("acc2", "v1", p)
                ds_obj = client.upload_dataset("acc2", "train.csv", p)
                rep_obj = client.upload_report("acc2", "report.json", {"ok": True})
                self.assertEqual(model_obj, "acc2/v1/artifact.bin")
                self.assertEqual(ds_obj, "acc2/train.csv")
                self.assertEqual(rep_obj, "acc2/report.json")
                self.assertEqual(client.get_report_url("acc2", "report.json"), "")


if __name__ == "__main__":
    unittest.main()
